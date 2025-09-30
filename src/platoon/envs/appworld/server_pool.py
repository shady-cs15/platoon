import subprocess
import socket
import atexit
import signal
import sys
import psutil
import time
import requests
from typing import Dict, List, Set

type Server = str

class ServerPool:
    def __init__(self):
        self.available_servers = []
        self.busy_servers = []
        # Track processes for each server
        self.server_processes: Dict[Server, subprocess.Popen] = {}
        # Track ports that are allocated but server may not be ready yet
        self.allocated_ports: Set[int] = set()
        
        # Register cleanup handlers
        atexit.register(self.shutdown_all_servers)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        print(f"\nReceived signal {signum}, shutting down servers...")
        self.shutdown_all_servers()
        sys.exit(0)

    @staticmethod
    def kill_orphaned_servers() -> None:
        """Kill any orphaned server processes from previous runs."""
        killed_count = 0
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if (cmdline and 
                        len(cmdline) >= 3 and 
                        'python' in cmdline[0] and 
                        cmdline[1] == '-m' and 
                        cmdline[2] == 'platoon.envs.appworld.env_server'):
                        
                        print(f"Killing orphaned server process PID {proc.info['pid']}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            proc.kill()
                        killed_count += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process disappeared or no access, skip
                    continue
                    
        except Exception as e:
            print(f"Error while killing orphaned servers: {e}")
            
        if killed_count > 0:
            print(f"Killed {killed_count} orphaned server process(es)")
        else:
            print("No orphaned server processes found")

    def _find_available_port(self) -> int:
        """Find an available port, avoiding already allocated ones."""
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    port = s.getsockname()[1]
                    
                    # Check if port is already allocated by us
                    if port not in self.allocated_ports:
                        return port
                        
            except OSError:
                continue
                
        raise RuntimeError("No available ports found after maximum attempts")

    def _wait_for_server_ready(self, server_url: str, timeout: int = 30) -> bool:
        """Wait for server to be ready to accept connections."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to connect to the server
                response = requests.get(f"{server_url}/health", timeout=1)
                if response.status_code == 200:
                    return True
            except (requests.exceptions.RequestException, requests.exceptions.Timeout):
                # Server not ready yet, wait a bit
                time.sleep(0.1)
                continue
                
        return False

    def provision_server(self) -> Server:
        """Provision a new server."""
        max_retries = 5
        
        for attempt in range(max_retries):
            port = self._find_available_port()
            
            # Reserve this port
            self.allocated_ports.add(port)
            
            try:
                # Run the appworld serve environment command
                process = subprocess.Popen(
                    ["python", "-m", "platoon.envs.appworld.env_server", "--port", str(port)],
                    stdin=subprocess.DEVNULL,              
                    stdout=subprocess.DEVNULL,             
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True
                )
                
                server = f"http://localhost:{port}"
                
                # Wait for server to actually start and be ready
                if self._wait_for_server_ready(server, timeout=10):
                    self.available_servers.append(server)
                    self.server_processes[server] = process
                    return server
                else:
                    # Server failed to start, clean up
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    self.allocated_ports.remove(port)
                    
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Server failed to start on port {port} after {max_retries} attempts")
                    
            except Exception as e:
                # Clean up allocated port on any error
                self.allocated_ports.discard(port)
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to provision server: {e}")
                    
        raise RuntimeError("Failed to provision server after maximum retries")

    def request_server(self, add_new_if_unavailable: bool = False, max_allowed: int = -1) -> Server:
        """Request a server from the pool."""
        if not self.available_servers:
            if max_allowed > 0 and len(self.busy_servers) >= max_allowed:
                raise ValueError("Max allowed servers reached")
            if add_new_if_unavailable:
                server = self.provision_server()
                self.available_servers.remove(server)
                self.busy_servers.append(server)
                return server
            else:
                raise ValueError("No available servers")
        server = self.available_servers.pop(0)
        self.busy_servers.append(server)
        return server

    def release_server(self, server: Server) -> None:
        """Release a server back to the pool."""
        self.available_servers.append(server)
        self.busy_servers.remove(server)

    def shutdown_server(self, server: Server) -> None:
        """Shutdown a specific server."""
        if server in self.server_processes:
            process = self.server_processes[server]
            try:
                process.terminate()
                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't shut down gracefully
                    process.kill()
                    process.wait()
                print(f"Shut down server: {server}")
            except Exception as e:
                print(f"Error shutting down server {server}: {e}")
            finally:
                del self.server_processes[server]
                
        # Remove from server lists
        if server in self.available_servers:
            self.available_servers.remove(server)
        if server in self.busy_servers:
            self.busy_servers.remove(server)
            
        # Clean up allocated port
        try:
            port = int(server.split(':')[-1])
            self.allocated_ports.discard(port)
        except (ValueError, IndexError):
            pass  # Couldn't parse port from server URL

    def shutdown_all_servers(self) -> None:
        """Shutdown all running servers."""
        if not self.server_processes:
            return
            
        print(f"Shutting down {len(self.server_processes)} servers...")
        servers_to_shutdown = list(self.server_processes.keys())
        
        for server in servers_to_shutdown:
            self.shutdown_server(server)
        
        print("All servers shut down.")

    def get_available_servers(self) -> List[Server]:
        """Get all available servers in the pool."""
        return self.available_servers

    def get_busy_servers(self) -> List[Server]:
        """Get all busy servers in the pool."""
        return self.busy_servers

    def get_running_servers(self) -> List[Server]:
        """Get all running servers (both available and busy)."""
        return list(self.server_processes.keys())

SERVER_POOL = ServerPool()