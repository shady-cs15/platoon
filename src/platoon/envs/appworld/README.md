```
UV_GIT_LFS=1 uv sync --group appworld
uv run appworld install
export export APPWORLD_ROOT=src/platoon/envs/appworld
uv run appworld download --root ${APPWORLD_ROOT} data
```