from __future__ import annotations

from areal.experimental.openai.proxy import (
    AReaLEndSessionRequest,
    ProxySession,
    _post_json_with_retry,
)


class ArealProxySession(ProxySession):
    
    async def __aenter__(self) -> ArealProxySession:
        data = await _post_json_with_retry(
            self.http_session,
            f"{self.stripped_base_url}/rl/start_session",
            payload={},
        )
        self.session_id = data["session_id"]
        self.session_base_url = f"{self.stripped_base_url}/{self.session_id}"
      
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            await self.http_session.close()
            return

        if self.session_id is None:
            raise ValueError("Session ID is not set")

        payload = AReaLEndSessionRequest(
            session_id=self.session_id, final_reward=self.final_reward
        ).model_dump()
        await _post_json_with_retry(
            self.http_session,
            f"{self.stripped_base_url}/rl/end_session",
            payload=payload,
        )
        await self.http_session.close()
