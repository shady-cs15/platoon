from __future__ import annotations

from areal.experimental.openai.client import ArealOpenAI
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


# Patch to override AReaL's behavior of setting max_tokens to 512 if not provided.
class PatchedArealOpenAI(ArealOpenAI):
    def __init__(self, max_tokens: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tokens = max_tokens
        
        self.old_responses_create = self.responses.create
        self.old_chat_completions_create = self.chat.completions.create
        
        def responses(*args, **kwargs):
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = self.max_tokens
            return self.old_responses_create(*args, **kwargs)
        
        def chat_completions(*args, **kwargs):
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = self.max_tokens
            return self.old_chat_completions_create(*args, **kwargs)
        
        self.responses.create = responses
        self.chat.completions.create = chat_completions

