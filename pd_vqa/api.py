"""DeepSeek-compatible JSON chat transport, without a local LLM or RL.

Official protocol: https://api-docs.deepseek.com/guides/json_mode/
Credentials are read from the process environment and never serialized.
"""
import json
import os
from urllib import request, error
from urllib.parse import urlsplit


class APIError(RuntimeError):
    pass


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ChatAPI:
    def __init__(self, base_url, model, api_key, timeout=90, max_tokens=2048, thinking=False):
        url = urlsplit(base_url)
        local = url.hostname in {"localhost", "127.0.0.1", "::1"}
        if url.scheme != "https" and not (local and url.scheme == "http"):
            raise ValueError("API requires HTTPS (except loopback tests)")
        if not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Invalid API base URL")
        if not model or not api_key or "\n" in api_key or "\r" in api_key:
            raise ValueError("Configure model and DEEPSEEK_API_KEY")
        self._url, self.model, self._key = base_url.rstrip("/") + "/chat/completions", model, api_key
        self.timeout, self.max_tokens, self.thinking = timeout, max_tokens, thinking

    @classmethod
    def from_env(cls):
        return cls(os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                   os.getenv("DEEPSEEK_MODEL", ""), os.getenv("DEEPSEEK_API_KEY"))

    def complete_json(self, messages):
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "max_tokens": self.max_tokens, "response_format": {"type": "json_object"},
                   "thinking": {"type": "enabled" if self.thinking else "disabled"}}
        req = request.Request(self._url, data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                              headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"})
        try:
            with request.build_opener(NoRedirect()).open(req, timeout=self.timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise APIError("API response too large")
            result = json.loads(raw)
            choice = result["choices"][0]
            if choice.get("finish_reason") == "length":
                raise APIError("API output truncated")
            result = json.loads(choice["message"]["content"])
            if not isinstance(result, dict):
                raise APIError("API output must be a JSON object")
            return result
        except APIError:
            raise
        except error.HTTPError as exc:
            code = exc.code
            exc.close()
            raise APIError(f"API HTTP status {code}") from None
        except Exception:
            # Do not leak a key, endpoint, patient text or raw provider response.
            raise APIError("API network or response validation failed") from None
