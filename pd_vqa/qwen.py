"""Qwen Omni streaming Chat Completions transport; media is never logged.

Protocol: https://help.aliyun.com/zh/model-studio/qwen-omni
"""
import json
import os
from urllib import request,error

from .api import ChatAPI,APIError,NoRedirect


class QwenAPI(ChatAPI):
    @classmethod
    def from_env(cls):
        return cls(os.getenv("QWEN_BASE_URL","https://dashscope.aliyuncs.com/compatible-mode/v1"),
                   os.getenv("QWEN_MODEL","qwen3.8-omni-flash"),
                   os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
                   timeout=120,max_tokens=2048)

    def complete_json(self,messages):
        payload={"model":self.model,"messages":messages,"stream":True,
                 "reasoning_effort":"none","max_tokens":self.max_tokens}
        data=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
        if len(data)>10_000_000:
            raise APIError("Multimodal request exceeds configured 10 MB budget")
        req=request.Request(self._url,data=data,headers={"Content-Type":"application/json",
                            "Authorization":"Bearer "+self._key})
        try:
            chunks,total,finished,stopped=[],0,False,False
            with request.build_opener(NoRedirect()).open(req,timeout=self.timeout) as response:
                if "text/event-stream" not in response.headers.get("Content-Type",""):
                    raise APIError("Expected Qwen streaming response")
                while True:
                    line=response.readline(262145)
                    if not line:
                        break
                    total+=len(line)
                    if total>2_000_000 or len(line)>262144:
                        raise APIError("API response too large")
                    if not line.startswith(b"data:"):
                        continue
                    event=line[5:].strip()
                    if event==b"[DONE]":
                        finished=True
                        break
                    item=json.loads(event)
                    if item.get("error"):
                        raise APIError("Qwen stream reported a provider error")
                    for choice in item.get("choices",[]):
                        reason=choice.get("finish_reason")
                        if reason not in {None,"stop"}:
                            raise APIError("API output truncated or rejected")
                        stopped=stopped or reason=="stop"
                        text=choice.get("delta",{}).get("content")
                        if text is not None:
                            if not isinstance(text,str):
                                raise APIError("Invalid API content type")
                            chunks.append(text)
            if not finished or not stopped:
                raise APIError("Incomplete Qwen response stream")
            content="".join(chunks).strip()
            if content.startswith("```json\n") and content.endswith("```"):
                content=content[8:-3].strip()
            result=json.loads(content)
            if not isinstance(result,dict):
                raise APIError("API output must be a JSON object")
            return result
        except APIError:
            raise
        except error.HTTPError as exc:
            status=exc.code
            exc.close()
            raise APIError(f"Qwen API HTTP status {status}") from None
        except Exception:
            raise APIError("Qwen network or response validation failed") from None
