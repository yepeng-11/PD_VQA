"""Provider boundary validation and non-disclosure on failure."""
import pytest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from threading import Thread


@contextmanager
def endpoint(status=200,body=None):
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())
        def log_message(self,*args):
            pass
    server=HTTPServer(("127.0.0.1",0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}",received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_api_requires_explicit_model_and_key():
    from pd_vqa.api import ChatAPI
    with pytest.raises(ValueError):
        ChatAPI("https://api.deepseek.com", "", "secret")


def test_api_refuses_plaintext_remote_endpoints():
    from pd_vqa.api import ChatAPI
    with pytest.raises(ValueError):
        ChatAPI("http://external.invalid", "model", "secret")


def test_json_transport_exercises_real_http_protocol():
    from pd_vqa.api import ChatAPI
    body={"choices":[{"finish_reason":"stop","message":{"content":'{"indices":[0]}'}}]}
    with endpoint(body=body) as (url,received):
        result=ChatAPI(url,"configured-model","test-only-key").complete_json([
            {"role":"user","content":"select existing evidence; return json"}])
    assert result=={"indices":[0]}
    assert received[0]["model"]=="configured-model"
    assert received[0]["response_format"]=={"type":"json_object"}


def test_http_failure_does_not_expose_key_or_provider_response():
    from pd_vqa.api import ChatAPI,APIError
    with endpoint(status=401,body={"error":"private information"}) as (url,_):
        with pytest.raises(APIError) as failure:
            ChatAPI(url,"model","private-key").complete_json([])
    assert "private" not in str(failure.value)
    assert "401" in str(failure.value)


def test_decision_rejects_invented_evidence_and_preserves_baseline():
    from pd_vqa.contracts import Task
    from pd_vqa.workflow import DecisionAgent,EvidenceCritic,WeightedEvidenceFusion
    from test_workflow import report
    class ExternalAPI:
        def complete_json(self,messages):
            return {"text":"fabricated interpretation","evidence_ids":["invented"]}
    reports=[report()]
    answer=DecisionAgent(ExternalAPI()).answer(Task("总结","evidence_summary",["gait"]),
            WeightedEvidenceFusion().fuse(reports),EvidenceCritic().assess(reports),[],"engineering",[])
    assert answer.text != "fabricated interpretation"
    assert answer.evidence_ids == ["gait-obs"]
