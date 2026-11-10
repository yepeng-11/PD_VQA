"""Multimodal perception is an external boundary; measurements/control stay real."""
from contextlib import contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler,HTTPServer
import importlib.util
import json
from threading import Thread

import pytest

from pd_vqa.contracts import Case,Record,Task
from pd_vqa.workflow import Workflow


def implementation():
    assert importlib.util.find_spec('pd_vqa.experts') is not None, 'Multimodal expert implementation missing'
    from pd_vqa.experts import MultimodalExpertTools
    from pd_vqa.qwen import QwenAPI
    return MultimodalExpertTools,QwenAPI


@contextmanager
def streaming_endpoint(*,status=200,done=True):
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(status)
            self.send_header('Content-Type','text/event-stream')
            self.end_headers()
            for text in ('{"find','ings":[]}'):
                item={'choices':[{'delta':{'content':text},'finish_reason':None}]}
                self.wfile.write(('data: '+json.dumps(item)+'\n\n').encode())
            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
            if done:
                self.wfile.write(b'data: [DONE]\n\n')
        def log_message(self,*args):
            pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}',received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_qwen_stream_transport_uses_omni_protocol_without_deepseek_parameters():
    _,API=implementation()
    with streaming_endpoint() as (url,received):
        response=API(url,'qwen3.8-omni-flash','test-key').complete_json([
            {'role':'user','content':[{'type':'text','text':'Return JSON'}]}])
    assert response=={'findings':[]}
    assert received[0]['stream'] is True
    assert received[0]['reasoning_effort']=='none'
    assert 'thinking' not in received[0]


def test_qwen_incomplete_stream_is_not_treated_as_a_complete_response():
    _,API=implementation()
    with streaming_endpoint(done=False) as (url,_),pytest.raises(RuntimeError,match='incomplete|Incomplete'):
        API(url,'qwen3.8-omni-flash','test-key').complete_json([])


def test_qwen_http_failure_does_not_disclose_credentials():
    _,API=implementation()
    with streaming_endpoint(status=401) as (url,_),pytest.raises(RuntimeError) as exc:
        API(url,'qwen3.8-omni-flash','private-test-key').complete_json([])
    assert '401' in str(exc.value)
    assert 'private' not in str(exc.value)


class PerceptionAPI:
    model='qwen3.8-omni-flash'
    def __init__(self,*,invalid=None):
        self.requests=[]
        self.invalid=invalid

    def complete_json(self,messages):
        self.requests.append(messages)
        payload=json.loads(messages[-1]['content'][-1]['text'])
        media=next(e for e in payload['evidence'] if e['kind']=='media_input')
        finding={'text':'录音包含持续的音调' if payload['modality']=='speech' else '画面中的动作信息不足',
                 'status':'observed','evidence_ids':[media['evidence_id']],'interval':media['interval']}
        response={'findings':[finding],'limitations':[],'evidence_sufficient':True}
        if self.invalid=='ids':
            finding['evidence_ids']=['invented']
        elif self.invalid=='time':
            finding['interval']=[1000.,1001.]
        elif self.invalid=='probabilities':
            response['probabilities']=[.01,.99]
        elif self.invalid=='uncertain':
            finding['status']='uncertain'
            response['evidence_sufficient']=False
        elif self.invalid=='network':
            raise RuntimeError('simulated external failure')
        elif self.invalid=='abstain':
            finding['status']='not_assessable'
            finding['evidence_ids']=[]
            response['evidence_sufficient']=False
        return response


def clean_audio(path):
    import math
    import struct
    import wave
    values=[int(2000*math.sin(2*math.pi*160*i/8000)) for i in range(16000)]
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(struct.pack('<'+'h'*len(values),*values))


def test_speech_expert_combines_actual_measurements_raw_audio_and_query(tmp_path):
    Tools,_=implementation()
    from pd_vqa.tools import MediaTools
    raw=tmp_path/'private_audio.wav'
    clean_audio(raw)
    api=PerceptionAPI()
    tools=Tools(MediaTools(workspace=tmp_path/'cache'),api)
    result=Workflow(tools).run(Case('p',[Record('p','r','speech',path=str(raw))],'engineering'),
                               Task('描述语音并报告测量','speech_observation',['speech']))
    report=result.fusion.reports[0]
    assert report.semantic_status=='complete'
    assert report.findings[0].evidence_ids
    assert report.findings[0].method=='model_interpretation_unverified'
    assert report.prediction is None
    assert any(e.name=='rms_digital' and isinstance(e.value,float) for e in report.evidence)
    assert any(p['type']=='input_audio' for p in api.requests[0][-1]['content'])
    assert str(tmp_path) not in json.dumps(api.requests)
    assert 'data:audio' not in json.dumps(asdict(report))
    assert '模型解读' in result.answer.text
    assert result.answer.candidate is None


@pytest.mark.parametrize('invalid',['ids','time','probabilities','uncertain','network'])
def test_bad_semantic_response_never_replaces_measurements_or_creates_a_diagnosis(tmp_path,invalid):
    Tools,_=implementation()
    from pd_vqa.tools import MediaTools
    raw=tmp_path/'s.wav'
    clean_audio(raw)
    result=Workflow(Tools(MediaTools(workspace=tmp_path/'cache'),PerceptionAPI(invalid=invalid)),max_reviews=0).run(
        Case('p',[Record('p','r','speech',path=str(raw))]),Task('描述语音','speech_observation',['speech']))
    report=result.fusion.reports[0]
    assert report.semantic_status in {'failed','insufficient'}
    assert any(e.name=='duration_seconds' and e.value==2. for e in report.evidence)
    assert report.prediction is None
    assert result.answer.status=='insufficient_evidence'
    assert 'invented' not in json.dumps(result.to_dict())


@pytest.mark.parametrize('modality',['gait','face'])
def test_visual_experts_send_video_and_keep_record_binding(tmp_path,modality):
    Tools,_=implementation()
    cv2=pytest.importorskip('cv2')
    import numpy as np
    from pd_vqa.tools import MediaTools
    raw=tmp_path/'private.avi'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'MJPG'),8,(64,64))
    for _ in range(16):
        writer.write(np.zeros((64,64,3),dtype=np.uint8))
    writer.release()
    api=PerceptionAPI()
    result=Workflow(Tools(MediaTools(workspace=tmp_path/'cache'),api),max_reviews=0).run(
        Case('p',[Record('p','r',modality,path=str(raw))]),Task('描述动作',modality+'_observation',[modality]))
    report=result.fusion.reports[0]
    assert report.semantic_status=='complete'
    assert report.record_id=='r' and report.modality==modality
    assert any(p['type']=='video_url' for p in api.requests[0][-1]['content'])
    assert str(raw) not in json.dumps(api.requests)


def test_review_cites_original_timeline_and_sends_only_selected_audio(tmp_path):
    Tools,_=implementation()
    from pd_vqa.tools import MediaTools
    raw=tmp_path/'s.wav'
    clean_audio(raw)
    api=PerceptionAPI()
    rec=Record('p','r','speech',path=str(raw),windows=[{'start':1.,'end':2.}])
    report=Tools(MediaTools(workspace=tmp_path/'cache'),api).analyze(
        rec,Task('描述语音','speech_observation',['speech']),{'operation':'window','start':1.,'end':2.})
    assert report.findings[0].interval==[1.,2.]
    assert next(e for e in report.evidence if e.kind=='media_input').interval==[1.,2.]
    content=api.requests[0][-1]['content']
    import base64
    import io
    import wave
    audio=next(p for p in content if p['type']=='input_audio')['input_audio']['data']
    with wave.open(io.BytesIO(base64.b64decode(audio.split(',',1)[1])),'rb') as f:
        assert f.getnframes()/f.getframerate()==1.


def test_honest_abstention_without_a_positive_claim_is_not_a_schema_failure(tmp_path):
    Tools,_=implementation()
    from pd_vqa.tools import MediaTools
    raw=tmp_path/'s.wav'
    clean_audio(raw)
    result=Workflow(Tools(MediaTools(),PerceptionAPI(invalid='abstain')),max_reviews=0).run(
        Case('p',[Record('p','r','speech',path=str(raw))]),Task('描述语音','speech_observation',['speech']))
    report=result.fusion.reports[0]
    assert report.semantic_status=='insufficient'
    assert report.semantic_error is None
    assert report.findings[0].status=='not_assessable'


def test_global_multimodal_query_is_scoped_to_assigned_expert(tmp_path):
    Tools,_=implementation()
    from pd_vqa.tools import MediaTools
    raw=tmp_path/'s.wav'
    clean_audio(raw)
    api=PerceptionAPI()
    question='分别描述三个记录，综合步态、面部和语音证据'
    Tools(MediaTools(),api).analyze(Record('p','r','speech',path=str(raw)),
                                  Task(question,'evidence_summary',['gait','face','speech']))
    payload=json.loads(api.requests[0][-1]['content'][-1]['text'])
    assert payload['global_question']==question
    assert payload['assigned_scope']=={'modality':'speech','record_id':'r','other_modalities_out_of_scope':True}
    assert payload['question']!=question


@pytest.mark.parametrize('corruption',['schema','pickle'])
def test_late_feature_failure_preserves_successful_task_aware_measurements(tmp_path,monkeypatch,corruption):
    Tools,_=implementation()
    import torch
    from pd_vqa.tools import MediaTools
    from pd_vqa.backends.measurements import Measurement
    import pd_vqa.backends.acoustics
    def acoustics(*args,**kwargs):
        return [Measurement('independent_acoustic_metric',160.,'Hz','speech','metric')],{'status':'measured'}
    monkeypatch.setattr(pd_vqa.backends.acoustics,'acoustic_measurements',acoustics)
    raw,cache=tmp_path/'s.wav',tmp_path/'bad.pt'
    clean_audio(raw)
    if corruption=='schema':
        torch.save({},cache)
    else:
        cache.write_bytes(b'invalid checkpoint')
    report=Tools(MediaTools(workspace=tmp_path/'cache'),PerceptionAPI()).analyze(
        Record('p','r','speech',path=str(raw),feature_cache=str(cache),capture_task='continuous'),
        Task('报告语音测量','speech_observation',['speech']))
    assert any(e.name=='independent_acoustic_metric' and e.value==160. for e in report.evidence)
    assert report.prediction is None
    assert report.semantic_status=='complete'


def test_cached_features_without_raw_media_do_not_masquerade_as_multimodal_perception(tmp_path):
    Tools,_=implementation()
    from pd_vqa.tools import MediaTools
    import torch
    cache=tmp_path/'x.pt'
    torch.save({'subject_id':'p','record_id':'r','modality':'speech','features':torch.zeros(2,4),
                'timestamps':torch.tensor([0.,1.]),'quality_metadata':{'score':1.,'method':'fixture'}},cache)
    api=PerceptionAPI()
    result=Workflow(Tools(MediaTools(workspace=tmp_path/'cache'),api),max_reviews=0).run(
        Case('p',[Record('p','r','speech',feature_cache=str(cache))]),Task('描述语音','speech_observation',['speech']))
    assert not api.requests
    assert result.fusion.reports[0].semantic_status=='unavailable'
    assert result.answer.status=='insufficient_evidence'


def test_cli_qwen_plan_registers_perception_without_uploading_media(tmp_path,monkeypatch,capsys):
    implementation()
    from pd_vqa.cli import main
    raw=tmp_path/'s.wav'
    clean_audio(raw)
    manifest=tmp_path/'case.json'
    manifest.write_text(json.dumps({'subject_id':'p','records':[{'record_id':'r','modality':'speech','path':'s.wav'}]}))
    monkeypatch.setenv('QWEN_API_KEY','test-key')
    with streaming_endpoint() as (url,received):
        monkeypatch.setenv('QWEN_BASE_URL',url)
        main(['plan','--case',str(manifest),'--question','Describe speech','--expert-backend','qwen',
              '--out',str(tmp_path/'plan.json')])
    assert not received
    plan=json.loads((tmp_path/'plan.json').read_text(encoding='utf-8'))
    assert 'analyze' in plan['assignments'][0]['task']['allowed_operations']
    assert not plan['assignments'][0]['task']['classification_enabled']
    capsys.readouterr()
