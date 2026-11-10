# Orchestrator Agent：任务识别、专家激活与执行协调

对应代码：`pd_vqa/orchestration.py`；数据契约：`pd_vqa/contracts.py`；接入点：`pd_vqa/workflow.py`。

## 职责与研究来源

参考 [Why Specialist Models Still Matter: A Heterogeneous Multi-Agent Paradigm for Medical Artificial Intelligence](https://arxiv.org/html/2605.29744) §3.2.2 的三项编排职责。以下为本项目针对同人异步步态、面部、语音记录的实现，不是对该论文模型、阈值或效果的完整复现。

| 步骤 | 解决的问题 | 当前实现 |
| --- | --- | --- |
| Task Identification | 疾病背景不等于分类请求；一个问题可能同时要求测量和解释 | 识别目标、涉及模态以及 observation / measurement / classification / explanation 意图。默认离线规则；`--online` 可调用 DeepSeek JSON，枚举校验失败回退。明确提到的模态不能被 API 丢弃。 |
| Agent Activation | 选错专家、缺失数据、存在数据但没有匹配模型 | 按请求检查记录与工具能力；每个专家绑定一个受试者、一个记录、一个模态。缺失为 `missing_data`，无可执行输入为 `unavailable`，可执行为 `ready`。 |
| Execution Coordination | 专家不知道具体任务；复核重复或越界；过程不可追溯 | 下发带分类开关、允许工具、分析要求与输出要求的 `ExpertTask`，收集报告并记录事件。复核方案先整批校验，再调用对应专家。共享一张 GPU 时按序执行。 |

Orchestrator 不负责诊断、判断证据可信度或制定复核方案。Evidence Critic 检查证据；Collaborative Reasoner 选择工具动作；Decision Agent 组织最终回答。

## 分类流何时打开

同时满足：

1. 目标是 `pd_presence` 或 `severity`，来自问题的分类请求，或使用者的显式 `--target` 覆盖。
2. 对应专家有可用特征来源，并登记了匹配目标的已训练检查点。
3. `severity` 提供分组定义；真正推理时还校验与训练定义的哈希一致。

`classification_requested` 表示任务需要分类；`classification_enabled` 表示该专家能够执行。二者不同：语音没有分类头时仍可测量音频，但不能生成疾病概率。阻塞原因写在 `blocked_requirements`。

例子：

- “请描述这位帕金森患者的步态表现” → Gait Expert，关闭分类；“帕金森患者”是背景。
- “不要分类，只计算语音声学测量” → Speech Expert，关闭分类。
- “结合步态和语音判断是否患有帕金森病，并解释依据” → Gait / Speech 各自执行匹配分类器并返回证据；模型缺失会明确记录。
- “结合面部和语音分析”，但没有面部记录 → 仍保留缺失的 Face Expert 任务，最终报告缺失，不静默改成只有语音的问题。

离线规则仅为初始实现，不是经过临床任务识别评价的模型。歧义问题应查看计划，必要时显式指定目标与模态；`--target` 是研究者的人工覆盖，具有优先权。

## 专家收到的任务包

```json
{
  "task_id": "由记录身份与问题生成的稳定ID",
  "subject_id": "P001",
  "record_id": "gait_visit1",
  "modality": "gait",
  "question": "结合步态和语音判断是否患有帕金森病",
  "target": "pd_presence",
  "modalities": ["gait"],
  "intents": ["classification"],
  "classification_requested": true,
  "classification_enabled": true,
  "allowed_operations": ["inspect_media", "measure", "read_features", "classify"],
  "instructions": ["分析步行与身体运动；仅报告工具支持的量", "每项依据保留证据ID、时间、单位和局限"],
  "output_requirements": ["findings", "measurements", "evidence_ids", "source", "quality", "limitations", "prediction", "probabilities", "uncertainty"]
}
```

示例只展示核心字段；实际计划包含完整 `Task` 字段和每项激活状态。文件路径留在本地记录中，不写入执行计划，不发送给任务识别 API。

能力检查只读取文件可用性及分类检查点元数据，不执行编码器或分类推理、不下载模型、不写特征缓存；它不保证模型文件完整或医学有效。真正推理仍校验受试者、原始源哈希、编码器版本与特征契约、分类目标和训练/验证来源。分类之外的描述目前以实际工具测量为基础，尚未接入直接理解原始视频/音频的多模态大模型。

## 查看计划与执行

先预览，不调用专家分析工具：

```bash
python -m pd_vqa plan --case examples/case.example.json \
  --question '结合步态与语音判断是否患有帕金森病，并给出测量依据' \
  --out runs/P001_plan.json
```

正式执行：

```bash
python -m pd_vqa run --case examples/case.example.json \
  --question '请描述步态并报告可核查的测量' \
  --out runs/P001_gait
```

运行目录写出 `plan.json`、`result.json`、`trace.jsonl`、`answer.md`。事件中可查 `plan / dispatch / completed / review_dispatch / review_completed`。输出目录或计划文件应使用新路径，避免覆盖原记录。

## 复核边界与测试

复核必须属于已经激活且被 Critic 指出的专家，工具与参数必须在对应记录的候选动作中。每轮同一模态最多一个动作，历史动作不得重复；使用与首轮相同的绑定任务，执行前重新校验身份。全部动作检查通过后才开始本轮执行。

当前窗口复核依靠清单预声明的真实可用片段，不由 LLM 发明动作。最大轮数、证据充分、无新动作等终止条件保留在 Workflow 中。

`tests/test_orchestration.py` 覆盖分类开关、中英文问题、API 越权回退、多意图、缺失模态、无分类头、严重度定义、真实音频不误启分类、计划预览、任务绑定、复核越界与重复动作。此类测试验证工程行为，不说明诊断准确率。
