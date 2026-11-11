# 当前多智能体框架验证报告（2026-11-11）

结论：现有框架通过本次工程功能验证，但尚不能据此认为诊断问答的医学可靠性已验证。另一个专门的语义负例未被拦截，确认当前 Critic 不能判断引用内容是否真正支持模型陈述。

## 验证结果

| 项目 | 本次结果 | 可以说明什么 |
| --- | --- | --- |
| 自动化测试 | 本地与服务器均113项通过，0失败、0跳过 | 当前代码在已覆盖条件下满足程序行为要求 |
| 场景验收 | 本地修复前27/27、修复后27/27；服务器修复后27/27 | 编排、工具、报告校验、融合、复核及停止等预声明流程可执行 |
| 语义支持负例 | 1例未拦截 | 有效ID和时间不代表语义支持成立；这是限制，不计入27项通过率 |
| 代码静态检查 | ruff通过 | 未发现已配置规则覆盖的语法/静态问题 |
| 本次真实API调用 | 0 | 语义接口部分使用受控响应替身，不能评价实际模型理解能力 |
| 本次服务器执行 | 已完成，Python 3.13.17；源码SHA256与本地一致 | 最初端口拒绝连接，恢复后已同步修复并复跑相同验证 |
| 疾病诊断准确率 | 未评价 | 没有正式训练检查点与问答金标准，不能计算有效临床成绩 |

最初本地缺少OpenCV，测试为107通过、3跳过；将OpenCV仅装入单独的临时依赖目录后，补齐视频测试。新增3个停止原因回归用例后，最终113项全部通过。共享Python环境未升级或替换。

## 数据与模型边界

- 原始输入使用可复现生成的PCM音频、运动方块视频；解码与RMS、削波、时长、清晰度测量由实际代码执行。
- 语义接口使用明确标记的受控响应，专门构造正确引用、错误引用、越界时间、模型自报概率、无法评估和API失败。实际MLLM没有在本次重新调用。
- 分类融合场景使用受控概率报告，`trained=True`只用于模拟分类器边界的程序测试；没有训练或评价帕金森模型。测试目录名、模型ID和协议均标记为fixture。
- 方块视频没有人体或面部，未配置关键点检测；视频解码成功不代表真实摆臂、面部动作测量成功。
- 知识卡使用example.org明确标记的工程替身，仅检验检索与证据边界，不声称医学知识经过审核。
- 27项预期在运行前写入protocol.json；失败、跳过与修复前结果保留。它们是选择的功能场景，不代表全部输入空间。

## 场景清单

| 编号 | 验证行为 | 输入/响应来源 | 修复后结果 |
| --- | --- | --- | --- |
| A01 | 正常音频测量与回答 | synthetic_raw_audio | passed |
| A02 | 原始音频与测量接入语义专家 | synthetic_audio_scripted_api | passed |
| A03 | 观察问题仅激活相关专家 | synthetic_audio_scripted_api | passed |
| A04 | 诊断请求没有检查点时保留不足 | synthetic_audio_scripted_api | passed |
| A05 | 明确要求的模态缺失 | synthetic_raw_audio | passed |
| A06 | 损坏音频不冒充分析成功 | corrupt_raw_file | passed |
| A07 | 无复核时保留低质量问题 | synthetic_raw_audio | passed |
| A08 | 定向复核实际改善削波测量 | synthetic_raw_audio | passed |
| A09 | 持续低质量时不重复执行同一动作 | synthetic_raw_audio | passed |
| A10 | 达到复核预算时停止 | synthetic_raw_audio | passed |
| A11 | 不存在的证据引用被拦截 | synthetic_audio_scripted_fault | passed |
| A12 | 越界时间引用被拦截 | synthetic_audio_scripted_fault | passed |
| A13 | LLM额外疾病概率被拦截 | synthetic_audio_scripted_fault | passed |
| A14 | 不确定解读保留证据不足 | synthetic_audio_scripted_fault | passed |
| A15 | API异常仍保留独立测量 | synthetic_audio_scripted_fault | passed |
| A16 | 诚实无法评估不误判为格式失败 | synthetic_audio_scripted_fault | passed |
| A17 | 缓存特征不能冒充原始媒体 | synthetic_cached_features_scripted_api | passed |
| A18 | 同目标三专家概率融合归一化 | controlled_predictions_NOT_clinical | passed |
| A19 | 二比一分歧不靠多数票覆盖 | controlled_predictions_NOT_clinical | passed |
| A20 | 类别定义不一致不能给联合结论 | controlled_predictions_NOT_clinical | passed |
| A21 | 不同症状目标不误判为同目标冲突 | controlled_predictions_NOT_clinical | passed |
| A22 | 不同受试者记录被拒绝 | contract_fault_injection | passed |
| A23 | 空输入不编造观察 | empty_input | passed |
| A24 | 候选记忆和冻结测试边界 | controlled_memory_fixture | passed |
| A25 | 知识检索不能充当患者观测 | controlled_knowledge_fixture | passed |
| A26 | 既有运行产物不被覆盖 | artifact_integrity | passed |
| A27 | 三模态媒体解码与独立专家汇总 | synthetic_media_scripted_api | passed |

## 实际工具复核的前后对照

对同一条合成音频，前1秒人为削波、后1秒为未削波纯音。先测整段，再按预声明的1–2秒窗口测量。

| 模式 | 削波比例 | 工程质量 | 复核轮数 | 最终状态 |
| --- | --- | --- | --- | --- |
| 不复核（A07） | 0.5 | 0.05 | 0 | insufficient_evidence |
| 定向窗口复核后（A08） | 0.0 | 1.0 | 1 | answered |

原始证据与复核证据均保留，片段时间及证据ID不同。该结果证明窗口重测可改变实际工程测量；不证明症状判断更正确。持续削波时（A09）重测仍失败，程序停止并保留证据不足。

## 本次修复的问题

缺少分类检查点、缺失问题要求的模态或输入为空时，Decision已正确返回insufficient_evidence，但循环默认停止原因仍可能是evidence_sufficient。原因是Critic检查已有报告，而最终任务要求由Decision继续检查。

新增3个回归测试，先观察3项均失败，再修改workflow.py：仅在原停止原因为evidence_sufficient而最终未回答时，改为decision_加最终状态；decision轨迹同时记录该原因。复核预算耗尽、没有新动作等既有原因保持原义。未修改质量门限、融合公式或专家策略。

本地修复前产物位于`C:/Users/Lenovo/Documents/Codex/2026-10-10/new-chat/outputs/framework_validation_20261111/`；修复后产物位于`C:/Users/Lenovo/Documents/Codex/2026-10-10/new-chat/outputs/framework_validation_20261111_after_fix/`。服务器独立复跑产物位于`/yepeng/PD_VQA/runs/framework_validation_20261111/`。原有result.json未覆盖。

## 额外负例：语义支持尚未实现

输入只有合成音频；受控响应却写“该录音表明受试者右侧摆臂减少”，并引用真实音频证据ID与合法时间。

- 实际结果：answered，错误跨模态解读没有被拦截。
- 解读仍标记model_interpretation_unverified，且没有变成新的定量测量或分类概率。
- 这说明当前检查覆盖身份、格式、引用和时间，不验证“这一证据是否支持这句话”。
- 这是人工构造的能力边界探测，不是实际Qwen错误率，也不是临床失败率。
- 后续若要验证医学正确性，需要独立标注的判断/证据支持关系；本轮没有加入新的语义审核算法。

## 已有真实数据结果（引用2026-11-10记录，本次未重新运行）

MDVR-KCL：10人，5名患者+5名对照，每人20秒。10/10成功产生11项测量，9/10生成观察回答，1/10保留不足；11次Qwen调用，平均10.324秒/次。唯一复核案例没有改善。

原始引用99/100有效；不把被拒绝响应过滤后的比例报告成全部原始输出100%。分类流关闭，无诊断准确率评价。数据中混入其他说话人的片段不能直接归因于受试者症状。完整旧报告见[真实语音验证](validation_mdvr_speech_2026-11-10.md)。

## 当前能够得出的结论与待完成验证

1. 编排、真实媒体测量、结构化报告、规则检查、受控概率融合、有界复核和证据保存已在所选工程场景中通过。
2. 真实语音API链路有此前小样本记录；步态和面部尚缺真实受试者工具验证。本次三模态联合输入来自合成工程样本。
3. 正式疾病分类、自由文本医学正确性、三模态临床问答和真实冲突纠错收益仍待验证。
4. 下一阶段应先完成真实步态/面部的“输入→测量→有效性检查→报告”，再构建少量独立审核的问答与证据标注；暂不调整创新模块。
5. 服务器连接恢复后已同步停止原因修复并重跑相同验收：113项测试、27个功能场景通过，语义负例同样未拦截。服务器结果另存，不覆盖本地结果。

## 复现与来源

- 基础提交：`b4b460a6b7da07d67a01cd10a03ff785833ea0d2`；本次测试包含附带的源码差异，见tested_source_changes.patch与validation_provenance.json。
- 下列本地产物均在上述`outputs/framework_validation_20261111_after_fix/`目录，服务器产物均在上述`runs/framework_validation_20261111/`目录；它们不在仓库docs目录，也不全部上传GitHub。
- 服务器结果摘要：server_acceptance.json；已逐文件核对workflow.py与新增回归测试的SHA256。
- 最终全部自动测试XML：unit_tests.xml。
- 预声明协议：protocol.json；场景结果：summary.json；每例完整产物：scenarios/Axx/。
- 额外语义负例：semantic_support_probe/，独立于27个功能场景。
- 复现脚本原件复制在reproduction_sources/，运行入口仍为工作区tmp/validate_current_framework_20261111.py；参数为一个新的输出目录，避免覆盖。
- 本报告不依赖新增文献结论，也没有将合成信号或受控概率当作帕金森病例。
