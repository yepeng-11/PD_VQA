# 真人步态与面部视频下载及验证（2026-11-11）

已将3个来源的5条真人视频下载到服务器，对应4名不同来源的受试者/演员。4条面部记录都能产生实际面部运动测量；2条RAVDESS记录完成完整回答。KTH步态样例未产生有效运动学指标；两条YouTubePD示例的完整报告仍未通过当前程序要求。失败和基础设施问题均保留，未将它们包装为诊断准确率。

## 数据与保存位置

- 原始数据：`/yepeng/PD_VQA/data/public/vision_pilot_20261111/`。
- 实验结果：`/yepeng/PD_VQA/runs/vision_experts_real_20261111/`。
- 本地报告及去除原始媒体/私有映射的结果副本：`C:/Users/Lenovo/Documents/Codex/2026-10-10/new-chat/outputs/vision_experts_real_20261111_verified/`。
- 5个选中文件原始大小合计17,894,685字节，约17.89 MB；没有下载完整数据集。
- RAVDESS两条来自同一演员，不算两名独立受试者；不同数据集之间没有构造虚假的同人病例。

| 来源 | 本次选择 | 访问与完整性 |
| --- | --- | --- |
| [KTH Actions](https://www.csc.kth.se/cvap/actions/) | 官网person15步行AVI，160×120、25fps | 非商业研究使用；计算本地SHA256。它不是帕金森诊断数据 |
| [YouTubePD作者示例](https://github.com/samwli/YouTubePD-data/tree/main/processed_examples) | video74.mp4、video91_original.mp4；作者README称为两名患者示例 | 固定Git提交，核对Git blob SHA1与文件大小，传输后核对SHA256；不重分发原始视频到GitHub |
| [RAVDESS](https://zenodo.org/records/1188976) | Actor01的neutral与happy，各取第一句、第一遍、普通强度的AV版本 | CC BY-NC-SA4.0；只读取所选ZIP条目与目录，约7.73 MB网络读取；校验条目CRC和文件SHA256 |

RAVDESS整包公布的MD5有记录，但没有下载整包，因此未声称验证其整包MD5。YouTubePD因服务器GitHub DNS失败，采用本地下载、固定版本校验、SSH传输；原下载失败条目仍在download_manifest.json中。

## 实验协议

- 在查看模型输出前固定选择：1条KTH、2条YouTubePD示例、2条RAVDESS。每条固定取开头最多10秒，不根据分类结果或症状预选片段。
- 保留原始文件；测试副本等比例缩小到最长边不超过640像素，不放大，保留源帧率，移除音轨并用mp4v编码。报告时间以测试片段时间为准，原始帧率/时间映射另存。
- Gait Expert配置Pose Landmarker Lite；Face Expert配置Face Landmarker。测量由服务器真实MediaPipe推理产生，未用手填或模拟关键点。
- 问题只要求描述可见动作、引用测量和解释局限；没有启动分类流、没有训练检查点、没有疾病预测概率。原始文件名/来源疾病标签不进入提示文本；公开人物可能被模型识别的影响未评价。
- Qwen使用`qwen3.8-omni-flash`真实API。首轮每条最多1次预声明的前半段复核；原先总预算12次。实际首轮8次，另对日志污染记录增加1次有单独协议的复测，总计9次。
- 运动学/面部数值的存在、有限性与来源可以核查；没有人工关键点真值、症状时段或医生QA答案，无法评价数值误差或医学解释准确率。

## 每条记录

| 记录ID | 来源 | 初次实际运动指标数 | 完整回答状态 |
| --- | --- | --- | --- |
| Rb2c55dd2f | KTH：person15步行 | 0 | insufficient_evidence |
| R22b7e9c50 | RAVDESS：Actor01，neutral | 8 | answered |
| Rdf1fbd36b | RAVDESS：Actor01，happy | 8 | answered |
| R7a7480841 | YouTubePD：yt74 | 8 | insufficient_evidence |
| R326b0f5b2 | YouTubePD：yt91（仅此条因基础设施问题单独复测） | 7 | insufficient_evidence |

表中YouTubePD yt91使用单独基础设施复测的结果，原始日志污染结果未覆盖；该表不是一次统一运行的临床成绩。8项面部指标包括下颌开合、微笑变化/不对称、下唇运动、头偏航、眼睑相关变化、下颌变化速度方差及满足采样条件时的眨眼速率。它们是RGB代理，不是直接FACS AU或MDS-UPDRS评分。yt91因有效观测覆盖等条件未输出眨眼速率，仅有7项。

## 定位到的实际问题

### 1. 步态的联合关键点门控过于严格

KTH片段的80个采样帧中，28帧检测到人体；这28帧右手腕的visibility均低于0.5。当前代码要求肩、腕、髋、踝共8点同时达到0.5，因而0帧通过，最终没有运动学测量。

两次真实API调用后系统仍保留证据不足。没有降低阈值、编造0值或把没有测量解释为没有症状。该结果定位了现有门控问题，并不能证明低分辨率是唯一原因。后续可考虑逐指标检查其所需关键点，分别保留可用测量；本轮没有改变生产算法。

### 2. 模型结构化输出不稳定

- yt74：初次响应含uncertain判断，现有规则触发复核；复核在API解析层返回“API output must be a JSON object”，保留APIError及工具测量。没有保存到可供引用审计的顶层JSON对象，无法重建该响应的所有内容。
- yt91：仅对基础设施污染后的单独复测，收到的一个observed判断没有任何evidence_ids，因此正确触发Invalid model interpretation。没有为了通过而自动补证据ID或放宽校验。
- 这说明测量成功不保证完整报告成功；模型输出格式与引用要求仍需改进。

### 3. 测试程序的日志与运行环境问题

首轮后段SSH输出断流，审计日志print抛出BrokenPipeError，导致yt91的两份已保存、可独立通过格式校验的响应没有正常进入报告。这是实验程序污染，不归因于模型医学能力。

新增日志断流回归测试，先复现失败，再让progress日志忽略BrokenPipeError。单独复测将日志写到服务器文件；首次后台启动错误地解析了venv解释器符号链接，使用基础解释器而缺少视频依赖，结果为unavailable，且0次API调用。纠正解释器路径后仅增加1次API调用，仍因模型正向判断空引用而拒绝回答。三份记录均保存，未反复调用直到通过。

### 4. 解释正确性还未自动验证

原始响应中存在把decoded_frame_fraction误用于支持“全程人脸可追踪”，以及把工程质量分数解释为清晰度/模糊程度的文字。解码、检测有效性、清晰度和临床可靠性不是同一概念。当前程序能拦截缺失引用/格式错误，不能自动验证每句解释是否得到证据支持。模型解读仍标记为待核验。

## 调用与证据审计

- 总API尝试：9次；首轮8次、仅日志污染案例单独复测1次；平均调用耗时12.49秒。未估算未获取的token计费。
- 其中8次保存了顶层JSON对象，1次在API解析层失败。首轮7份对象通过格式/区间校验；新增复测对象因observed空引用失败。
- 可解析对象中已出现的证据ID引用：首轮41/41有效，新增复测4/4有效。该指标不能发现“应该引用却没有引用”的判断，也不能证明语义支持成立；失败的非对象响应不在引用分母中。
- 最终保留的引用/时间范围通过程序检查，工具数值均有限，分类流均关闭。
- 首轮原始状态为2条answered、3条insufficient，其中1条受到测试日志污染。独立修复复测后仍只有2条完整回答；不将这一比例称为帕金森诊断准确率。

## 产物与下一步

本地产物根目录如上；`download_manifest.json`记录下载及失败恢复，`measurement_summary.json`记录真实测量，`gait_detection_diagnostic.json`记录逐帧可见性，`api_calls.jsonl`记录全部调用，`provider_responses/`保存可解析响应，`results/`保存首轮结果，`results_infrastructure_retry*/`保存基础设施复测。`independent_audit.json`与`infrastructure_retry_audit.json`分别审计首轮及单独复测，`implementation_provenance.json`记录代码/脚本哈希。

服务器的输入清单位于实验目录`inputs/<记录ID>/case.json`，原始来源映射offline_mapping.json权限0600，未复制到本地报告目录。生产代码保持基础提交`31d7604f97cdcbba5efd1a20d1df528a259c6118`；本轮仅测试脚本修复日志和启动环境，未调整专家模型、门限、提示词或融合策略。

下一步优先处理：按指标所需关键点分别判断测量有效性；提高模型结构化响应与引用的稳定性；建立少量人工审核的动作/测量/回答样本，随后再验证医学内容正确性。
