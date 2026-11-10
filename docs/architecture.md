# 架构、模块来源与研究边界

## 本次实现采用的研究流程

输入不是同步三通道视频，而是带问题的同人多任务记录。保持独立模态专家，先取得各自证据，再评估质量和可比较的冲突；避免在初步专家判断前强制做跨模态特征交互，导致证据来源混杂。

| 模块 | 解决的问题、实现 | 参考与适配边界 |
| --- | --- | --- |
| Orchestrator | 识别目标与多意图；按数据和模型能力激活专家，生成绑定记录的任务包与分类开关；初次和复核均校验后调度。支持计划预览、显式覆盖和 DeepSeek JSON，详见 [编排说明](orchestrator.md) | HetMedAgent §3.2.2 的三项编排职责；本项目是固定工具注册与类型约束，没有复现其专科生成模型 |
| Encoder | VideoMAE gait / DINOv2 face / WavLM speech；冻结、串行加载、本地权重、缓存时间序列 | 用户旧 PD_MAgent 已测工具移植；是工程选择，不是宣称最优编码器 |
| 局部时间与模态内增强 | 输入投影、核3 Conv1D、带位置编码的2头 self-attention、残差与 LayerNorm、mask池化和分类头 | Conv1D 借鉴 DHMD 解耦前处理；自注意力为通用基线，没有移植其解耦或蒸馏 |
| 运动学/声学测量 | 实际信号与 MediaPipe 关键点计算，保留单位、任务与原始片段；数值确定后组织文本 | BioGait-VLM 的显式测量路线启发；采用自己的2D代理，不复现其 SKEL 估计 |
| Expert | 工具输出成为结构化报告：目标、概率、预测熵、质量、观测、证据ID、局限 | HetMedAgent 异构输出思路；这里预测熵来自分类 softmax，不是生成token概率，不声称校准 |
| Evidence Critic | subject/record/target/证据引用校验；质量不足；仅比较同目标且同类别定义的预测；不同定义报告不可比较 | MedAgent-Pro 每步证据检查 + HetMedAgent 同类证据对照思想；当前冲突判定是可审计规则，尚未训练细粒度冲突模型 |
| Collaborative Reasoner | 在工具实际可用的候选窗口中选择动作；相同动作不重复；可用 LLM 规划但不可发明工具参数 | HealthFlow 的定向修复思路；“跨模态冲突后重提特征”的有效性仍需实验 |
| Clinical Agent | 根据目标检索本地有原文来源的知识卡；知识不冒充患者观测 | MedAgent-Pro RAG 规划启发；当前为小规模本地检索，不是完整 PubMed RAG 服务 |
| Weighted Fusion | 同目标概率晚期加权；保留原报告和分歧 | HetMedAgent 证据权重思想，具体公式不同；是规则基线，没有使用 MoE 训练或 EDL |
| Decision | 证据不足/无检查点/数据域不匹配/分歧未解决时保留不确定；API 仅选择已有证据，最终测量和类别由程序渲染 | MedAgent-Pro 的证据依据与终止思路；尚未引入自由医学解释的验证器 |
| Reflection/Memory | 已完成的动作提炼为候选；外部反馈激活；失效经验不再检索；测试只读 | HealthFlow / SkeMex 思路，独立简化实现。当前是程序性技能，不自动生成新的临床诊断规则 |

## 主要公式及对应文件

`models.py`：给定每模态真实序列 $X_m\in\mathbb R^{T_m\times d_m}$，

$$H_m=W_m X_m+\operatorname{Conv1D}(W_m X_m),$$
$$Z_m=\operatorname{LayerNorm}(H_m+PE+\operatorname{Attention}(H_m+PE)),$$
$$p_m=\operatorname{softmax}(W_c\operatorname{MaskedMean}(Z_m)).$$

这些是模块示意式，矩阵左右乘按代码实际张量布局实现；padding 既在卷积前屏蔽，也在注意力与池化时屏蔽。

预测不确定性：$u_m=-\sum_k p_{mk}\log p_{mk}/\log K$。它是未校准预测熵，不能直接解释成医学错误概率。

`workflow.py`：本版本规则融合为 $w_m=q_m\max_k p_{mk}/\sum_j q_j\max_k p_{jk}$，$p=\sum_mw_mp_m$。只在可比较类别空间中计算；$q$ 是工程质量代理。原始报告始终保留。存在未解决冲突时，不将融合的最大概率视为确定答案。

`memory.py`：技能效用为简化的 $u_i=(1+s_i)/(2+s_i+f_i)$。至少有一次外部成功反馈且 $s_i>f_i$ 才参与对应模态、触发条件和作用范围的检索；不等同于 SkeMex 的分类别优势归因。

## 需要真实数据后验证的两项研究贡献

1. 冲突来源细分：目前可靠性与同目标分歧有规则，时间/用药/临床记录原因尚无标注和输入，不能从 LLM 解释直接认定归因正确。异步采集本身不判定为时间冲突。
2. 冲突驱动复核：现阶段候选窗口由清单提供；后续可用经过验证的定位工具产生候选，但不能靠诊断标签预选窗口。实验比较无复核、文本反思和定向工具复核，报告净纠错、引入错误、证据有效性、延迟与调用成本。

技能库必须在开发集构建并在测试集冻结，受试者不能跨集合。同一受试者的问题模板变化不创造新独立样本。

## 官方代码与独立改写

本次实际查看了 MedAgent-Pro 的 `Case_level.py` / `Examiner.py`，以及 HealthFlow 的 `evaluator_agent.py` / `reflector_agent.py`。版本与网址记录在 `references/sources.lock.json`。GitHub 当时没有为两仓库报告许可证，因此这里不直接复制其源码。MedAgent-Pro 当前公开示例的主循环也没有接入 Examiner，不能把文件存在当作已完整复现论文反思。

SkeMex 依据论文 §3.3–3.6 和附录实现思想改写，没有声称取得作者正式代码或复现论文分数。所有移植进 `backends/` 的代码来自用户自己的旧项目，并已记录提交版本。
