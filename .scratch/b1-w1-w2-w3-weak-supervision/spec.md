# B1/W1/W2/W3 弱监督扩展规格

**Feature slug:** `b1-w1-w2-w3-weak-supervision`  
**Status:** `ready-for-agent`；规格接缝、参数和主/辅助实验边界已由用户确认；票据已按确认的颗粒度发布  
**Source of truth:** 本文件是本 feature 的设计单一事实源；用户确认前不创建实现票据。

本规格建立在阶段 0 的迁移、数据准备和现有 B0 基线之上。阶段 0 的旧规格、旧 issues、`vendor/DeepUtils` 和既有 B0 行为均不在本 feature 中重写。Haller 的 IVD 定义与二维 Eulerian extraction 流程骨架已按 Zotero 本地全文核对；本文的离散 contour 参数、阈值和 fallback 仍是工程候选，不代表已经核实为原始论文 canonical 参数。

## Problem Statement

当前 B0 基线将两件事情耦合在一起：

1. 模型输入包含当前的 5×5 local-IVD 连续特征；
2. 训练监督来自按切片 percentile threshold 生成的旧式二值 IVD weak label，默认语义接近 legacy p85。

因此，现有结果无法回答以下问题：

- 预测能力是否依赖 local-IVD 输入通道，而不只是依赖速度、位置和时间信息；
- 在没有人工像素标签的情况下，p90/p60 三态 weak label、Haller-IVD 闭合边界 physics anchors、teacher pseudo-label 和 uncertainty gate 能否逐步替代旧式硬阈值监督；
- 弱监督信号是否在六个有效数据集、不同 IVD 尺度和 Boussinesq 的 domain/threshold drift 下仍保持可比较的泛化；
- 当 pathline window 跨越时间切分边界时，是否会把未来信息或 test Haller 标签泄漏到训练与模型选择中。

本 feature 需要形成一条可审计、可复现且逐步可验收的弱监督扩展链：从 split-contained pathline 数据和 train-only physics anchors 开始，经过 B1 输入消融、W1 基础设施、W2 uncertainty gate 和 W3 trajectory-level contrastive learning，最终在 test-only Haller-IVD GT 上逐数据集评价。端到端训练/评价缝是主验收缝；Haller anchor 缝和 split/label 缝是必须独立验证的前置扩展缝。

## Solution

### 方法矩阵

所有新实验使用同一套六数据集范围、0–50%/50–60%/60–100% 时间切分和完整窗口约束。除 B1 外，W1/W2/W3 都保留现有 5×5 local-IVD 输入。standard Haller-IVD 只用于 train anchor 或 calibration/test GT，不作为 W 方法的模型输入。

| 方法 | 模型输入 | 训练监督/损失 | 实验角色 |
|---|---|---|---|
| B0 | 现有 7 通道，含 local-IVD | legacy p85 binary weak label + 原有 BCE 路径 | 新 split 的基准；旧 checkpoint 结果仍标为 historical |
| B1 | 从 7 通道移除 local-IVD 后的 6 通道 | 与 B0 对齐的 legacy p85 binary weak label | 仅诊断性 IVD 输入消融；不作为 W1 prerequisite |
| W1-P | 现有 7 通道，含 local-IVD | train-only p90 positive、p60 negative、中间 unknown；masked anchor BCE、EMA teacher、pseudo-label 和 consistency loss | 先行的基础设施版本 |
| W1-H | 现有 7 通道，含 local-IVD | train-only Haller-IVD 闭合边界的 positive/negative anchors，边界与失败区域为 unknown；teacher/pseudo/consistency | 正式 W1 物理 anchor 版本 |
| W2 | 现有 7 通道，含 local-IVD | W1-H anchor + 多 stochastic views 的 mean/variance/entropy uncertainty gate | W1-H 的置信度扩展 |
| W3 | 现有 7 通道，含 local-IVD | W2 监督路径 + trajectory embedding projection head + 两视图 in-batch contrastive loss | 第一版 proposed method；单 GPU、最多 512 embeddings、无 memory bank |

其中 p85 仅允许出现在 B0 的监督、train patch sampling 和 diagnostics 中。W1 的正式 `Lweak` 不包含 legacy p85；W1-P 的 p90/p60 label 和 W1-H 的 Haller anchor 是彼此可区分的监督来源。

### 数据和信息流

对每个有效数据集，先按 frame index 建立半开区间：

- train：`[0, floor(0.50*T))`；
- calibration：`[floor(0.50*T), floor(0.60*T))`；
- test：`[floor(0.60*T), T)`。

给定 pathline window 长度 `t_win` 和 window start `s`，只有满足 `s >= split_start` 且 `s + t_win <= split_end` 的窗口才能进入该 split。窗口不能被截断后再归入 split。每个样本记录 split name、frame start/end、`t_win`、window step、生成版本和相关 label/anchor 来源。

训练阶段只使用 train 窗口和 train-only normalization statistics。calibration 窗口不更新模型权重；test 窗口在模型、threshold、gate、epoch 和 method 均冻结后才可进入最终评价。

### HANDOFF §8 的扩展接缝

本 feature 采用以下三条接缝，并以第三条作为主验收缝：

| 接缝 | 输入 → 输出 | 独立验收重点 |
|---|---|---|
| Haller anchor seam | 单帧 `u/v + geometry mask` → standard IVD、候选局部极大值、闭合 contour、三态 anchor、coverage/failure metadata | 只使用 fluid 区域；闭合、凸性、最小周长、outermost 和 unknown band 语义稳定；synthetic Rankine-style 与真实 train frame 可检查 |
| split/label seam | 数据集 frame index + `t_win` → split-contained windows、anchor/unknown mask、来源和版本 metadata | 0–50/50–60/60–100 边界严格；任何 window 不越界；test Haller 来源无法进入训练/调参 |
| training/evaluation seam（主验收） | pathline batch → B0/B1/W1/W2/W3 probability、loss/pseudo/uncertainty/contrastive stats；checkpoint → 六数据集 test metrics + macro | 从头训练、checkpoint round-trip、single-GPU W3 cap、test-only Haller GT、逐数据集 Precision/Recall/F1/IoU 和 macro 全链路通过 |

端到端主验收至少覆盖一套可控 synthetic fixture 和六个有效数据集的 pilot/最终运行协议；前两条接缝的失败必须 fail loudly，不能由训练脚本静默回退到旧 split 或 p85 label。

### 最终评价

最终评价使用 clean CFD 上固定生成的 Haller-IVD GT。Haller GT 的 unknown boundary、solid 和无效 frame 不计入已知流体区域的 confusion denominator，但必须报告 known coverage、unknown coverage、无效 frame 数和 Haller anchor failure 数。每个有效数据集分别报告 Precision、Recall、F1、IoU、有效样本/帧数、预测正区域比例和 GT 正区域比例；六个数据集再计算等权 macro average。Boussinesq 单独列出并作为 threshold/domain-drift stress test，不进行数据集专属 test threshold 调整。

## User Stories

1. 作为流体力学研究者，我希望 B1 只移除 local-IVD 输入通道，使 IVD 输入消融与 B0 的标签、数据和训练路径尽量可比，从而把“输入贡献”与“监督来源变化”分开。
2. 作为流体力学研究者，我希望 W1/W2/W3 仍接收当前 5×5 local-IVD 特征，以便研究 physics anchor、uncertainty 和 contrastive learning，而不是同时更换模型输入定义。
3. 作为数据工程师，我希望所有新 pathline window 完整落在所属 split 内，这样任何样本都不会通过 window 尾部跨入 calibration 或 test。
4. 作为实验负责人，我希望 split 边界按每个数据集的 frame fraction 计算，而不是依赖不同数据集不一致的绝对时间单位。
5. 作为训练负责人，我希望 p90 positive、p60 negative 和中间 unknown 能先以最小闭环运行，便于在 Haller anchor 接入之前验证 masked loss、EMA teacher、pseudo-label 和 ramp-up 的基础设施。
6. 作为训练负责人，我希望 formal W1 的 `Lweak` 可追溯到 p90/p60 或 Haller anchor，而不是悄悄复用 legacy p85 监督。
7. 作为流体力学研究者，我希望 Haller anchor 从单帧 `u/v + geometry mask` 独立产生，并记录 contour、convexity、最小周长、unknown band、failure 和版本信息，便于审查物理假设。
8. 作为流体力学研究者，我希望闭合边界内的高置信区域和边界带的 unknown 语义明确，固体区域永远不会被当成 vortex positive。
9. 作为训练负责人，我希望 train Haller anchor 可以参与 physics loss 或 anchor loss，而 test Haller 标签只在最终评价阶段读取。
10. 作为实验负责人，我希望 calibration 的用途预先写清楚，允许的 threshold/gate/epoch/method 选择不会被误用为额外训练或 test 调参。
11. 作为不熟悉仓库的实现 agent，我希望 B1、W1、W2、W3 的输入 schema、label source 和 checkpoint mode 被显式记录，从而不依赖隐含的旧配置默认值。
12. 作为复现实验者，我希望 checkpoint 同时保存 student、EMA teacher、projection head（如适用）、optimizer/scheduler、epoch、RNG、seed、split 配置、anchor hash、feature schema 和实验 mode。
13. 作为训练负责人，我希望 W2 能在多个 stochastic views 上报告 mean、variance、entropy 和 gate acceptance，而不是只以单次 teacher probability 生成 pseudo-label。
14. 作为训练负责人，我希望 W3 在第一版保持 single GPU、最多 512 trajectory embeddings、2 stochastic views、in-batch contrastive，并且没有 memory bank，以控制实现和资源边界。
15. 作为评价负责人，我希望最终模型在六个有效数据集上逐数据集输出 Precision、Recall、F1 和 IoU，并额外输出等权 macro average，而不是只给一张合并 confusion matrix。
16. 作为评价负责人，我希望 Boussinesq 的结果单独显示，使 threshold/domain drift 的影响不会被其他数据集的平均值掩盖。
17. 作为实验负责人，我希望 smoke、pilot 和 final 的 epoch、seed、method selection 和 warm-start 规则固定，避免把一次临时调试运行误报成主实验。
18. 作为实验负责人，我希望主实验全部从头训练；若需要使用旧 B0 checkpoint，只能以明确标记的 auxiliary warm-start 运行出现，且不会污染主表。
19. 作为后续 robustness 实验负责人，我希望 clean CFD Haller GT 生成一次并有 hash 固定，然后分别扰动输入、重新计算 Haller-IVD，并区分 Model 相对 clean GT 的误差与 Model 相对 recomputed Haller 的一致性。
20. 作为审计者，我希望训练日志能报告 anchor coverage、pseudo acceptance、teacher/student disagreement、W2 uncertainty、W3 pair count 和每个 split 的样本计数，从而发现 label collapse 或泄漏。
21. 作为服务器运行者，我希望长时间 pilot/final 运行遵守 SHU-server 的单 GPU 默认、数据分区缓存和指定 Python 环境约束，不在共享根分区产生临时产物。
22. 作为维护者，我希望本 feature 的实现不修改 `vendor/DeepUtils`、阶段 0 迁移结果和旧 baseline specification/issues，从而保持旧基线可复核。
23. 作为评审者，我希望所有 Haller 原始论文依据在 Zotero 全文或用户提供材料核实前标记为“待核实”，工程默认值不会被写成已证实的 canonical 参数。

## Implementation Decisions

### 1. 范围、数据集和冻结边界

- 六个有效数据集固定为：`boussinesq`、`cylinder2d`、`doublegyre2d`、`fourcenters2d`、`jungtelziemniak2d`、`pipedcylinder2d`。
- `forceddampedduffing2d` 因 time freeze/label degeneration 不进入新实验池。旧 cavity 等历史 strict-zero-shot 辅助结果不替代六个核心数据集。
- 阶段 0 的迁移、数据清理和 vendor 代码是前置事实，不在本 feature 重新执行或修改。
- 现有 backbone 的 pathline extractor、geometry mask 和 5×5 local-IVD 计算作为稳定输入基础。标准 Haller-IVD 不替换 W1/W2/W3 的 local-IVD 输入。
- B1 是诊断性输入消融，不是 W1 的依赖项，也不因 B1 成绩而改变 W1/W2/W3 的路线。

### 2. Split、窗口和 normalization 契约

- 每个数据集独立计算 `s50=floor(0.5*T)`、`s60=floor(0.6*T)`，使用半开区间 train/calibration/test。
- 所有 pathline window 必须满足 `start >= split_start` 且 `start + t_win <= split_end`。`window_step` 只能影响窗口枚举，不能放宽完整窗口约束。
- 任何 split 长度不足以容纳一个完整 window 时立即报错，并包含数据集名、`T`、边界和 `t_win`；不得静默缩短 window 或借用相邻 split。
- 训练 normalization statistics 只从 train fluid cells/windows 计算。calibration/test 只消费冻结统计量。
- metadata 至少保存 split frame ranges、`t_win`、window step、采样配置、feature schema、label source、生成版本和 hash。label source 必须能区分 `legacy_p85`、`local_p90_p60`、`haller_anchor_train`、`haller_gt_calibration` 和 `haller_gt_test`。
- p85 的使用边界是：B0 监督、train patch sampling、diagnostics。patch sampling 产生的 seed/pool membership 是采样来源，不得被当作 W1 formal `Lweak` 的监督标签。

### 3. Haller-IVD anchor 候选流程

Zotero 本地全文已核对 Haller 的 IVD 定义以及“局部极大值 → 附近闭合等值线 → 最外围、凸性和最小弧长约束”的二维 Eulerian extraction 流程骨架。以下是项目为离散网格定义的工程候选流程，用于把接口和测试先定义清楚；具体数值仍不得表述为 canonical paper parameters：

1. 输入单帧 `u/v` 和 geometry mask，仅在 fluid cells 上计算二维瞬时 vorticity；
2. 以 fluid 区域的 domain-average vorticity 构造 standard IVD 候选场 `abs(omega - mean_fluid(omega))`；
3. 在 fluid 区域寻找局部极大值，围绕每个候选峰搜索一组闭合 IVD contours；
4. 对 contour 做 solid crossing、闭合性、convexity defect 和 minimum perimeter 检查；
5. 对嵌套候选保留每个 vortex 候选的 outermost valid contour；多个不冲突候选的 positive/unknown 区域取并集；
6. 将 contour 内部、boundary unknown band、远离 contour 的低 IVD fluid cells 分成 positive、unknown、negative；无法可靠分类的 fluid cells 维持 unknown；solid cells 永不进入 positive/negative confusion。

已确认的流程不以 p85 fallback 代替失败的 Haller contour。训练 anchor 失败时整帧 fluid anchor 为 unknown 并记录 failure；test GT 失败时将该 frame 标记为 invalid、从该 frame 的 metric denominator 排除并报告 failure，不制造全负 GT。

Haller train anchor 和 calibration/test GT 使用分离的 artifact/source 标识，不能覆盖既有 local-IVD label，也不能通过同一加载默认值被训练脚本自动读取。

### 4. Haller 参数与证据边界

下表是用户确认的工程实现参数，不是已核实的 Haller canonical 参数。Zotero 全文支持 IVD 公式和上述流程骨架；项目的具体数值映射（例如 0.10、`8*max(dx,dy)`、`2*max(dx,dy)` 和 p60）仍待进一步核对其 canonical 对应关系。实现必须把这些参数写入版本化 metadata，并保留可审计的 sensitivity diagnostics。

| 参数 | 已确认工程值 | 语义 |
|---|---|---|
| domain mean | 每帧 fluid、非 solid 单元的 vorticity mean | standard IVD 为 `abs(omega - mean_fluid(omega))` |
| local maximum | fluid 8-neighborhood | 不额外引入未预注册的 prominence/峰间距过滤 |
| contour levels | 每个峰从 `1.0 * peak` 到 `0.1 * peak` 的 32 个线性 level | level 范围和数量固定并写入 metadata |
| convexity defect | `(A_hull-A_contour)/A_hull <= 0.10` | 使用物理面积；保留 sensitivity diagnostics |
| minimum perimeter | `P >= 8 * max(dx, dy)` | 使用物理长度，不按数据集静默调整 |
| outermost | 每个局部峰的最外层 valid nested contour | 合法候选的 positive/unknown 区域按显式 union 规则合并 |
| unknown band | contour 法向/形态学带宽 `2 * max(dx, dy)` | 按物理长度转换为网格 morphology 半径 |
| low-IVD negative | contour/band 外且 standard IVD 不高于该帧 fluid p60 | p60 按帧计算；不使用 test 结果调节 |
| failed contour fallback | train：整帧 fluid unknown；test：invalid frame，metric 排除但计数 | 不允许全负或 p85 静默 fallback |

positive anchor 只包含 contour 内且不在 unknown band 的 fluid cells；unknown 包含 boundary band、未通过可信性检查的候选区域和失败帧；negative 只来自满足低 IVD 条件且远离 boundary 的 fluid cells。最终 Haller GT 的 metrics 只在已知 fluid cells 上计算，coverage 与 unknown 带宽同时报告。

所有 Haller artifact 至少携带算法版本、完整参数、参数 hash、输入字段/geometry mask hash、solid policy、frame index、failure count、positive/unknown/negative coverage 和来源（train anchor 或 calibration/test GT）。

### 5. B1 诊断性 IVD 输入消融

- B1 只在模型输入适配层移除 local-IVD 通道，使用剩余位置、时间、distance、`u/v` 六通道；下游模型通过显式 input schema/config 得到 `in_channels=6`。
- B1 仍使用 B0 对齐的 legacy p85 weak label 和同一新 split，便于把输入通道差异作为主要变量。
- B1 不需要生成 Haller anchor，也不作为 W1-H 的前置条件。
- B1 有独立 mode、checkpoint 和结果目录，结果标记为 diagnostic。除非用户在规格确认时另行授权，B1 不进入 final headline 的“best baseline”候选池。
- 不修改 vendor 实现；适配层必须在输入侧完成 channel selection，并对 channel order/schema 做 fail-loud 校验。

### 6. W1-P 基础设施与 W1-H physics anchor

#### W1-P

- 在 train frames 上按当前 local-IVD 语义生成 p90 positive、p60 negative 和中间 unknown。solid cells 从 anchor loss 中排除。
- positive 必须经过既有的有效区域/最小面积语义；`p60 < local-IVD < p90` 进入 unknown，不强行二值化。
- student 和 EMA teacher 初始权重相同。student 更新后再更新 teacher；初始 EMA decay 候选为 `0.99`，具体调度必须记录。
- anchor loss 为已知 p90/p60 区域的 masked BCE；unknown 区域可接受 teacher 产生的 pseudo-label，但只在 `teacher probability >= 0.90` 或 `<= 0.10` 时接受。
- pseudo-label 和 consistency loss 的权重从 0 ramp 到目标值，建议 ramp-up 为 12 epochs，允许范围为约 10–15 epochs。
- 训练日志报告 known anchor coverage、unknown coverage、pseudo acceptance、teacher/student disagreement 和每类有效 cell 数。

#### W1-H

- 保留 W1-P 的训练/teacher 基础设施，但将正式 anchor source 换成 Haller train anchor。
- W1-H 的 formal `Lweak` 由 Haller known positive/negative anchor、teacher pseudo-label 和 consistency 项组成；legacy p85 不进入其中。
- Haller boundary band、failed contour 和 solid 都使用 unknown/ignored mask，而不是把不确定区域当 negative。
- W1-P 先作为基础设施 smoke/pilot 变体跑通；W1-H 才是物理 anchor 版本。两者的 label source、coverage 和 checkpoint mode 必须可区分。

### 7. W2 confidence/uncertainty gate

- W2 从 W1-H 继承输入、split、anchor 和 teacher 语义。
- 对同一候选 unknown region 使用 3 次 stochastic teacher view，计算 mean probability、predictive variance，并报告 Bernoulli entropy 作为诊断量。
- 冻结 gate 为：`mean >= 0.90` 作为 positive、`mean <= 0.10` 作为 negative，并要求 predictive variance 不超过单一全局 gate；不满足任一条件的样本保持 unknown。Bernoulli entropy 只作为诊断量。
- uncertainty gate 不按数据集分别调参；threshold/gate 的 calibration 用途见下文，test 不参与 gate 选择。
- 训练日志至少记录 view 数、mean/variance/entropy 分布、gate acceptance、正负 pseudo 比例和 disagreement。

### 8. W3 trajectory-level contrastive 扩展

- W3 在 local adapter 中取得每条 trajectory 的 pre-classifier embedding，再接 projection head；不修改 vendor/DeepUtils。
- 每条 trajectory 生成 2 个 stochastic views；同一 trajectory 的两视图构成 positive pair，in-batch 其他样本构成 negatives。
- Haller anchor 或 W2 pseudo-label 只能在 known/reliable 状态下形成语义 pair；unknown 不参与语义正负 pair。
- 第一版采用 single GPU、最多 512 个送入 contrastive loss 的 trajectory embeddings（两视图合计）、2 stochastic views、in-batch contrastive、无 memory bank、无跨 GPU gathering。
- projection dimension 固定为 64，temperature 固定为 0.1；contrastive loss 权重从同一 10–15 epoch ramp-up 开始，实际 pair count、有效 batch 数和被 unknown 排除的数量都要记录。

### 9. Checkpoint、模式和可复现性

每个新 checkpoint 必须能恢复对应的训练/评价语义，至少包括：

- `format_version`、method mode、feature schema/channel order、dataset/split 配置、`t_win` 和 sampling 配置；
- student model；W1/W2/W3 的 EMA teacher；W3 projection head；
- optimizer、scheduler、epoch、global step、loss/metric summary；
- Python/torch 相关运行信息、seed 和 RNG states；
- Haller anchor algorithm/version/parameter hash（适用时）；
- calibration policy、threshold/gate 配置和 warm-start 标志。

主实验 `warm_start_aux=false` 且从头训练。旧 B0 checkpoint 只能作为明确命名、单独报告的 `warm_start_aux=true` 辅助实验；它不能悄悄成为主实验初始化，也不能把其旧 split/旧 normalization 带入新 split。

### 10. Smoke、pilot、final 和选择规则

- smoke：每个实现 mode 运行 5–10 epochs；至少检查 synthetic fixture 和一个真实 train-frame/小样本路径。W1-P smoke 先只证明基础设施可运行，W1-H smoke 证明 Haller artifact 能被训练消费。
- pilot：B0、B1、W1-P、W1-H、W2、W3 等所有预注册方法各运行 50 epochs、1 seed；unsupervised ramp-up 约 12 epochs。pilot 输出用于发现失败、比较候选和冻结最终规则。
- final：只运行 B0、一个 best baseline 和一个 proposed method，最终 epoch 固定为 130，3 seeds 固定为 `[0, 1, 2]`。B1 仍是 diagnostic，不进入这三个 headline slot。
- best baseline 的候选池固定为 W1-H/W2，proposed method 固定为 W3；best baseline 使用 calibration 规则选择，tie-break 在查看 test 之前冻结。
- calibration Haller GT 可用于预注册候选中的全局 prediction threshold、W2 uncertainty gate 和 best-baseline method 选择；final epoch 不由 calibration 改动。calibration 不更新模型权重、normalization、Haller contour 参数或 pseudo-label 生成规则，不进行 per-dataset test tuning。
- test Haller GT 只在最终 checkpoint、threshold、gate、epoch、method 和 seed 均冻结后读取，用于最终评价和报告。
- pilot/final 主实验均从头训练；B0 historical checkpoint 和 warm-start 结果单列 auxiliary。

### 11. 六数据集评价和 Boussinesq stress test

- 对六个有效数据集分别在 test split 的有效 fluid/known Haller cells 上累计 TP、FP、FN，并计算 Precision、Recall、F1、IoU。
- 汇总报告必须包含 dataset name、seed、checkpoint epoch、global threshold、有效 frame/cell/sample count、Haller known/unknown coverage、invalid/failure count、预测/GT positive area ratio。
- macro average 是六个数据集指标的等权算术平均；不能用所有像素混合后的 micro score 替代 macro。3 seeds 报告 mean 和 standard deviation。
- Boussinesq 额外作为 threshold/domain-drift stress test 展示，使用与其他数据集相同的冻结 global threshold/gate，不能从 Boussinesq test 反向改参数。
- evaluation loader 必须明确接收 `haller_gt_test` source，不能默认加载 train anchor 或 legacy p85 label；任何 test label 被训练、pseudo、threshold/gate 或 method selection 访问都应 fail loudly。

### 12. 后续 robustness 协议

robustness 是核心 W1/W2/W3 闭环之后的后续实验，但本规格先固定 seam 和可追溯协议：

- clean CFD 的 Haller-IVD GT 只生成一次，保存输入/geometry/参数 hash，作为固定 GT；
- 对 `u/v` 输入加入零均值 Gaussian noise，solid/geometry mask 不加噪；不做 clipping。噪声标准差按每数据集 train fluid speed RMS 的 `alpha ∈ {0.01, 0.05, 0.10}` 缩放，不使用 test statistics；
- downsampling 固定为 anti-aliased、mask-aware 的 factor 2 和 factor 4 面积/低通降采样，再插值回模型网格。重建后的 `u/v` 用相同固定参数重新计算 local-IVD 和 Haller-IVD；
- 同时报告三种关系：Model vs clean Haller GT、Model vs recomputed Haller-IVD、recomputed Haller-IVD vs clean Haller GT，用来区分模型误差和 physics label drift；
- noise 注入位置、无 clipping、downsampling 的 mask-aware 聚合/插值细节和 recompute 参数均在本规格中冻结，不能根据 test 结果修改。

### 13. 已确认的规格决策

用户已确认以下工程边界，tickets 必须按此冻结实现：

1. Haller 参数采用 §4 的 domain mean、8-neighborhood、32 个线性 levels、0.10 convexity defect、`8 * max(dx,dy)` minimum perimeter、outermost、`2 * max(dx,dy)` unknown band、frame p60 low-IVD negative 和 train-unknown/test-invalid fallback；这些是工程参数，文献 canonical 依据仍待核实。
2. calibration 可以读取 calibration Haller GT，用于全局 prediction threshold、W2 uncertainty gate 和 best-baseline method selection；不能改训练权重、normalization、Haller 参数、pseudo-label 规则或 final epoch，不能做 per-dataset test tuning。
3. final 固定 130 epochs、seeds `[0,1,2]`；B0、best baseline（W1-H/W2 中选择）和 proposed W3 进入 headline；B1 永远单独 diagnostic。
4. W2 固定 3 stochastic views，以 mean/variance 做 gate，mean confidence 为 0.90/0.10，variance 为主 uncertainty，entropy 仅诊断；uncertainty threshold 为全局 calibration 选择。
5. W3 固定 512 个两视图合计 embeddings、2 views、projection dimension 64、temperature 0.1、single GPU、in-batch contrastive、无 memory bank。
6. robustness 固定 clean-Haller GT hash、`u/v` Gaussian noise（train speed RMS 的 0.01/0.05/0.10、无 clipping、solid 不加噪）、factor 2/4 anti-aliased mask-aware downsampling/reconstruction 和同参数 recompute。
7. B1 old/new split 结果完全分开报告；B1 diagnostic 不进入 final headline，也不成为 W1 前置。

## Testing Decisions

测试以接缝的可观察行为和端到端产物为中心，不把 vendor 内部实现细节作为测试目标。已有阶段 0/B0 回归测试保持通过；新测试应以增量方式覆盖本 feature。

### Haller anchor seam

- 用可控的 synthetic Rankine-style velocity field 检查 standard IVD、局部峰、闭合 contour、outermost 选择、solid exclusion、convexity defect、minimum perimeter 和 unknown band 的可解释行为。
- 用至少一个包含非理想轮廓、固体邻近区域和无合法 contour 的 fixture 检查：solid 不变 positive、失败返回 unknown/invalid contract、failure count 和参数/hash metadata 完整。
- 对真实数据只允许读取 train frame 做 preview/smoke；测试不得为了调 contour 参数读取 test Haller frame。
- 参数变动必须能在 artifact metadata 和 sensitivity diagnostics 中观察到；不能有隐式 p85 fallback。

### Split/label seam

- 为边界前、边界处和边界后的 frame start 构造 window fixture，验证 train/calibration/test 的半开区间和 `start + t_win <= split_end` 约束。
- 六个有效数据集的 metadata 校验 frame range、完整窗口数、window step、normalization source 和 label source。
- 测试应故意尝试跨 split 的 start，并确认 fail loudly；尝试把 `haller_gt_test` 作为训练 label source 时也必须 fail loudly。
- 验证 p85 sampling source 与 W1 formal loss source 分离；W1-P/W1-H 的 unknown mask 不得被二值化成静默 negative。

### Training/evaluation seam（主验收）

- CPU synthetic smoke 覆盖 B0/B1/W1-P/W1-H/W2/W3 的 batch shape、probability range、channel schema、masked loss、EMA update、ramp-up、W2 statistics/gate、W3 projection/pair cap 和 checkpoint round-trip。
- B1 必须验证移除的确是 local-IVD 通道且 channel order 与 `in_channels=6` 一致；W1/W2/W3 必须验证 local-IVD 通道仍在输入中。
- W3 测试必须证明两 stochastic views、最多 512 total embeddings、in-batch pairs 和无 memory bank；超 cap 时应可预测地截断或 fail，并记录有效 pair count。
- 主端到端 fixture 从 split-contained pathline batch 开始，完成一次训练/保存/恢复/评价，最终产出每个 fixture dataset 的 Precision、Recall、F1、IoU、coverage 和 macro report。
- six-dataset pilot/final 的评价必须逐数据集输出四项指标和 macro；Boussinesq 必须有单独 stress-test 行；test Haller source 必须只在 evaluator 阶段读取。
- calibration 选择、threshold、gate、epoch 和 method selection 的日志应能证明没有访问 test metrics；最终 checkpoint metadata 应能复现该选择。

### 运行层级

- 本地优先运行 CPU targeted tests 和小型 synthetic fixture；不得为了单元测试修改共享服务器环境。
- 长时间 pilot/final 在 SHU-server 上运行，先确认工作目录、Python 解释器、CUDA 和 GPU 状态，临时文件/缓存/日志位于 `/data/xuziyi/` 数据分区。
- smoke 使用 5–10 epochs；pilot 使用所有预注册方法、50 epochs、1 seed；final 仅三个 headline slots、选定 100/130 epochs、3 seeds。
- 每次新票据完成后运行该票据的 targeted verification；全部 feature 完成时运行既有 B0 回归和本规格的主端到端验收。

## Out of Scope

- 重新迁移、清理或重构阶段 0 数据，修改 `vendor/DeepUtils`，修改旧 `.scratch/vortex-extraction-pipeline/spec.md` 或旧 issues。
- 把 standard Haller-IVD 作为 B1 之外的模型输入，或以 standard IVD 替换 W1/W2/W3 当前 5×5 local-IVD 输入。
- 将 legacy p85 放入 formal W1 `Lweak`；p85 只保留给 B0、patch sampling 和 diagnostics。
- 把 `forceddampedduffing2d` 或其他被阶段 0 排除的数据集重新加入六数据集核心评价。
- 人工像素标注、Vatistas synthesis、LAVD/time-integrated Haller、三维 vortex segmentation、GUI/LIC 产品化或新的数据源迁移。
- W3 的 multi-GPU、memory bank、跨卡 negatives、超过 512 embeddings、超过 2 views 或更复杂的 temporal contrastive 机制。
- 使用 test Haller 标签训练、生成 pseudo-label、调 threshold/gate/loss、选择 method/epoch/seed，或用 test 结果反馈 Haller 参数。
- 把旧 B0 checkpoint warm-start 当成主实验；warm-start 只允许单独标注的 auxiliary run。
- 在核心闭环完成前进行完整 robustness 网格搜索或生产部署。本规格只定义后续 robustness 的固定 GT、扰动、recompute 和比较 seam。
- 在 Haller 原始论文全文或用户授权的可靠材料核实前，把候选 contour/convexity/minimum-perimeter 参数表述为论文已证实的 canonical 实现。

## Further Notes

### 文献与证据状态

- Zotero 本地检索已确认存在 Haller 相关条目：`Defining coherent vortices objectively from the vorticity`，Zotero key `L2PX3NQX`，作者/年份信息与用户所指的 2016 条目相符。已读取本地附件全文，核对 IVD 公式 `|ω−ω̄|`、二维 Eulerian vortex 的 nested level-set/outermost 定义，以及局部极大值、闭合 contour、凸性缺陷和最小弧长检查的流程骨架；论文中的 convexity bound（文中示例为 `10^-3` 或更低）和 minimum arc length 仍是数据依赖/文献语境，不能直接等同于本项目的工程参数。
- 因此本规格中 Haller 的公式与流程骨架标记为“已由 Zotero 全文核对”，而项目离散参数（0.10 convexity defect、`8*max(dx,dy)` minimum perimeter、`2*max(dx,dy)` unknown band、frame p60 和 fallback 的工程映射）继续标记为“待核实”；不把它们写成论文 canonical 实现。
- Zotero 中已确认存在 VortexTransformer 条目；本 feature 只沿用当前仓库的 backbone/input seam，不把论文条目当作未经实现验证的额外接口。
- 当前无需外部网页；后续若冻结 Haller 数值参数，仍须补充逐项文献对应关系或由用户明确授权其他可靠来源，不把待核实的工程映射写成验收事实。

### 变更与票据流程

- 本文件写入后等待用户确认：三条接缝、Haller 参数候选、calibration 允许用途、final epoch、W2 gate、robustness 定义、W3 cap 解释，以及主/辅助方法边界。
- 用户已确认规格并在同一上下文确认 blockers-first tracer-bullet 票据的颗粒度和依赖边；11 张 issue 已逐票写入新 feature issue 目录。
- 每张实现票独立可验证；后续 `/implement` 在新上下文中执行，并在票内使用 `/tdd` 与 `/code-review`。每完成一票，更新 HANDOFF §12，并在 §11 追加简洁变更日志。
- 本 feature 的主验收不是某个单独的 anchor 数值，而是 split-safe、test-only Haller GT、六数据集逐数据集指标 + macro 的训练/评价端到端闭环。

### 票据发布状态

用户已确认上述规格以及 11 张票的颗粒度和依赖边。当前阶段完成 `/to-tickets`；后续按每张 issue 的路径在新上下文中逐票 `/implement`，每票内部遵循 `/tdd` 与 `/code-review`。
