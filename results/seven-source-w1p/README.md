# 七源 W1-P 三 seed 开发结果

本轮选择 W1-P 作为七源核心方法。三个 seed 均完成 100 epoch / 39,100 updates / 2,000,000 sampled patches：前25 epoch batch32，后75 epoch batch64。seed0复用此前已完成并核对的模型，seed1/2新增训练均正常退出0。

## 权重和逐 seed 结果

| Seed | checkpoint | 七源 calibration F1 | IoU | AUPRC | worst-source F1 |
|---|---|---:|---:|---:|---:|
| 0 | [epoch_0100.pt](seed0/epoch_0100.pt) | 0.963704 | 0.934079 | 0.990209 | 0.849180 |
| 1 | [epoch_0100.pt](seed1/epoch_0100.pt) | 0.961985 | 0.930297 | 0.991306 | 0.866184 |
| 2 | [epoch_0100.pt](seed2/epoch_0100.pt) | 0.961925 | 0.931347 | 0.989448 | 0.836406 |

seed0作为默认展示：它在本轮三个seed中macro F1/IoU最高。这是已观察到的开发结果选择；seed1的AUPRC和worst-source F1更高，不将任一seed称为所有指标唯一最优。每个seed目录同时提供100行[history](seed0/history.json)、逐帧/逐源`evaluation.json`及单帧`reproduction.json`。

七源三seed macro F1 = **0.962538 ± 0.001011**，IoU = **0.931907 ± 0.001953**，AUPRC = **0.990321 ± 0.000934**（样本标准差）。[three-seed-summary.json](three-seed-summary.json)保留train/calibration下原六源、MFC、七源的Precision、Recall、F1、IoU、AP，逐seed及均值/标准差。

## 评价范围与证据

每个模型7源×3train+3calibration共42帧，每帧至多64个按空间行序均匀选取的可用patch；fourcenters有49个。batch16、单个登记stochastic grouping view、固定全局p=.1。帧、patch坐标、view seed规则和两阶段预算见[protocol.json](protocol.json)。

指标仅基于直接observed-known支持的P95/P80弱参考。MFC三calibration帧有效流体共10,684,140格次，直接覆盖35,652（0.3336909%）；known positive缺651,321格次。missing、unknown、solid、invalid、failed保持独立计数，未作为负例。IVD同时是输入和操作性弱参考组成，不提供独立物理准确性。旧final-test未访问。

四项新增训练的完整checkpoint/history/迁移状态已完成CPU审计，实际新模型各42帧GPU推理完成。三份核心checkpoint均在MFC calibration frame400按相同坐标、batch16和view seed做实际reload/reproduce，每次11,884个共同观测流体格点，最大/均值概率差、count变化与finite支持变化均0；范围为该单帧。详见各seed的`reproduction.json`及[checkpoint-verification.json](checkpoint-verification.json)。

## W3 科学判读

W3两seed均完整训练及评价。逐帧共同observed-known calibration支持上，W3−W1-P的macro F1变化为−0.003333388/−0.003632069，IoU变化为−0.006761023/−0.006747147；worst-source F1变化为+0.014741514/+0.009499810。按原规则为 **no-go / no_repeatable_gain**，不启动条件seed2。

两seed均改善cylinder2d、pipedcylinder2d和MFC，均退化boussinesq、doublegyre2d、fourcenters2d和jungtelziemniak2d。保留[实际helper输入输出](w3-diagnostic-summary.json)，不生成W3三seed统计。独立projection不作为分割输出；W2对此分支为NA，MFC无可用Haller。

## 文件与复现

- 三个`epoch_0100.pt`是原始完整checkpoint的逐字节副本，合计33,003,911字节，含student/EMA/AdamW/scheduler/RNG和原来源metadata。
- [data-metadata.json](data-metadata.json)从真实checkpoint提取源身份、split、P80/P95与MFC train normalization。六源normalization不在checkpoint中，由实际train数据重新计算；不填造数值。
- [p95p80_protocol_manifest.json](p95p80_protocol_manifest.json)为原manifest副本，其既有身份已与三个checkpoint核对相符。里面的原路径属于来源记录。
- `evaluation.json`保留帧级计数、AP及完整支持分母；未发布dense probability/count数组，因此该文件是可读结果，不是原完整预测目录。缺少原始概率时无法从confusion counts重算AP。
- 原始流场、参考数组、完整日志、临时测试和W3权重未纳入精选结果包。重新推理与重评分的外部输入、路径匹配限制见[复现说明](../../docs/seven_source_reproduction.md)。
