# 方法和评价口径

W1-P使用standard IVD的P95/P80/unknown弱参考与EMA teacher。standard IVD为`abs(omega - mean_valid_fluid(omega))`；弱参考的阈值和归一化只由train得到。positive连通组件过滤是弱标签处理步骤，未改变IVD定义。unknown、solid、invalid和failed path保持各自掩码，不作negative。

模型输入保留原7通道、256条迹线、24个保存帧、L16与RK4四子步；按实际保存时间跨度进行积分。精确配置见`config/`和结果[protocol.json](../results/seven-source-w1p/protocol.json)。模型结构及既有安全/来源校验沿当前生产代码。

评价对每源三个帧pool confusion counts，再等源宏平均。AUPRC定义为average precision，对相同概率分数合并后进行阶梯积分。只有三个seed齐备才报告均值和样本标准差。固定全局p=.1，未按source/seed重调阈值。

指标范围为直接observed-known支持。missing预测为NaN/count0，unknown、solid、invalid及failed单列。MFC coverage约0.334%，不是密集全场恢复；IVD参与输入和弱参考，不是独立物理真值。七源匹配Qrotstrain/lambda-ci/Gamma2审计尚无结果，Gamma1无登记二元oracle，MFC Haller与七源W2 selective保持NA。

W3用可靠P95/P80轨迹identity、两patch两随机view、最多512 embeddings，未知不作为负样本，独立投影不作为分割输出。两seed只有macro F1与IoU都改善、worst-source F1不退化才纳入；真实两seed宏指标下降，登记为no-go。三seedW1-P结果和W3两seed诊断分别保存。
