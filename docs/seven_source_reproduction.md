# 七源训练与开发评价复现

## 环境与输入

原训练/评价使用Python3.12.14、PyTorch2.7.1+cu118、CUDA11.8运行时。安装时先选择与机器匹配的PyTorch CUDA构建，再安装`requirements.txt`中的Python依赖。正式设置为FP32、单GPU、18GiB allocator上限、runtime关闭激活重算；batch64需约16.63GiB reserved。这里只提供已执行实验对应的代码和结果，不重新训练来完成发布。

当前七源核心eager依赖为NumPy、PyTorch、h5py、SciPy和PyYAML；共享模块的图形/几何分支还使用Matplotlib、scikit-image、contourpy和tqdm。可选原生Haller路径另需Numba及NumbaCS，七源W1-P不以Haller为loss。本轮MFC没有可用Haller anchor。

原始大数据和参考数组未随仓库发布。运行前需提供：

1. 六源`--input-root/<source>/`的`meta.json`、`u.npy`、`v.npy`、`mask.npy`，源顺序为boussinesq、cylinder2d、doublegyre2d、fourcenters2d、jungtelziemniak2d、pipedcylinder2d。
2. `--ticket03-root/p95p80_protocol_manifest.json`及各源`standard_ivd/`、`train_target/`、`calibration_reference/`的原metadata和数组。当前使用standard IVD，未读取旧`ivd.npy`/`label_field.npy`作为正式弱参考。
3. MFC `--mfc-flow` HDF5的u/v/derivative_valid/geometry/xdim/ydim/tdim/solver_step，以及对应`--mfc-mask`。真实501保存帧、dt=.005，train[0,400)、calibration[400,501)，完整24帧窗口。
4. `--mfc-cache`的`cache_metadata.json`、`standard_ivd.npy`、`state.npy`、`valid_fluid_mask.npy`、`invalid_mask.npy`。metadata中的train统计/阈值及原来源身份必须与输入一致。

公开的[data-metadata.json](../results/seven-source-w1p/data-metadata.json)和manifest用于解释原实验身份，不替代以上真实数组。六源train normalization在`load_dataset_view`中从实际train重新计算；MFC的train-only normalization与P80/P95来自保存的reference metadata。

## 训练入口

从仓库根执行。以下路径是由使用者提供的已有数据目录，输出使用新的独立目录；该命令运行完整训练，不是快速示例。

```bash
python run_ticket11_seven_source_core.py \
  --input-root /path/to/datasets \
  --ticket03-root /path/to/ticket03-artifacts \
  --mfc-flow /path/to/mfc/flow.h5 \
  --mfc-mask /path/to/mfc/mask.npy \
  --mfc-cache /path/to/mfc/reference-cache \
  --config config/ticket11_seven_source_core.json \
  --output runs/w1p-seed1 --seed 1 --device cuda:0 \
  --gpu-memory-limit-gib 18 --checkpoint-every-updates 25 --prefetch-batches 1
```

该入口允许fresh seed1/2。seed0为已审核完成模型复用，不能将`--seed 0`传给此入口。前25epoch batch32、后75epoch batch64，总39100updates/2Mpatches；阶段迁移和后续恢复使用完整状态。AdamW lr1e-4、weight_decay1e-6、grad_clip1，TwoStepLR第二lr5e-6，warmup60，EMA和12epoch ramp沿原实现。

## 评价与数据身份

最终stage64 checkpoint的sampling.batch_size为64，评价使用`config/mfc_mixed_w1p_seed0_batch64.json`。`config/ticket11_seven_source_core.json`的起始batch32不能直接充当最终评价配置。

已有`load_checkpoint`严格核对mode、feature schema、source、split和sampling；MFC flow/mask/cache路径及其metadata也进入dataset_config。原始checkpoint和manifest保留归档路径与身份。把旧数据搬到任意目录后，不能保证严格加载成立；应提供匹配的原输入布局及metadata。本次发布不重写原checkpoint，也不削弱现有校验。

在输入身份与路径确实匹配时，可使用[示例runs spec](../results/seven-source-w1p/evaluation-runs.example.json)，新建完整预测目录并生成报告：

```bash
python run_ticket12_seven_source_report.py \
  --input-root /path/to/datasets \
  --ticket03-root /path/to/ticket03-artifacts \
  --mfc-flow /path/to/mfc/flow.h5 \
  --mfc-mask /path/to/mfc/mask.npy \
  --mfc-cache /path/to/mfc/reference-cache \
  --config config/mfc_mixed_w1p_seed0_batch64.json \
  --runs results/seven-source-w1p/evaluation-runs.example.json \
  --output outputs/seven-source-report --infer-missing --device cuda:0
```

每seed按登记42帧/至多64patch/batch16/单view规则推理；输出须是新的目录。公开`seed*/evaluation.json`只是结果摘要，没有dense数组，不能伪装成`evaluation_report.json`用于重评分。单帧复载使用`evaluate_mfc_mixed_w1p.py --reproduce-from <completed-evaluation-root> --reproduce-frame mfc_airfoil/calibration/400`，其余输入、checkpoint、seed和batch必须与原评价匹配。

旧final-test保持封存。查看三seed汇总不需要读取任何原数据或执行训练。
