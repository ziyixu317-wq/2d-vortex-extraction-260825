# 02: 固体几何掩膜（geometry）

**What to build:** 从原始仿真场自动识别固体几何：|v|<ε 逐帧取与 + 连通域标记，输出随数据集 (T,Y,X) 存储的掩膜，供迹线提取（种子排除/截断）与弱标签（IVD 置零）共用；圆柱定位为不与壁相连的孤立连通块；无障碍物数据集输出空掩膜、代码路径不变。

**Blocked by:** None (can start immediately)

**Status:** done

- [x] 掩膜可视化目检：圆柱、台阶管道壁面、死区被正确标记
- [x] 圆柱定位结果与数据集元数据一致（radius=0.0625）
- [x] 无障碍物输入路径不变（输出空掩膜不报错）
- [x] 掩膜连通性属性测试通过

## 完成记录

- **2026-08-25 完成**（commit `7b3dcb9` 代码 + `8e3cb50` 之后的 docs 收尾 commit）。
- **做了什么**：
  - 新增 `geometry.py`（262 行）：`static_mask_from_speed`（速度模 <ε 逐帧取与，ε 默认 1e-5）、`label_components`（自写并查集两遍扫描连通域，8/4 邻接，**无 scipy**，遵守 HANDOFF §2 依赖清单）、`component_stats`（每块 cells/bbox/质心物理坐标/touches_border/r_eff/r_phys）、`locate_cylinders`（孤立连通块，默认无尺寸判据，`min_block_cells` 为显式收紧选项）、`build_geometry_mask` 主入口（落盘 `mask.npy` (T,Y,X) uint8 每帧相同 + `geometry_meta.json`）、`plot_mask` 目检图（imshow/contour 物理坐标系 extent，与圆柱拟合圆对齐）、CLI（h5py 直读中文路径）。
  - 新增 `tests/test_geometry.py`（10 项测试）+ `pytest.ini`（pythonpath 配置）：合成数据属性测试（无障碍物空掩膜路径、逐帧取与排除瞬态零速、连通性 cells 守恒/8 邻接合并、圆柱定位圆心/半径容差、min_block_cells 显式收紧行为、落盘时间不变性）+ 真实数据已知事实测试（28213 格/41.8%、4 连通块/2 孤立圆柱、圆心 (0,0)/(3,1)、r_phys vs radius=0.0625 差 <1 格、落盘 (T,Y,X) 每帧相同）。
- **验证证据**：
  - `python -m pytest -q` → **18 passed**（含票 01 vendor 8 项 + 本票 10 项；本地 CPU，Python 3.12.3）；
  - CLI 实测：`python geometry.py <nc> --out-dir outputs/geometry --visualize outputs/geometry/mask_overview.png --frame 1000` → 固体 28213 格（41.80%，与 HANDOFF §2 一致）、4 连通块、圆柱 id=2 圆心 (-0.0043,-0.0034) r_phys=0.0629（vs 元数据 0.0625，差 0.6%）、圆柱 id=4 圆心 (3.0011,1.0034) r_phys=0.0641（差 2.6%）；
  - 产物：`outputs/geometry/{mask.npy (1501,150,450) uint8, geometry_meta.json, mask_overview.png}`（outputs/ 被 gitignore，走 Kaggle Dataset）；
  - 数值核验代替视觉目检（本会话模型无法渲染图像）：4 连通块清单（两块矩形壁面 + 两个孤立圆角方块）、圆柱边界速度过渡带（0.007–0.48 中间值 → 表面层插值证据）、Re=160=U·D/ν 自洽。
- **/code-review 结果与处置**（双轴并行评审，commit 前修复）：
  - Standards 轴：无硬违规。判断性处置：① plot_mask 物理/格坐标混用（拟合圆画错位置）→ 已修（extent 物理坐标对齐）；② 速度模重复 → 提取 `speed_magnitude` 公共函数；③ 测试死代码（transient/mask 未用）→ 已删；④ assert 输入校验 -O 剥离 → 改 raise ValueError。
  - Spec 轴：① 测试期望值非独立来源（4 块/2 圆柱/圆心为本票实测）→ 已按 §11 协议回写 HANDOFF §2（固体几何行），测试 docstring 注明权威来源；② `min_block_cells` 尺寸过滤扩展了规格"圆柱 = 孤立连通块"定义 → 默认改 1（规格字面），参数保留为显式收紧选项，测试改为验证默认/显式两种行为；③ plot_mask 坐标 bug（同 Standards ①）已修。
  - 掩膜/连通域/壁接触/落盘/空掩膜路径经双轴确认正确。
- **事实回写（HANDOFF §2/§4）**：数据集含**两个**圆柱（入口 (≈0,0) 与拐角后 (≈3,1)）；radius=0.0625 语义 = 圆柱真实半径（Re 自洽），零速区为内切区（表面层 ≈1 格），物理半径 ≈ 面积等效 + 1 格。
- **遗留**：无阻塞性问题。掩膜产物走 Kaggle Dataset 不进 git；下一步按 frontier：票 03（迹线提取，依赖 02）与 04（弱标签，依赖 02）可启动。
