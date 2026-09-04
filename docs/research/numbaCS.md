# NumbaCS（`numbacs`）调研笔记

调研日期：2026-09-03。这里将用户所说的 “NumbaCS” 解释为官方项目 **NumbaCS (Numba Coherent Structures)**；它的发行包名是 `numbacs`，官方仓库是 [alb3rtjarvis/numbacs](https://github.com/alb3rtjarvis/numbacs)。这是名称大小写/发行包名的歧义，以下结论均针对该项目，不代表已经决定把它加入本项目依赖。[官方 README](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/README.md#L1-L26)

## 结论摘要

- 官方 0.2.0 源码把 `rotcohvrt` 实现为：局部峰值筛选 → 多个标量等值线 → 闭合/长度/包含峰值/凸性判定；等值线后端是 **ContourPy**，不是 `skimage.measure.find_contours`。[椭圆结构提取源码](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/elliptic.py#L7-L119)
- 对重复的 32-level 场景，单次 `rotcohvrt` 会创建一个 ContourPy generator，并对 32 个 level 分别调用 `c.lines(level)`；源码没有跨 level 或跨 frame 的几何缓存。因此它可作为替代后端/算法基线，但不是现有 `find_contours` 的 Numba 加速包装器。这是由源码循环结构得到的工程推断。[同上](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/elliptic.py#L75-L103)
- 连通分量只在 FTLE ridge 路径中出现：先用 Numba 计算 ridge 点，再用 `scipy.ndimage.label(generate_binary_structure(2, 2))` 标记 `ridge_bool`，并按最小点数过滤；它不是一个面向任意二值 mask 的通用 connected-components API。[ridge 提取源码](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/ridges.py#L157-L229)
- 加速重点在粒子积分、插值和数值诊断：`@njit(parallel=True)`/`prange`、numbalsoda 的 DOP853/LSODA，以及 Numba-compatible interpolation。轮廓后处理本身仍调用 Python 层的 ContourPy 和 SciPy `ConvexHull`。[依赖说明](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/README.md#L189-L191)、[积分源码](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/integration.py#L1-L120)

## 轮廓提取：`rotcohvrt`

`rotcohvrt(lavd, x, y, r, ...)` 接受形状为 `(nx, ny)` 的 LAVD/IVD 场。默认 `min_val=-1` 时用场的 80th percentile 筛选峰值，默认起始等值线为 70th percentile，终止等值线为场最大值，默认扫描 20 个等值线 level。源码把场转置为 `lavd.T` 后交给 `contourpy.contour_generator(x=x, y=y, z=lavd.T)`。[函数签名和参数](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/elliptic.py#L7-L80)

峰值由 `max_in_radius` 完成：它反复取全场 `max/argmax`，再把峰值周围的轴对齐矩形区域置零；函数要求传入副本以避免覆盖原数组。由这一循环结构可推断，候选峰越多，重复扫描场的成本越高；它也不同于一次邻域卷积或单次 8-neighborhood 局部极大值扫描。[峰值工具源码](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/utils.py#L814-L878)

每个 level 调用一次 `c.lines(clevels[k])`。候选 contour 必须通过以下条件：

1. 首尾坐标精确相等（没有闭合容差）；
2. 若设置 `min_len`，弧长必须超过阈值；
3. contour 内必须有一个尚未匹配的峰值，且 `pts_in_poly` 返回的是第一个命中的峰值；
4. 用 SciPy `ConvexHull` 与 shoelace 面积计算相对凸性缺陷；
5. 返回的是凸包顶点组成的 `ch` 和一个中心点，并删除该峰值，避免同一峰值重复产出。[判定与返回值源码](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/elliptic.py#L80-L119)、[弧长/点在多边形工具](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/utils.py#L1057-L1209)

对本项目的直接影响：

- `nlevs=32` 会带来 32 次 ContourPy 等值线提取调用；generator 只在这一次函数调用内复用，frame 之间不会复用。
- 输出不是原始 marching-squares 顶点，而是凸包后的 contour；若后续需要原始边界、孔洞或精确闭合语义，不能直接当作 drop-in 替换。
- `convexity_method` 的 docstring 声称支持 `"convex_hull"` 和 `"angle"`，但 0.2.0 函数体没有分支读取该参数，实际路径始终调用 `ConvexHull`。这是采用前必须先写回归测试的兼容性风险。[源码签名/实现](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/elliptic.py#L12-L42)、[实际判定路径](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/elliptic.py#L89-L110)
- `rotcohvrt` 的签名没有 mask 参数；0.2.0 的 release note 虽称多数 integration/diagnostic 方法加入 masked support，但不能据此推断该轮廓函数自动排除固体区域。[函数签名](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/elliptic.py#L7-L19)、[0.2.0 release note](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/RELEASES.md#L3-L9)
- 当筛选不到任何峰值时，代码仍执行 `max(max_vals)`，没有显式空结果 fallback；这与需要稳定处理“无候选 contour”的批处理流程不完全匹配。[源码](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/elliptic.py#L64-L75)

## 连通分量与 ridge linking

`ftle_ridges` 先由 `_ftle_ridges` 在内部网格上用二阶方向导数求 ridge 点，并用 `@njit(parallel=True)`/`prange` 并行；随后调用 SciPy 的 `label`，结构元素由 `generate_binary_structure(2, 2)` 产生，最后按 `min_ridge_pts` 过滤每个标签。[ridge 点计算与标记](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/ridges.py#L9-L76)、[连通标记与过滤](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/ridges.py#L157-L229)。SciPy 将 rank=2、connectivity=2 定义为 2D 的全邻域结构（即包含对角邻居的 8-neighborhood）；实际项目若依赖该语义，仍应固定 SciPy 版本并用对角相接 fixture 验证。[SciPy API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html)

同一文件中的 `ftle_ordered_ridges` 是另一条路径：Numba stepper 沿主方向寻找邻点，再按距离和端点切向角做贪心拼接。它解决的是 ridge 的有序连接，不等价于对二值 mask 做连通分量标记，也不适合直接替换本项目的通用 component labeling。[有序 ridge 源码](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/ridges.py#L418-L717)、[入口](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/extraction/ridges.py#L720-L1054)

工具层还有 `binary_mask_dilation`，它是一次 4-neighborhood 膨胀，`corners=True` 时增加四个对角邻居；这是 mask 膨胀而非 connected-components。任意 mesh 版本同样只检查给定邻接表并返回膨胀 mask。[结构网格膨胀](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/utils.py#L1945-L2007)、[mesh 膨胀](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/utils.py#L2082-L2133)

## 计算加速与重复时间序列

- `flowmap`、`flowmap_n`、`flowmap_grid_2D` 等函数都在粒子或网格维度上使用 `@njit(parallel=True)` 和 `prange`，并在 JIT 函数中调用 numbalsoda 的 DOP853/LSODA；`flows.py` 用 `@cfunc` 创建 ODE callback，再把 Numba-compatible spline interpolant 接入 callback。[积分实现](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/integration.py#L1-L182)、[流与插值 callback](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/flows.py#L11-L48)、[callback 实现](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/flows.py#L123-L154)
- 官方文档的 flow-map composition 针对时间序列复用中间 flow map：initial 阶段计算并插值多个短时间 flow map，step 阶段左移缓存、只计算一个新的短 map，再做插值组合。文档明确称它以少量精度损失换取显著加速，并要求 `T/h` 为自然数。[用户指南](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/docs/source/userguide.rst#L291-L373)、[composition 实现](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/src/numbacs/integration.py#L609-L735)、[理论中的复用条件](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/docs/source/theory.rst#L466-L550)
- 官方 examples 特别记录了 JIT 首次调用的 warm-up 成本，并把 warm-up 与稳态时间分开；公开的对比脚本测的是 flowmap + FTLE，不是 `rotcohvrt` 或重复 `find_contours`。[示例说明](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/examples/GALLERY_HEADER.rst#L7-L20)、[对比脚本](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/examples/time_series/plot_dg_numbacs_vs_scipy.py#L168-L276)
- 0.2.0 的包元数据要求 Python `>=3.10,<3.12`，依赖包括 `numba`、`numbalsoda`、`contourpy`、`numpy`、`scipy` 和 `interpolation`。[官方包元数据](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/pyproject.toml#L6-L37) 当前项目服务器环境是 Python 3.12.14，因此不能按该 release 的元数据直接安装到现有环境；这不是算法不可用的证明，但需要先决定隔离 Python 3.11 环境或等待上游放宽约束。
- 在 0.2.0 的 `src/numbacs` 和包元数据中检索到的是 CPU Numba/JIT、`prange` 和编译 ODE 路径，没有发现 `numba.cuda`/CuPy 实现；因此应把 NumbaCS 自身视为 CPU-oriented，不能把 README 比较表中 **Aquila-LCS** 的 “GPU and CPU versions” 误归给 NumbaCS。[NumbaCS 自身条目](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/README.md#L317-L322)、[官方依赖元数据](https://github.com/alb3rtjarvis/numbacs/blob/0.2.0/pyproject.toml#L30-L37)。这是基于已检查源码范围的负向结论，若要确认未发布分支仍需另行审计。

## 对本项目重复 contour workload 的建议

1. 把 NumbaCS 当作 **ContourPy 后端和椭圆结构筛选逻辑的参考实现/基线**，不要当作 `skimage.measure.find_contours` 的 Numba 加速层。首先在合成旋涡场上比较 contour 顶点、闭合判定、边界/固体 mask、凸性阈值和空结果行为。
2. 若保留当前每 frame、32 levels 的契约，应单独计时：局部峰值筛选、32 次等值线提取、点在多边形、凸性/周长后处理。NumbaCS 能说明“generator 在单次调用内复用”这一点，但没有提供跨 frame contour cache。
3. 不要直接移植其 `ftle_ridges` 的 label 或 `ftle_ordered_ridges`：前者绑定 FTLE ridge 布尔场和 SciPy 的 8-neighborhood，后者是带方向/切向约束的 ridge linking。项目自己的 connectivity、去重、最外层 contour 和 unknown-band 语义必须通过现有测试保持。

## 未解决问题与本次验证边界

- 没有找到官方 benchmark 专门测量 `rotcohvrt` 在多 frame、32-level 场景下相对 `find_contours` 的耗时或输出一致性；公开 time-series benchmark 的对象是 flowmap/FTLE。[官方示例](https://github.com/alb3rtjarvis/numbacs/tree/0.2.0/examples/time_series)
- 尚未运行 NumbaCS 的 runtime test：当前项目 Python 3.12.14 与 0.2.0 的 `<3.12` 元数据约束冲突。本文的 contour、label、加速结论来自官方 0.2.0 tag 的源码、文档和包元数据静态核对。
- 已核对官方 0.2.0 的 `README.md`、`pyproject.toml`、`RELEASES.md`、`elliptic.py`、`ridges.py`、`utils.py`、`integration.py`、`flows.py` 及 time-series 文档/示例，并检查 CUDA 相关符号。没有修改生产代码、测试、产物或 `HANDOFF.md`；本文件是本次唯一新增的仓库笔记。

