# -*- coding: utf-8 -*-
"""notebook cell5 改用 pick_bench_source：前缀保留，替换 BENCH 头。"""
import json

path = "kaggle/train_kaggle.ipynb"
nb = json.load(open(path, encoding="utf-8"))
c5 = nb["cells"][5]
src = "".join(c5["source"])

anchor = 'BENCH_INFO = "outputs/bench_info.json"'
i_anchor = src.index(anchor)
prefix_lines = c5["source"][:src[:i_anchor].count("\n")]
assert "".join(prefix_lines) == src[:i_anchor], "前缀切分异常"

# 原 BENCH 头块（到 else: 为止）
old_block = [
    'BENCH_INFO = "outputs/bench_info.json"\n',
    'BENCH_RESTORED = "outputs/train/bench_info.json"  # 上个会话块尾打包 → 本会话 cell 3 还原（跨会话免重复校准）\n',
    'bench_src = BENCH_INFO if os.path.exists(BENCH_INFO) else (BENCH_RESTORED if os.path.exists(BENCH_RESTORED) else None)\n',
    'if bench_src is not None:\n',
    '    bench = json.load(open(bench_src, encoding="utf-8"))\n',
    '    print(f"复用步速基准（{bench_src}）: 1 epoch = {bench[\'seconds_per_epoch\']:.1f} s（{bench[\'timestamp\']} 实测，省 ~18min 校准）")\n',
    'else:\n',
]
rest = "".join(c5["source"])[src.index(anchor):]
assert rest.startswith("".join(old_block)), "旧块不匹配"
tail = rest[len("".join(old_block)):]

new_block = [
    'BENCH_INFO = "outputs/bench_info.json"\n',
    'BENCH_RESTORED = "outputs/train/bench_info.json"   # 上个会话块尾打包 → cell 3 还原（跨会话免重复校准）\n',
    'from kaggle.chunking import pick_bench_source\n',
    'bench_src, bench = pick_bench_source(BENCH_INFO, BENCH_RESTORED)\n',
    'if bench_src is not None:\n',
    '    print(f"复用步速基准（{bench_src}）: 1 epoch = {bench[\'seconds_per_epoch\']:.1f} s（{bench[\'timestamp\']} 实测，省 ~18min 校准）")\n',
    'else:\n',
]
c5["source"] = prefix_lines + new_block + [repr_helper(l) for l in tail.split("\n") if l][:1] if False else prefix_lines + new_block + tail_block(tail)
json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
open(path, "a", encoding="utf-8").write("\n")
print("cell5 head swapped")
