"""Kaggle Dataset A 打包（07 票）：nc 数据 + prepare_dataset 产物 → 可上传目录/zip。

领域词汇（HANDOFF §4/§5 与规格，唯一权威）：
- Dataset A = nc 数据文件 + 预处理产物（票 05 prepare_dataset 的 meta.json +
  u/v/ivd/label/mask memmap ≈1.3GB；gitignore 不走 GitHub，随 Kaggle Dataset 上传）；
- Kaggle 训练从 GitHub 克隆代码（Dataset B 概念由 git clone 承担，HANDOFF §9）；
- manifest.json：逐文件 sha256 + 大小（Kaggle 端自检/审计清单；notebook 自检 cell 引用）。

用法（本地，用户终端或本脚本）：
    python kaggle/prepare_dataset_a.py --nc "CFD数据集/pipedcylinder2d.nc" \
        --dataset-dir outputs/dataset --out kaggle_dataset_a [--zip]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import zipfile


def _sha256_file(path):
    """流式 sha256（大文件不整读；manifest 审计依据）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset_a_multi(pairs, out_dir):
    """多数据集 Dataset A 组装（票 07 延伸：7 数据集联合训练）。

    pairs = [(nc 路径, prepare_dataset 产物目录), ...]（一一对应，调用方保证
    配对顺序）。布局：
      <out>/data/<nc 文件名>          各数据集原始 nc
      <out>/datasets/<目录名>/ ...    各 prepare_dataset 产物（meta.json + memmap）
    manifest.json 与单数据集同构（逐文件 sha256，Kaggle 端自检清单）。
    返回 manifest dict。
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []           # (相对路径, 绝对路径)
    used_names = set()
    for nc_path, ds_src in pairs:
        nc_path = pathlib.Path(nc_path)
        ds_src = pathlib.Path(ds_src)
        if not nc_path.exists():
            raise FileNotFoundError(f"nc 数据文件不存在: {nc_path}")
        if not ds_src.exists():
            raise FileNotFoundError(f"prepare_dataset 产物目录不存在: {ds_src}")
        files.append((pathlib.Path("data") / nc_path.name, nc_path))
        name = ds_src.name
        if name in used_names:
            raise ValueError(f"数据集目录名重复: {name}（manifest 路径歧义）")
        used_names.add(name)
        for p in sorted(ds_src.rglob("*")):
            if p.is_file():
                files.append((pathlib.Path("datasets") / name
                              / p.relative_to(ds_src), p))

    manifest_files = []
    total = 0
    for rel, src in sorted(set(files), key=lambda x: str(x[0])):
        rel = pathlib.Path(rel)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        size = dst.stat().st_size
        total += size
        manifest_files.append({"path": rel.as_posix(), "size": size,
                               "sha256": _sha256_file(str(dst))})
    manifest = {"files": manifest_files, "total_bytes": total,
                "multi": True, "datasets": [str(p) for p in pairs]}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def build_dataset_a(nc_path, dataset_dir, out_dir, aux_dirs=None):
    """组装 Dataset A 目录 → manifest dict（逐文件 shasum，可作 Kaggle 端自检清单）。

    布局：<out>/pipedcylinder2d.nc + <out>/dataset/（prepare_dataset 产物整目录）
    + <out>/aux/（aux_dirs 提供的 .png 目检图，体积小、供后台诊断）。
    返回 manifest = {"nc": ..., "files": [{path, size, sha256}], "total_bytes": ...}。
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []           # (相对路径, 绝对路径)

    # ---- nc 原始数据（Kaggle 端万一重算的原始来源；保留文件名）
    nc_path = pathlib.Path(nc_path)
    nc_dst = out_dir / nc_path.name
    if not nc_path.exists():
        raise FileNotFoundError(f"nc 数据文件不存在: {nc_path}")
    files.append((nc_path.name, nc_path))

    # ---- prepare_dataset 产物（meta.json + memmap；相对引用 → 整目录复制）
    ds_src = pathlib.Path(dataset_dir)
    if not ds_src.exists():
        raise FileNotFoundError(f"prepare_dataset 产物目录不存在: {ds_src}")
    for p in sorted(ds_src.rglob("*")):
        if p.is_file():
            rel = pathlib.Path("dataset") / p.relative_to(ds_src)
            files.append((rel, p))

    # ---- aux 目检图（仅 .png；大数组已在 dataset/ 一份，不冗余）
    for aux in (aux_dirs or []):
        aux_dir = pathlib.Path(aux)
        if not aux_dir.exists():
            raise FileNotFoundError(f"aux 目录不存在: {aux_dir}")
        for p in sorted(aux_dir.rglob("*.png")):
            rel = pathlib.Path("aux") / p.relative_to(aux_dir)
            files.append((rel, p))

    # ---- 拷贝 + manifest
    manifest_files = []
    total = 0
    for rel, src in sorted(set(files), key=lambda x: str(x[0])):
        rel = pathlib.Path(rel)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        size = dst.stat().st_size
        total += size
        manifest_files.append({"path": rel.as_posix(), "size": size,
                               "sha256": _sha256_file(str(dst))})
    manifest = {"files": manifest_files, "total_bytes": total, "nc": nc_path.name}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def make_zip(out_dir, zip_path):
    """目录 → zip（Kaggle 网页/API 上传用；成员相对路径与目录布局一致）。"""
    out_dir = pathlib.Path(out_dir)
    zip_path = pathlib.Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                zf.write(str(p), p.relative_to(out_dir).as_posix())
    return zip_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Kaggle Dataset A 打包：nc + prepare_dataset 产物 → 上传目录/zip"
                    "（多数据集：--nc/--dataset-dir 可各传多个，一一对应）")
    ap.add_argument("--nc", nargs="+", required=True,
                    help="nc 数据文件路径（h5py 直读，支持中文路径；多个 = 多数据集）")
    ap.add_argument("--dataset-dir", nargs="+", required=True,
                    help="prepare_dataset 产物目录（meta.json + memmap；与 --nc 一一对应）")
    ap.add_argument("--out", default="kaggle_dataset_a", help="输出目录")
    ap.add_argument("--aux-dirs", nargs="*", default=None,
                    help="可选 aux 目录（仅单数据集：weak_labels 目检图，只复制 .png）")
    ap.add_argument("--zip", action="store_true", help="额外打包 zip")
    args = ap.parse_args(argv)

    if len(args.nc) != len(args.dataset_dir):
        raise ValueError(
            f"--nc ({len(args.nc)}) 与 --dataset-dir ({len(args.dataset_dir)}) "
            f"个数不匹配，须一一对应")
    if len(args.nc) == 1:
        manifest = build_dataset_a(args.nc[0], args.dataset_dir[0], args.out,
                                   aux_dirs=args.aux_dirs)
    else:
        if args.aux_dirs:
            raise ValueError("多数据集打包不支持 --aux-dirs（aux 图放各数据集目录内）")
        manifest = build_dataset_a_multi(list(zip(args.nc, args.dataset_dir)),
                                         args.out)
    print(f"Dataset A 已组装: {args.out}")
    print(f"  文件数 = {len(manifest['files'])}  总大小 = "
          f"{manifest['total_bytes'] / 1e6:.1f} MB")
    if args.zip:
        zip_path = make_zip(args.out, f"{args.out}.zip")
        print(f"  zip = {zip_path}")
    return 0


if __name__ == "__main__":
    main()
