"""Read MFC's five-equation MPI restart into the project's physical time grid.

Source: pinned MFC m_data_output, m_mpi_common and m_variables_conversion.
The raw restart has x-fast float64 [rho, rho*u, rho*v, energy, alpha] blocks;
the initial lustre_ib.dat block has x-fast native int32 body markers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch


def read_primitives(path, shape, gamma):
    expected_bytes = 5*int(np.prod(shape))*8
    if Path(path).stat().st_size != expected_bytes:
        raise ValueError(f"Expected five float64 conservative fields: {path}")
    q = np.memmap(path, mode="r", dtype="<f8", shape=(5, *shape))
    rho = np.asarray(q[0])
    u, v = q[1]/rho, q[2]/rho
    pressure = (q[3]-0.5*rho*(u*u+v*v))*(gamma-1)/q[4]
    return {"rho": np.array(rho, dtype=np.float32), "pressure": pressure.astype(np.float32),
            "u": u.astype(np.float32), "v": v.astype(np.float32)}


def qrotstrain(u, v, x, y, solid, device):
    """The existing project decomposition and stencil support on a torch device."""
    u = torch.as_tensor(u.astype(np.float64), device=device)
    v = torch.as_tensor(v.astype(np.float64), device=device)
    valid = ~torch.as_tensor(solid, device=device) & torch.isfinite(u) & torch.isfinite(v)
    supported = valid.clone()
    for axis in (0, 1):
        stencil = valid & torch.roll(valid, 1, axis) & torch.roll(valid, -1, axis)
        start = [slice(None), slice(None)]
        end = start.copy()
        start[axis], end[axis] = 0, -1
        stencil[tuple(start)] = torch.index_select(valid, axis, torch.arange(3, device=device)).all(dim=axis)
        stencil[tuple(end)] = torch.index_select(valid, axis, torch.arange(valid.shape[axis]-3, valid.shape[axis], device=device)).all(dim=axis)
        supported &= stencil
    axes = (torch.as_tensor(y, device=device), torch.as_tensor(x, device=device))
    b, a = torch.gradient(torch.where(valid, u, torch.nan), spacing=axes, edge_order=2)
    d, c = torch.gradient(torch.where(valid, v, torch.nan), spacing=axes, edge_order=2)
    rotation_squared = (c-b).square()/2
    strain_squared = a.square()+d.square()+(b+c).square()/2
    q = torch.where(supported, (rotation_squared-strain_squared)/2, torch.nan)
    return q.cpu().numpy().astype(np.float32), supported.cpu().numpy()


def convert(case_root, output_root, device="cuda"):
    started = time.perf_counter()
    root, output = Path(case_root), Path(output_root)
    case = json.loads((root / "case.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    shape = tuple(meta["grid_shape_yx"])
    x_cb = np.fromfile(root / "restart_data/lustre_x_cb.dat", dtype="<f8")
    y_cb = np.fromfile(root / "restart_data/lustre_y_cb.dat", dtype="<f8")
    x, y = (x_cb[:-1]+x_cb[1:])/2, (y_cb[:-1]+y_cb[1:])/2
    solid = np.memmap(root / "restart_data/lustre_ib.dat", mode="r", dtype="<i4", shape=shape) != 0
    steps = np.arange(meta["training_first_step"], case["t_step_stop"]+1, case["t_step_save"])
    times = steps.astype(np.float64)*case["dt"]
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "mask.npy", solid.astype(np.uint8))
    records = []
    with h5py.File(output / "flow.h5", "w") as z:
        z.attrs["source_case"] = json.dumps(case)
        z.attrs["dataset_role"] = "training"
        z.attrs["solver_commit"] = meta["solver_commit"]
        z.attrs["q_definition"] = meta["physics_q"]
        for name, values in (("xdim", x), ("ydim", y), ("tdim", times), ("solver_step", steps)):
            z.create_dataset(name, data=values)
        z.create_dataset("geometry", data=solid.astype(np.uint8))
        fields = {name: z.create_dataset(name, shape=(len(steps), *shape), dtype="f4",
                   chunks=(1, min(128, shape[0]), min(128, shape[1])), compression="lzf")
                  for name in ("rho", "pressure", "u", "v", "q_rotstrain")}
        valid_output = z.create_dataset("derivative_valid", shape=(len(steps), *shape), dtype="u1",
                                        chunks=(1, min(128, shape[0]), min(128, shape[1])), compression="lzf")
        fluid = ~solid
        for i, step in enumerate(steps):
            primitives = read_primitives(root / f"restart_data/lustre_{step}.dat", shape, meta["flow"]["gamma"])
            for name, values in primitives.items():
                if not np.isfinite(values[fluid]).all() or (name in ("rho", "pressure") and np.any(values[fluid] <= 0)):
                    raise ValueError(f"Invalid observed-fluid {name} at solver step {step}")
                fields[name][i] = values
            q, valid = qrotstrain(primitives["u"], primitives["v"], x, y, solid, device)
            fields["q_rotstrain"][i], valid_output[i] = q, valid
            records.append({"solver_step": int(step), "time": float(times[i]),
                            "derivative_valid_pixels": int(valid.sum()),
                            "rho_min_fluid": float(primitives["rho"][fluid].min()),
                            "pressure_min_fluid": float(primitives["pressure"][fluid].min())})
            if i % 25 == 0 or i == len(steps)-1:
                z.flush()
                print(f"converted {i+1}/{len(steps)} t={times[i]:.6f}", flush=True)
    result = {**meta, "frames_written": len(steps), "shape_tyx": [len(steps), *shape],
              "solid_pixels": int(solid.sum()), "spatial_resampling": False,
              "temporal_interpolation": False, "conversion_device": device,
              "physical_coordinate_convention": "native MFC uniform cell centers",
              "pathline_24_frame_window_count_step4": max(0, (len(steps)-24)//4+1),
              "records": records, "elapsed_seconds": time.perf_counter()-started}
    (output / "dataset_metadata.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("frames_written", "shape_tyx", "solid_pixels", "elapsed_seconds")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    convert(args.case_root, args.output_root, args.device)
