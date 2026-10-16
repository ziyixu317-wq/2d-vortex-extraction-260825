"""Materialize the author's MFC airfoil with dense, equally spaced saves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

AUTHOR_COMMIT = "c07346dc21479926fe0b12dcd04ae53e6c50016c"
MFC_COMMIT = "0c9a1d434410175ac483b8d71646455444e3b7eb"
AUTHOR_CONFIG = Path(__file__).parent / "config/mfc_airfoil_training"


def prepare(output_root, grid="f180", *, final_time=3.5, save_dt=0.005, training_start=1.0,
            dt_factor=1, mp_weno=False):
    root = Path(output_root).resolve()
    command = [sys.executable, str(AUTHOR_CONFIG / "author_case.py"),
               "--grid", grid, "--final-time", str(final_time), "--save-dt", str(save_dt), "--format", "binary"]
    case = json.loads(subprocess.check_output(command, text=True))
    case["dt"] /= dt_factor
    for name in ("t_step_start", "t_step_stop", "t_step_save"):
        case[name] *= dt_factor
    case["mp_weno"] = "T" if mp_weno else "F"
    first_step = round(training_start / case["dt"])
    if not 0 <= first_step <= case["t_step_stop"] or first_step % case["t_step_save"]:
        raise ValueError("Training start must be a retained solver time")
    root.mkdir(parents=True, exist_ok=True)
    (root / "case.json").write_text(json.dumps(case, indent=2)+"\n", encoding="utf-8")
    (root / "case.py").write_text('import json\nfrom pathlib import Path\nprint((Path(__file__).with_name("case.json")).read_text())\n', encoding="utf-8")
    shutil.copy2(AUTHOR_CONFIG / "Diamond_Airfoil_2D_MFC.stl", root)
    metadata = {
        "source_repository": "https://github.com/Ehsan-Roohi/ShockVortexML",
        "source_commit": AUTHOR_COMMIT,
        "source_case": "examples/cfd_showcase_2026/configs/airfoil_m3_a40_re1e4_f270_case.py",
        "solver_repository": "https://github.com/MFlowCode/MFC", "solver_commit": MFC_COMMIT,
        "flow": {"mach": 3, "flow_angle_deg": 40, "reynolds_chord": 10000, "gamma": 1.4,
                 "boundary": "viscous no-slip diamond airfoil", "grid": grid},
        "grid_shape_yx": [case["n"]+1, case["m"]+1],
        "solver_dt": case["dt"], "save_every_solver_steps": case["t_step_save"],
        "saved_dt": case["dt"]*case["t_step_save"],
        "solver_stop_step": case["t_step_stop"],
        "raw_saved_frames": case["t_step_stop"]//case["t_step_save"]+1,
        "training_first_step": first_step,
        "training_frames": (case["t_step_stop"]-first_step)//case["t_step_save"]+1,
        "training_start_time": first_step*case["dt"], "training_stop_time": case["t_step_stop"]*case["dt"],
        "dataset_role": "training", "initial_transient_excluded_until": training_start,
        "physics_q": "Qrotstrain = 0.5*(rotation_norm^2-strain_norm^2)",
        "numerical_adjustments": {"dt_factor": dt_factor, "mp_weno": mp_weno},
    }
    (root / "run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    return case, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grid", choices=("f180", "f270", "f405"), default="f180")
    parser.add_argument("--final-time", type=float, default=3.5)
    parser.add_argument("--save-dt", type=float, default=0.005)
    parser.add_argument("--training-start", type=float, default=1.0)
    parser.add_argument("--dt-factor", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--mp-weno", action="store_true")
    args = parser.parse_args()
    prepare(args.output_root, args.grid, final_time=args.final_time,
            save_dt=args.save_dt, training_start=args.training_start,
            dt_factor=args.dt_factor, mp_weno=args.mp_weno)
