from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common_variantB import (
    bethe_mu_guess_for_filling,
    fmt,
    linspace,
    parse_list,
    save_solution,
    solution_dir,
    solve_target_ne,
)


FIELDNAMES = [
    "T",
    "U",
    "Delta",
    "kind",
    "w1_nf",
    "ne_target",
    "mu",
    "ne_actual",
    "dne",
    "A_arith_0",
    "A_typ_0",
    "R0_typ_over_arith",
    "iterations",
    "converged",
    "diff",
    "runtime_dmft_sec",
    "solution_dir",
    "status",
    "error",
]


def load_done(csv_path: Path) -> set[tuple[str, str, str, str, str]]:
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                done.add((r["T"], r["U"], r["Delta"], r["kind"], r["w1_nf"]))
    return done


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if need_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def ne_for_w1(w1: float, mode: str) -> float:
    if mode == "complement":
        return 1.0 - w1
    raise ValueError(f"unknown ne mode: {mode}")


def path_task(payload: dict) -> list[dict]:
    T = payload["T"]
    U = payload["U"]
    Delta = payload["Delta"]
    kind = payload["kind"]
    w1_list = payload["w1_list"]
    args = payload["args"]
    outdir = Path(payload["outdir"])
    done = payload["done"]

    rows: list[dict] = []
    initial_hybrid = None
    slope = None
    previous_mu = bethe_mu_guess_for_filling(ne_for_w1(w1_list[0], args["ne_mode"]), args["band_halfwidth"])
    previous_ne = None
    previous_w1 = None

    for w1 in w1_list:
        key = (fmt(T), fmt(U), fmt(Delta), kind, fmt(w1))
        if key in done:
            continue
        ne_target = ne_for_w1(w1, args["ne_mode"])
        if previous_ne is not None and slope is not None and abs(slope) > 1e-10:
            mu_guess = previous_mu + (ne_target - previous_ne) / slope
        elif previous_w1 is not None:
            mu_guess = previous_mu + 2.0 * (ne_target - previous_ne)
        else:
            mu_guess = previous_mu

        try:
            res, slope = solve_target_ne(
                ne_target=ne_target,
                T=T,
                U=U,
                disorder_W=Delta,
                w1=w1,
                kind=kind,
                mu_guess=mu_guess,
                initial_hybrid=initial_hybrid,
                mu_min=args["mu_min"],
                mu_max=args["mu_max"],
                ne_tol=args["ne_tol"],
                max_mu_iter=args["max_mu_iter"],
                accept_diff=args["mu_accept_diff"],
                previous_slope=slope,
                omega0=args["omega0"],
                n_omega=args["n_omega"],
                n_eps=args["n_eps"],
                eta=args["eta"],
                band_halfwidth=args["band_halfwidth"],
                max_iter=args["max_iter"],
                tol=args["tol"],
                mix=args["mix"],
            )
            i0 = int(np.argmin(np.abs(res.omega)))
            A_arith_0 = float(res.rho_arith[i0])
            A_typ_0 = float(res.rho_typ[i0])
            sol_dir = solution_dir(outdir / "dmft_solutions", T, U, Delta, w1, kind)
            row = {
                "T": fmt(T),
                "U": fmt(U),
                "Delta": fmt(Delta),
                "kind": kind,
                "w1_nf": fmt(w1),
                "ne_target": fmt(ne_target),
                "mu": fmt(res.mu),
                "ne_actual": fmt(res.ne_actual),
                "dne": fmt(res.ne_actual - ne_target),
                "A_arith_0": fmt(A_arith_0),
                "A_typ_0": fmt(A_typ_0),
                "R0_typ_over_arith": fmt(A_typ_0 / A_arith_0 if abs(A_arith_0) > 1e-300 else np.nan),
                "iterations": res.iterations,
                "converged": res.converged,
                "diff": fmt(res.diff),
                "runtime_dmft_sec": fmt(res.runtime_sec),
                "solution_dir": str(sol_dir),
                "status": "ok",
                "error": "",
            }
            save_solution(sol_dir, res, row)
            rows.append(row)
            initial_hybrid = res.hybrid
            previous_mu = res.mu
            previous_ne = res.ne_actual
            previous_w1 = w1
        except Exception as exc:
            rows.append(
                {
                    "T": fmt(T),
                    "U": fmt(U),
                    "Delta": fmt(Delta),
                    "kind": kind,
                    "w1_nf": fmt(w1),
                    "ne_target": fmt(ne_target),
                    "status": "error",
                    "error": repr(exc),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T-list", required=True)
    ap.add_argument("--w1-list", required=True, help="localized filling n_f values; variant B uses n_e=1-w1")
    ap.add_argument("--ne-mode", choices=["complement"], default="complement")
    ap.add_argument("--U-min", type=float, default=0.0)
    ap.add_argument("--U-max", type=float, default=2.0)
    ap.add_argument("--n-U", type=int, default=21)
    ap.add_argument("--Delta-min", type=float, default=0.05)
    ap.add_argument("--Delta-max", type=float, default=2.0)
    ap.add_argument("--n-Delta", type=int, default=21)
    ap.add_argument("--U-list")
    ap.add_argument("--Delta-list")
    ap.add_argument("--kind", choices=["typ", "arith", "both"], default="typ")
    ap.add_argument("--omega0", type=float, default=12.0)
    ap.add_argument("--n-omega", type=int, default=501)
    ap.add_argument("--n-eps", type=int, default=51)
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--band-halfwidth", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=800)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--mix", type=float, default=0.12)
    ap.add_argument("--mu-min", type=float, default=-8.0)
    ap.add_argument("--mu-max", type=float, default=8.0)
    ap.add_argument("--ne-tol", type=float, default=5e-3)
    ap.add_argument("--max-mu-iter", type=int, default=8)
    ap.add_argument("--mu-accept-diff", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="pbs_variantB")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "dmft_summary.csv"
    done = load_done(csv_path) if args.resume else set()

    T_grid = parse_list(args.T_list)
    w1_list = parse_list(args.w1_list)
    U_grid = parse_list(args.U_list) if args.U_list else list(linspace(args.U_min, args.U_max, args.n_U))
    Delta_grid = parse_list(args.Delta_list) if args.Delta_list else list(linspace(args.Delta_min, args.Delta_max, args.n_Delta))
    kinds = ["typ", "arith"] if args.kind == "both" else [args.kind]
    common_args = vars(args).copy()

    tasks = [
        {"T": T, "U": U, "Delta": Delta, "kind": kind, "w1_list": w1_list, "args": common_args, "outdir": str(outdir), "done": done}
        for T in T_grid
        for Delta in Delta_grid
        for U in U_grid
        for kind in kinds
    ]
    print(f"Variant-B DMFT paths: {len(tasks)}, workers={args.workers}", flush=True)

    if args.workers <= 1:
        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] T={task['T']:g} U={task['U']:g} Delta={task['Delta']:g} {task['kind']}", flush=True)
            append_rows(csv_path, path_task(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(path_task, task): task for task in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                task = futures[fut]
                print(f"[{i}/{len(tasks)}] done T={task['T']:g} U={task['U']:g} Delta={task['Delta']:g} {task['kind']}", flush=True)
                append_rows(csv_path, fut.result())
    print(f"DMFT summary saved: {csv_path}")


if __name__ == "__main__":
    main()
