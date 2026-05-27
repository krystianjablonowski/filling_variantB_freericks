from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common_variantB import (
    fmt,
    linspace,
    parse_list,
    solve_dmft_for_mu,
    solve_target_ne,
    transport_from_arrays,
)


FIELDNAMES = [
    "T",
    "U",
    "Delta",
    "kind",
    "hilbert_mode",
    "run",
    "mu",
    "mu_minus_U_over_2",
    "ne_actual",
    "dne",
    "A_arith_0",
    "A_typ_0",
    "sigma",
    "kappa",
    "lorenz",
    "L11",
    "L12",
    "L22",
    "iterations",
    "converged",
    "diff",
    "max_ImSigma_raw",
    "max_ImSigma_thermal",
    "n_causality_bad",
    "n_causality_bad_thermal",
    "status",
    "error",
]


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if need_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def summarize_solution(*, res, T: float, U: float, Delta: float, kind: str, hilbert_mode: str, run: str, args) -> dict:
    i0 = int(np.argmin(np.abs(res.omega)))
    tr = transport_from_arrays(
        res.omega,
        res.sigma,
        res.mu,
        T,
        args.n_band,
        args.band_halfwidth,
        args.eta,
    )
    row = {
        "T": fmt(T),
        "U": fmt(U),
        "Delta": fmt(Delta),
        "kind": kind,
        "hilbert_mode": hilbert_mode,
        "run": run,
        "mu": fmt(res.mu),
        "mu_minus_U_over_2": fmt(res.mu - 0.5 * U),
        "ne_actual": fmt(res.ne_actual),
        "dne": fmt(res.ne_actual - 0.5),
        "A_arith_0": fmt(float(res.rho_arith[i0])),
        "A_typ_0": fmt(float(res.rho_typ[i0])),
        "iterations": res.iterations,
        "converged": res.converged,
        "diff": fmt(res.diff),
        "status": "ok",
        "error": "",
    }
    row.update({k: fmt(v) for k, v in tr.items()})
    return row


def diagnostic_point(T: float, U: float, Delta: float, kind: str, hilbert_mode: str, args) -> list[dict]:
    rows: list[dict] = []
    kwargs = dict(
        T=T,
        U=U,
        disorder_W=Delta,
        w1=0.5,
        kind=kind,
        omega0=args.omega0,
        n_omega=args.n_omega,
        n_eps=args.n_eps,
        eta=args.eta,
        band_halfwidth=args.band_halfwidth,
        max_iter=args.max_iter,
        tol=args.tol,
        mix=args.mix,
        hilbert_mode=hilbert_mode,
    )
    try:
        target_res, _ = solve_target_ne(
            ne_target=0.5,
            mu_guess=0.5 * U,
            initial_hybrid=None,
            mu_min=args.mu_min,
            mu_max=args.mu_max,
            ne_tol=args.ne_tol,
            max_mu_iter=args.max_mu_iter,
            accept_diff=args.mu_accept_diff,
            previous_slope=None,
            **kwargs,
        )
        rows.append(
            summarize_solution(
                res=target_res,
                T=T,
                U=U,
                Delta=Delta,
                kind=kind,
                hilbert_mode=hilbert_mode,
                run="target_ne",
                args=args,
            )
        )

        pinned_res = solve_dmft_for_mu(mu=0.5 * U, initial_hybrid=None, **kwargs)
        rows.append(
            summarize_solution(
                res=pinned_res,
                T=T,
                U=U,
                Delta=Delta,
                kind=kind,
                hilbert_mode=hilbert_mode,
                run="mu_U_over_2_diagnostic",
                args=args,
            )
        )
    except Exception as exc:
        rows.append(
            {
                "T": fmt(T),
                "U": fmt(U),
                "Delta": fmt(Delta),
                "kind": kind,
                "hilbert_mode": hilbert_mode,
                "run": "diagnostic",
                "status": "error",
                "error": repr(exc),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Half-filling regression diagnostic for Variant B. This is not a production shortcut."
    )
    ap.add_argument("--T", type=float, default=0.02)
    ap.add_argument("--U-list")
    ap.add_argument("--Delta-list")
    ap.add_argument("--U-min", type=float, default=0.0)
    ap.add_argument("--U-max", type=float, default=2.0)
    ap.add_argument("--n-U", type=int, default=5)
    ap.add_argument("--Delta-min", type=float, default=0.05)
    ap.add_argument("--Delta-max", type=float, default=2.0)
    ap.add_argument("--n-Delta", type=int, default=5)
    ap.add_argument("--kind", choices=["typ", "arith"], default="typ")
    ap.add_argument("--hilbert-modes", default="eta,pv")
    ap.add_argument("--omega0", type=float, default=8.0)
    ap.add_argument("--n-omega", type=int, default=301)
    ap.add_argument("--n-eps", type=int, default=31)
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--band-halfwidth", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--mix", type=float, default=0.15)
    ap.add_argument("--mu-min", type=float, default=-8.0)
    ap.add_argument("--mu-max", type=float, default=8.0)
    ap.add_argument("--ne-tol", type=float, default=5e-3)
    ap.add_argument("--max-mu-iter", type=int, default=20)
    ap.add_argument("--mu-accept-diff", type=float, default=2e-3)
    ap.add_argument("--n-band", type=int, default=201)
    ap.add_argument("--out", default="half_filling_diagnostic")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "half_filling_diagnostic.csv"
    if csv_path.exists():
        csv_path.unlink()

    U_grid = parse_list(args.U_list) if args.U_list else list(linspace(args.U_min, args.U_max, args.n_U))
    Delta_grid = parse_list(args.Delta_list) if args.Delta_list else list(linspace(args.Delta_min, args.Delta_max, args.n_Delta))
    modes = [x.strip() for x in args.hilbert_modes.split(",") if x.strip()]

    total = len(U_grid) * len(Delta_grid) * len(modes)
    counter = 0
    for mode in modes:
        for Delta in Delta_grid:
            for U in U_grid:
                counter += 1
                print(f"[{counter}/{total}] T={args.T:g} U={U:g} Delta={Delta:g} mode={mode}", flush=True)
                append_rows(csv_path, diagnostic_point(args.T, U, Delta, args.kind, mode, args))

    print(f"Diagnostic saved: {csv_path}")
    print("Check that target_ne has mu_minus_U_over_2 close to zero and matches the old half-filling map.")


if __name__ == "__main__":
    main()
