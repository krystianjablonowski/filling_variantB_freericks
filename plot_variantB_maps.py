from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def ptag(x: float) -> str:
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "ok"]


def make_grid(rows, w1, kind, value):
    sub = [r for r in rows if abs(float(r["w1_nf"]) - w1) < 1e-12 and r["kind"] == kind]
    if not sub:
        raise ValueError(f"no data for w1={w1:g}, kind={kind}")
    Us = sorted({float(r["U"]) for r in sub})
    Ds = sorted({float(r["Delta"]) for r in sub})
    Z = np.full((len(Ds), len(Us)), np.nan)
    ui = {x: i for i, x in enumerate(Us)}
    di = {x: i for i, x in enumerate(Ds)}
    for r in sub:
        try:
            z = float(r[value])
        except Exception:
            z = np.nan
        Z[di[float(r["Delta"])]][ui[float(r["U"])]] = z
    return np.array(Us), np.array(Ds), Z


def norm_for(Z, args):
    finite = Z[np.isfinite(Z)]
    if args.vmin is not None:
        vmin = args.vmin
    else:
        vmin = float(np.nanmin(finite))
    if args.vmax is not None:
        vmax = args.vmax
    elif args.vmax_percentile is not None:
        vmax = float(np.nanpercentile(finite, args.vmax_percentile))
    else:
        vmax = float(np.nanmax(finite))
    if args.norm == "log":
        vmin = max(vmin, args.log_floor)
        vmax = max(vmax, vmin * 10.0)
        return LogNorm(vmin=vmin, vmax=vmax), vmin, vmax
    if args.norm == "power":
        return PowerNorm(gamma=args.gamma, vmin=vmin, vmax=vmax), vmin, vmax
    return None, vmin, vmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pbs_variantB_T002")
    ap.add_argument("--csv")
    ap.add_argument("--w1-list", required=True)
    ap.add_argument("--values", default="sigma,kappa,lorenz,L12,kappa_bare,thermoelectric_correction")
    ap.add_argument("--kinds", default="typ")
    ap.add_argument("--norm", choices=["linear", "log", "power"], default="power")
    ap.add_argument("--gamma", type=float, default=0.35)
    ap.add_argument("--log-floor", type=float, default=1e-8)
    ap.add_argument("--vmin", type=float)
    ap.add_argument("--vmax", type=float)
    ap.add_argument("--vmax-percentile", type=float, default=99.0)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    outdir = Path(args.out)
    csv_path = Path(args.csv) if args.csv else outdir / "transport_summary.csv"
    rows = load_rows(csv_path)
    figdir = outdir / "figures_maps"
    figdir.mkdir(parents=True, exist_ok=True)

    for kind in parse_list(args.kinds):
        for value in parse_list(args.values):
            for w1 in [float(x) for x in parse_list(args.w1_list)]:
                U, D, Zraw = make_grid(rows, w1, kind, value)
                Z = Zraw.copy()
                if args.norm == "log":
                    Z = np.where(Z > args.log_floor, Z, args.log_floor)
                norm, vmin, vmax = norm_for(Z, args)
                fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
                im = ax.imshow(
                    Z,
                    origin="lower",
                    aspect="auto",
                    extent=[U.min(), U.max(), D.min(), D.max()],
                    cmap=args.cmap,
                    norm=norm,
                    vmin=None if norm else vmin,
                    vmax=None if norm else vmax,
                    interpolation="bicubic",
                )
                cb = fig.colorbar(im, ax=ax)
                cb.set_label(value)
                ax.set_xlabel(r"$U$")
                ax.set_ylabel(r"$\Delta$")
                ax.set_title(fr"{value}, {kind}, $w_1=n_f={w1:g}$, $n_e={1-w1:g}$")
                path = figdir / f"heatmap_{value}_{kind}_w1_{ptag(w1)}.png"
                fig.savefig(path, dpi=args.dpi)
                plt.close(fig)
                print(f"Saved: {path}")


if __name__ == "__main__":
    main()
