from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm


L0 = math.pi * math.pi / 3.0


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def ptag(x: float) -> str:
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "ok"]


def row_is_good(row: dict, max_abs_dne: float, max_diff: float, max_bad_thermal: int) -> bool:
    try:
        dne = abs(float(row.get("dne", "nan")))
        diff = float(row.get("diff", "nan"))
        bad = int(float(row.get("n_causality_bad_thermal") or 0))
    except ValueError:
        return False
    return (
        row.get("converged") in ("True", "true", "1")
        and dne <= max_abs_dne
        and diff <= max_diff
        and bad <= max_bad_thermal
    )


def make_grid(rows: list[dict], w1: float, kind: str, value: str, mask_bad: bool, args):
    sub = [
        r
        for r in rows
        if abs(float(r["w1_nf"]) - w1) < 1e-12 and r["kind"] == kind
        and abs(float(r["T"]) - args.temperature) < 1e-12
    ]
    if not sub:
        raise ValueError(f"no data for T={args.temperature:g}, w1={w1:g}, kind={kind}")

    Us = sorted({float(r["U"]) for r in sub})
    Ds = sorted({float(r["Delta"]) for r in sub})
    Z = np.full((len(Ds), len(Us)), np.nan)
    good = np.zeros((len(Ds), len(Us)), dtype=bool)
    ui = {x: i for i, x in enumerate(Us)}
    di = {x: i for i, x in enumerate(Ds)}

    for r in sub:
        i = di[float(r["Delta"])]
        j = ui[float(r["U"])]
        try:
            z = float(r[value])
        except Exception:
            z = np.nan
        is_good = row_is_good(r, args.max_abs_dne, args.max_diff, args.max_bad_thermal)
        if mask_bad and not is_good:
            z = np.nan
        Z[i, j] = z
        good[i, j] = is_good
    return np.array(Us), np.array(Ds), Z, good


def prb_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 22,
            "axes.linewidth": 2.4,
            "axes.labelsize": 30,
            "axes.titlesize": 34,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 10,
            "ytick.major.size": 10,
            "xtick.minor.size": 5,
            "ytick.minor.size": 5,
            "xtick.major.width": 2.2,
            "ytick.major.width": 2.2,
            "xtick.minor.width": 1.6,
            "ytick.minor.width": 1.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def value_label(value: str, kind: str) -> str:
    suffix = r"\mathrm{%s}" % kind
    if value == "sigma":
        return r"\sigma_{%s}" % suffix
    if value == "kappa":
        return r"\kappa_{%s}" % suffix
    if value == "lorenz":
        return r"L_{%s}" % suffix
    if value == "kappa_bare":
        return r"\kappa^{\mathrm{bare}}_{%s}" % suffix
    if value == "thermoelectric_correction":
        return r"\kappa^{\mathrm{corr}}_{%s}" % suffix
    return value


def positive_map_norm(Z: np.ndarray, args):
    finite = Z[np.isfinite(Z)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        raise ValueError("no positive finite values for logarithmic map")
    vmin = args.vmin if args.vmin is not None else args.log_floor
    vmax = args.vmax if args.vmax is not None else float(np.nanpercentile(finite, args.vmax_percentile))
    vmin = max(vmin, args.log_floor)
    vmax = max(vmax, 10.0 * vmin)
    return LogNorm(vmin=vmin, vmax=vmax), vmin, vmax


def lorenz_map_data(Z: np.ndarray, args):
    if args.lorenz_mode == "raw":
        finite = Z[np.isfinite(Z)]
        vmin = args.vmin if args.vmin is not None else float(np.nanpercentile(finite, 1.0))
        vmax = args.vmax if args.vmax is not None else float(np.nanpercentile(finite, args.vmax_percentile))
        return Z, None, vmin, vmax, r"$L=\kappa/(T\sigma)$"
    Zrel = Z / L0 - 1.0
    finite = Zrel[np.isfinite(Zrel)]
    span = args.lorenz_span
    if span is None:
        span = float(np.nanpercentile(np.abs(finite), args.vmax_percentile))
    span = max(span, 1e-8)
    return Zrel, TwoSlopeNorm(vcenter=0.0, vmin=-span, vmax=span), -span, span, r"$L/L_0-1$"


def plot_one(rows: list[dict], value: str, kind: str, w1: float, args, figdir: Path):
    U, D, Zraw, good = make_grid(rows, w1, kind, value, args.mask_bad, args)
    label = value_label(value, kind)

    if value == "lorenz":
        Z, norm, vmin, vmax, cbar_label = lorenz_map_data(Zraw, args)
        cmap = plt.get_cmap(args.lorenz_cmap).copy()
        plot_kwargs = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    else:
        Z = np.where(Zraw > args.log_floor, Zraw, args.log_floor)
        norm, _, _ = positive_map_norm(Z, args)
        cbar_label = rf"${label}$"
        cmap = plt.get_cmap(args.cmap).copy()
        cmap.set_under("black")
        plot_kwargs = {"norm": norm}

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[U.min(), U.max(), D.min(), D.max()],
        aspect="auto",
        cmap=cmap,
        interpolation=args.interpolation,
        **plot_kwargs,
    )

    if args.contour and value != "lorenz":
        try:
            ax.contour(
                U,
                D,
                Zraw,
                levels=[args.contour_level],
                colors=args.contour_color,
                linewidths=args.contour_width,
            )
        except ValueError:
            pass

    if args.bad_markers and np.any(~good):
        yy, xx = np.where(~good)
        ax.scatter(U[xx], D[yy], marker="x", s=42, c="white", linewidths=1.4)

    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$W$")
    ax.set_title(rf"${label},\; T={args.temperature:g},\; w_1={w1:g}$")
    ax.minorticks_on()
    ax.set_xlim(U.min(), U.max())
    ax.set_ylim(D.min(), D.max())

    cb = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.055)
    cb.set_label(cbar_label, rotation=270, labelpad=30)
    cb.ax.tick_params(direction="in", width=2.0, length=8)
    cb.outline.set_linewidth(2.0)

    tag = f"{value}_{kind}_w1_{ptag(w1)}_T_{ptag(args.temperature)}_prb"
    for fmt in parse_list(args.formats):
        path = figdir / f"map_{tag}.{fmt}"
        fig.savefig(path, dpi=args.dpi)
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pbs_variantB_T002")
    ap.add_argument("--csv")
    ap.add_argument("--w1-list", required=True)
    ap.add_argument("--values", default="sigma,kappa,lorenz")
    ap.add_argument("--kinds", default="typ")
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--log-floor", type=float, default=1e-4)
    ap.add_argument("--vmin", type=float)
    ap.add_argument("--vmax", type=float)
    ap.add_argument("--vmax-percentile", type=float, default=99.0)
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--lorenz-cmap", default="RdBu_r")
    ap.add_argument("--lorenz-mode", choices=["relative", "raw"], default="relative")
    ap.add_argument("--lorenz-span", type=float)
    ap.add_argument("--contour", action="store_true")
    ap.add_argument("--contour-level", type=float, default=1e-4)
    ap.add_argument("--contour-color", default="white")
    ap.add_argument("--contour-width", type=float, default=3.0)
    ap.add_argument("--interpolation", default="bicubic")
    ap.add_argument("--mask-bad", action="store_true")
    ap.add_argument("--bad-markers", action="store_true")
    ap.add_argument("--max-abs-dne", type=float, default=1e-2)
    ap.add_argument("--max-diff", type=float, default=2.5e-3)
    ap.add_argument("--max-bad-thermal", type=int, default=10)
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    prb_style()
    outdir = Path(args.out)
    csv_path = Path(args.csv) if args.csv else outdir / "transport_summary.csv"
    rows = load_rows(csv_path)
    figdir = outdir / "figures_prb_maps"
    figdir.mkdir(parents=True, exist_ok=True)

    for kind in parse_list(args.kinds):
        for value in parse_list(args.values):
            for w1 in [float(x) for x in parse_list(args.w1_list)]:
                plot_one(rows, value, kind, w1, args, figdir)


if __name__ == "__main__":
    main()
