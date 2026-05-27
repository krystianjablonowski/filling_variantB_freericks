from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common_variantB import fmt, load_solution, transport_from_arrays


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
    "sigma",
    "kappa",
    "kappa_bare",
    "thermoelectric_correction",
    "lorenz",
    "L11",
    "L12",
    "L22",
    "tau_min",
    "tau_max",
    "max_ImSigma_raw",
    "max_ImSigma_thermal",
    "n_causality_bad",
    "n_causality_bad_thermal",
    "hilbert_mode",
    "solution_dir",
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


def load_dmft_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "ok"]


def transport_task(payload: dict) -> dict:
    row = payload["row"]
    args = payload["args"]
    out = {k: row.get(k, "") for k in FIELDNAMES}
    try:
        meta, data = load_solution(Path(row["solution_dir"]))
        tr = transport_from_arrays(
            omega=data["omega"],
            sigma=data["sigma"],
            mu=float(row["mu"]),
            T=float(row["T"]),
            n_band=args["n_band"],
            band_halfwidth=args["band_halfwidth"],
            eta=args["eta"],
        )
        out.update({k: fmt(v) for k, v in tr.items()})
        out["status"] = "ok"
        out["error"] = ""
    except Exception as exc:
        out["status"] = "error"
        out["error"] = repr(exc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pbs_variantB")
    ap.add_argument("--dmft-summary")
    ap.add_argument("--n-band", type=int, default=301)
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--band-halfwidth", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output-name", default="transport_summary.csv")
    args = ap.parse_args()

    outdir = Path(args.out)
    dmft_summary = Path(args.dmft_summary) if args.dmft_summary else outdir / "dmft_summary.csv"
    rows = load_dmft_rows(dmft_summary)
    output = outdir / args.output_name
    if output.exists():
        output.unlink()
    payloads = [{"row": row, "args": vars(args)} for row in rows]
    print(f"Transport points: {len(payloads)}, workers={args.workers}", flush=True)

    if args.workers <= 1:
        for i, payload in enumerate(payloads, 1):
            row = payload["row"]
            print(f"[{i}/{len(payloads)}] T={row['T']} w1={row['w1_nf']} U={row['U']} Delta={row['Delta']} {row['kind']}", flush=True)
            append_rows(output, [transport_task(payload)])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(transport_task, payload): payload for payload in payloads}
            for i, fut in enumerate(as_completed(futures), 1):
                append_rows(output, [fut.result()])
                if i % 50 == 0 or i == len(payloads):
                    print(f"[{i}/{len(payloads)}] transport done", flush=True)
    print(f"Transport summary saved: {output}")


if __name__ == "__main__":
    main()
