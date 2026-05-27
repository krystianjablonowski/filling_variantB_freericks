# Variant B: Freericks/Joura filling path with Byczuk DMFT-TMT

This folder starts a clean workflow for the disorder Falicov-Kimball model away from half filling.

Variant B means:

```text
w1 = n_f
n_e = 1 - w1
```

For every independent path `(T, U, Delta, kind)` the code follows a sequence in `w1`, using warm starts.

## Files

- `common_variantB.py` - DMFT/TMT loop, filling search, solution IO, transport integrals.
- `solve_variantB_dmft.py` - stage A: solve DMFT/TMT and save `solution.npz`.
- `compute_variantB_transport.py` - stage B: compute `L11`, `L12`, `L22`, `sigma`, `kappa`, Lorenz.
- `plot_variantB_maps.py` - simple maps for `sigma`, `kappa`, Lorenz and diagnostics.
- `submit_pbs_variantB.pl` - PBS submission script.

## Copy to cluster

From Windows PowerShell:

```powershell
scp -r "C:\Users\avoga\Documents\Codex\2026-05-24\otw-rz-wsl-localhost-ubuntu-home\filling_variantB_freericks" kj405942@kruk-host.fuw.edu.pl:~/Byczuk_Freericks/
```

## Cluster smoke test

```bash
cd ~/Byczuk_Freericks/filling_variantB_freericks

python3 -m py_compile common_variantB.py solve_variantB_dmft.py compute_variantB_transport.py plot_variantB_maps.py
perl -c submit_pbs_variantB.pl

python3 solve_variantB_dmft.py \
  --T-list 0.02 \
  --w1-list 0.5,0.45,0.4 \
  --U-list 0.0,1.0,2.0 \
  --Delta-list 0.05,1.0,2.0 \
  --kind typ \
  --omega0 12 \
  --n-omega 501 --n-eps 51 \
  --tol 1e-4 --ne-tol 5e-3 \
  --max-iter 800 --max-mu-iter 8 \
  --mix 0.12 \
  --workers 3 \
  --out smoke_variantB_typ \
  --resume

python3 compute_variantB_transport.py \
  --out smoke_variantB_typ \
  --n-band 301 \
  --workers 3
```

Quick diagnostics:

```bash
python3 - <<'PY'
import csv
from collections import defaultdict

p="smoke_variantB_typ/transport_summary.csv"
d=defaultdict(list)

with open(p) as f:
    for r in csv.DictReader(f):
        d[r["w1_nf"]].append((
            float(r["dne"]),
            int(float(r["n_causality_bad_thermal"] or 0)),
            float(r["sigma"]),
            float(r["kappa"]),
            float(r["lorenz"]),
        ))

for w1, vals in sorted(d.items(), key=lambda x: float(x[0])):
    print("w1=", w1, "ne=", 1-float(w1))
    print("  dne:", min(v[0] for v in vals), max(v[0] for v in vals))
    print("  bad thermal:", min(v[1] for v in vals), max(v[1] for v in vals))
    print("  sigma:", min(v[2] for v in vals), max(v[2] for v in vals))
    print("  kappa:", min(v[3] for v in vals), max(v[3] for v in vals))
    print("  Lorenz:", min(v[4] for v in vals), max(v[4] for v in vals))
PY
```

## PBS exploratory map

```bash
cd ~/Byczuk_Freericks/filling_variantB_freericks

perl submit_pbs_variantB.pl \
  --T-list 0.02 \
  --w1-list 0.5,0.45,0.4,0.35,0.3 \
  --U-min 0.0 --U-max 2.0 --n-U 21 \
  --Delta-min 0.05 --Delta-max 2.0 --n-Delta 21 \
  --Delta-chunk 1 \
  --kind typ \
  --omega0 12 \
  --n-omega 501 --n-eps 51 \
  --tol 1e-4 --ne-tol 5e-3 \
  --max-iter 800 --max-mu-iter 8 \
  --mix 0.12 \
  --ppn 4 --workers 4 \
  --walltime 12:00:00 --mem 12gb \
  --python python3 \
  --out pbs_variantB_T002_typ \
  --submit
```

After jobs finish:

```bash
bash pbs_variantB_jobs/merge_dmft_results.sh

python3 compute_variantB_transport.py \
  --out pbs_variantB_T002_typ \
  --n-band 301 \
  --workers 4
```

Plot maps:

```bash
python3 plot_variantB_maps.py \
  --out pbs_variantB_T002_typ \
  --w1-list 0.5,0.45,0.4,0.35,0.3 \
  --values sigma,kappa,lorenz,L12,kappa_bare,thermoelectric_correction \
  --kinds typ \
  --norm power --gamma 0.35
```

## Notes

The filling is always computed from the arithmetic LDOS:

```text
n_e = integral f(omega) A_arith(omega) d omega
```

This is deliberate. The typical LDOS is used to build the typical medium, not to define the conserved particle number.

Transport follows Freericks/Joura:

```text
tau(omega) = integral d epsilon Phi(epsilon) A(epsilon, omega)^2
L11 = integral (-df/domega) tau
L12 = integral omega (-df/domega) tau
L22 = integral omega^2 (-df/domega) tau
kappa = (L22 - L12^2/L11) / T
Lorenz = kappa / (T sigma)
```
