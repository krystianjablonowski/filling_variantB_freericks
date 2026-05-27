#!/bin/bash
set -e
cd '/home/2/kj405942/Byczuk_Freericks/filling_variantB_freericks'
out='/home/2/kj405942/Byczuk_Freericks/filling_variantB_freericks/pbs_variantB_T002_typ_21x21_mu20/dmft_summary.csv'
first=1
rm -f "$out"
for f in '/home/2/kj405942/Byczuk_Freericks/filling_variantB_freericks/pbs_variantB_T002_typ_21x21_mu20'/chunks/*/dmft_summary.csv; do
  [ -f "$f" ] || continue
  if [ $first -eq 1 ]; then cat "$f" > "$out"; first=0; else tail -n +2 "$f" >> "$out"; fi
done
echo "Merged DMFT CSV: $out"
