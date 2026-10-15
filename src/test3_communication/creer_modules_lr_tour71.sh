#!/bin/sh
# Tour 71 : une copie de reduction_numba_tour59.py par lr (numba fige LR a la compilation, cf. reduction_numba_tour63_lr002.py).
# Usage : sh creer_modules_lr_tour71.sh 0.305 0.31 ...   -> reduction_numba_tour71_lr0305.py etc. (nom = lr sans le point)
for lr in "$@"; do
  tag=$(echo "$lr" | tr -d '.')
  f="reduction_numba_tour71_lr${tag}.py"
  { echo "# Copie de reduction_numba_tour59.py avec LR = $lr (tour 71)."; cat reduction_numba_tour59.py; } > "$f"
  sed -i "s/^B, C, LR, N = 0.02, 26.0, 0.05, 27.0/B, C, LR, N = 0.02, 26.0, $lr, 27.0/" "$f"
  grep -n "^B, C, LR" "$f"
done
