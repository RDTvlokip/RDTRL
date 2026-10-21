#!/bin/bash
# Tour 81 : departs frais a x = 1e-6, niveau exact (TOUR80_NIVEAU=1), moments a l'echelle, eps/lr = 2e-9, module a lr runtime ; six processus au plus a la fois.
# lr 0,1 0,125 0,15 0,175 0,2 0,3 et departs ks[0], ks[1], ks[2] (indices 0, 1, 2) : 18 runs.
# Usage (depuis src/test3_communication) : bash lancer_tour81.sh
export TOUR78_ECHELLE=1
export TOUR80_NIVEAU=1
n=0
for k in 0 1 2; do
  for lr in 0.1 0.125 0.15 0.175 0.2 0.3; do
    eps=$(awk "BEGIN{printf \"%.4e\", 2e-9*$lr}")
    TOUR81_LR=$lr TOUR81_K=$k python verifier_tour78_lr.py reduction reduction_numba_tour71_lrvar $eps 1e-6 > /d/tmp/t81_lr${lr}_k${k}.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo fini
