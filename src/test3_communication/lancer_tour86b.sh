#!/bin/bash
# Tour 86 (suite) : echantillon aleatoire de 60 salves par depart (groupes 9 a 559, graine 860 + k), lecture directe au jacobien, x = 1e-4, lr 0,1 a 0,3, departs ks[0..4].
# Les enregistrements longs (t86_lr*_k*.npz, lancer_tour86.sh) doivent exister. Six processus au plus a la fois.
# Usage (depuis src/test3_communication) : bash lancer_tour86b.sh
export TOUR78_ECHELLE=1
export TOUR80_NIVEAU=1
export TOUR83_DETAIL=1
export TOUR84_ROBUSTE=1
export TOUR85_NSAL=1000
n=0
for lr in 0.3 0.25 0.2 0.15 0.1; do
  eps=$(awk "BEGIN{printf \"%.4e\", 2e-9*$lr}")
  for k in 0 1 2 3 4; do
    liste=$(python verifier_tour86_longue.py liste $k 60)
    TOUR86_GLIST=$liste TOUR84_PAS=3 TOUR81_LR=$lr TOUR81_K=$k python verifier_tour78_lr.py reduction reduction_numba_tour71_lrvar $eps 1e-4 > /d/tmp/t86_alea_lr${lr}_k${k}.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo fini
