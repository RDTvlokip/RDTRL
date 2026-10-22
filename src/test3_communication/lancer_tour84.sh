#!/bin/bash
# Tour 84 : lecture directe (TOUR84_ROBUSTE=1, balayage exact de a-140 a a avant le debut de chaque groupe, pas de 3), niveau exact, moments a l'echelle, eps/lr = 2e-9,
# module a lr runtime ; departs ks[0..4] ; six processus au plus a la fois.
#  - x = 1e-4 : lr 0,225, 0,25, 0,275, 0,35, 0,4 ; x = 1e-6 : lr 0,2 (departs ks[3], ks[4])
# Usage (depuis src/test3_communication) : bash lancer_tour84.sh
export TOUR78_ECHELLE=1
export TOUR80_NIVEAU=1
export TOUR83_DETAIL=1
export TOUR84_ROBUSTE=1
export TOUR84_PAS=3
n=0
lancer() {
  x=$1; lr=$2; k=$3
  eps=$(awk "BEGIN{printf \"%.4e\", 2e-9*$lr}")
  TOUR81_LR=$lr TOUR81_K=$k python verifier_tour78_lr.py reduction reduction_numba_tour71_lrvar $eps $x > /d/tmp/t84_x${x}_lr${lr}_k${k}.log 2>&1 &
  n=$((n+1))
  if [ $((n % 6)) -eq 0 ]; then wait; fi
}
lancer 1e-6 0.2 3; lancer 1e-6 0.2 4
for lr in 0.25 0.225 0.275 0.35 0.4; do
  for k in 0 1 2 3 4; do lancer 1e-4 $lr $k; done
done
wait
echo fini
