#!/bin/bash
# Tour 85 : ecart de salve a salve a l'interieur d'un depart (8 lectures par depart), x = 1e-4, lecture directe (TOUR84_ROBUSTE=1, pas 3), niveau exact, moments a l'echelle,
# eps/lr = 2e-9, module a lr runtime ; departs ks[0..4] ; six processus au plus a la fois.
#  - lr 0,05 / 0,1 / 0,15 / 0,2 / 0,25 / 0,3 ; plus lr 0,15, ks[0] au pas 1 (controle de methode)
# Usage (depuis src/test3_communication) : bash lancer_tour85.sh
export TOUR78_ECHELLE=1
export TOUR80_NIVEAU=1
export TOUR83_DETAIL=1
export TOUR84_ROBUSTE=1
export TOUR85_NSAL=8
n=0
lancer() {
  x=$1; lr=$2; k=$3; pas=$4; sortie=$5
  eps=$(awk "BEGIN{printf \"%.4e\", 2e-9*$lr}")
  TOUR84_PAS=$pas TOUR81_LR=$lr TOUR81_K=$k python verifier_tour78_lr.py reduction reduction_numba_tour71_lrvar $eps $x > $sortie 2>&1 &
  n=$((n+1))
  if [ $((n % 6)) -eq 0 ]; then wait; fi
}
for lr in 0.15 0.1 0.05 0.2 0.25 0.3; do
  for k in 0 1 2 3 4; do lancer 1e-4 $lr $k 3 /d/tmp/t85_x1e-4_lr${lr}_k${k}.log; done
done
lancer 1e-4 0.15 0 1 /d/tmp/t85_x1e-4_lr0.15_k0_pas1.log
wait
echo fini
