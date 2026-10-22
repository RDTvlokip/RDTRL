#!/bin/bash
# Tour 83 : degre de fusion des salves et lectures par salve, niveau exact (TOUR80_NIVEAU=1), calibration recalee (TOUR82_CALIB=1), moments a l'echelle, eps/lr = 2e-9,
# module a lr runtime ; departs ks[0..4] ; six processus au plus a la fois.
#  - x = 1e-6 : lr 0,2, 0,225, 0,25, 0,275, 0,3, 0,35 ; x = 1e-4 : lr 0,3
# Usage (depuis src/test3_communication) : bash lancer_tour83.sh
export TOUR78_ECHELLE=1
export TOUR80_NIVEAU=1
export TOUR81_FEN=30
export TOUR82_CALIB=1
export TOUR83_DETAIL=1
n=0
lancer() {
  x=$1; lr=$2; k=$3
  eps=$(awk "BEGIN{printf \"%.4e\", 2e-9*$lr}")
  TOUR81_LR=$lr TOUR81_K=$k python verifier_tour78_lr.py reduction reduction_numba_tour71_lrvar $eps $x > /d/tmp/t83_x${x}_lr${lr}_k${k}.log 2>&1 &
  n=$((n+1))
  if [ $((n % 6)) -eq 0 ]; then wait; fi
}
for lr in 0.3 0.2 0.225 0.25 0.275 0.35; do
  for k in 0 1 2 3 4; do lancer 1e-6 $lr $k; done
done
for k in 0 1 2 3 4; do lancer 1e-4 0.3 $k; done
wait
echo fini
