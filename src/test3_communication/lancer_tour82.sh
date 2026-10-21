#!/bin/bash
# Tour 82 : departs multiples ks[0..4] du gain lr p_z, niveau exact (TOUR80_NIVEAU=1), calibration S_c recalee (TOUR82_CALIB=1), moments a l'echelle, eps/lr = 2e-9,
# module a lr runtime (reduction_numba_tour71_lrvar) ; six processus au plus a la fois.
#  - x = 1e-6 : lr 0,05, 0,075, 0,25 (cinq departs) ; plus les departs ks[2] de lr 0,1 et 0,125, ecartes au tour 81
#  - x = 1e-4 : lr 0,05, 0,1, 0,15, 0,2, 0,25, 0,3 (cinq departs)
# Usage (depuis src/test3_communication) : bash lancer_tour82.sh
export TOUR78_ECHELLE=1
export TOUR80_NIVEAU=1
export TOUR81_FEN=30
export TOUR82_CALIB=1
n=0
lancer() {
  x=$1; lr=$2; k=$3; sortie=$4
  eps=$(awk "BEGIN{printf \"%.4e\", 2e-9*$lr}")
  TOUR81_LR=$lr TOUR81_K=$k python verifier_tour78_lr.py reduction reduction_numba_tour71_lrvar $eps $x > $sortie 2>&1 &
  n=$((n+1))
  if [ $((n % 6)) -eq 0 ]; then wait; fi
}
lancer 1e-6 0.1 2 /d/tmp/t81_lr0.1_k2.log
lancer 1e-6 0.125 2 /d/tmp/t81_lr0.125_k2.log
for lr in 0.25 0.05 0.075; do
  for k in 0 1 2 3 4; do lancer 1e-6 $lr $k /d/tmp/t82_x1e-6_lr${lr}_k${k}.log; done
done
for lr in 0.25 0.05 0.1 0.15 0.2 0.3; do
  for k in 0 1 2 3 4; do lancer 1e-4 $lr $k /d/tmp/t82_x1e-4_lr${lr}_k${k}.log; done
done
wait
echo fini
