#!/bin/bash
# Tour 79 : derive du bloc (gap, sender) avec lr a eps/lr = 2e-9, a plusieurs x, moments mis a l'echelle ; six processus au plus a la fois.
# Usage (depuis src/test3_communication) : bash lancer_tour79.sh
export TOUR78_ECHELLE=1
declare -A MOD=( [0.05]=reduction_numba_tour59 [0.1]=reduction_numba_tour67_lr010 [0.2]=reduction_numba_tour69_lr020 [0.3]=reduction_numba_tour70_lr030 )
declare -A EPS=( [0.05]=1e-10 [0.1]=2e-10 [0.2]=4e-10 [0.3]=6e-10 )
n=0
for x in 1e-6 1e-5 1e-4; do
  for lr in 0.05 0.1 0.2 0.3; do
    python verifier_tour78_lr.py reduction ${MOD[$lr]} ${EPS[$lr]} $x > /d/tmp/t79_red_lr${lr}_x${x}.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
# test direct du x effectif : lr 0,05, eps 1e-10, x = 1,06 fois 4,10595e-4
python verifier_tour78_lr.py reduction reduction_numba_tour59 1e-10 4.35231e-4 > /d/tmp/t79_red_lr0.05_x4.35e-4.log 2>&1 &
wait
echo fini
