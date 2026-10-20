#!/bin/bash
# Tour 80 : durees, amplitudes et remplissages des salves (verifier_tour79_duree_salves.py) a x = 1e-4 et 1e-5, eps/lr = 2e-9, moments a l'echelle ; six processus au plus a la fois.
# Usage (depuis src/test3_communication) : bash lancer_tour80.sh
export TOUR78_ECHELLE=1
declare -A MOD=( [0.05]=reduction_numba_tour59 [0.1]=reduction_numba_tour67_lr010 [0.2]=reduction_numba_tour69_lr020 [0.3]=reduction_numba_tour70_lr030 )
declare -A EPS=( [0.05]=1e-10 [0.1]=2e-10 [0.2]=4e-10 [0.3]=6e-10 )
n=0
for x in 1e-4 1e-5; do
  for lr in 0.05 0.1 0.2 0.3; do
    python verifier_tour79_duree_salves.py ${MOD[$lr]} ${EPS[$lr]} $x > /d/tmp/t80_dur_lr${lr}_x${x}.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo fini
