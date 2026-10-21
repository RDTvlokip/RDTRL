#!/bin/bash
# Tour 81 (suite) : departs manquants (fenetre de balayage trop etroite) et departs ks[3], ks[4], x = 1e-6, niveau exact, moments a l'echelle, eps/lr = 2e-9 ; six processus au plus a la fois.
# Usage (depuis src/test3_communication) : bash lancer_tour81b.sh
export TOUR78_ECHELLE=1
export TOUR80_NIVEAU=1
export TOUR81_FEN=30
n=0
lancer() {
  lr=$1; k=$2
  eps=$(awk "BEGIN{printf \"%.4e\", 2e-9*$lr}")
  TOUR81_LR=$lr TOUR81_K=$k python verifier_tour78_lr.py reduction reduction_numba_tour71_lrvar $eps 1e-6 > /d/tmp/t81_lr${lr}_k${k}.log 2>&1 &
  n=$((n+1))
  if [ $((n % 6)) -eq 0 ]; then wait; fi
}
lancer 0.1 1; lancer 0.1 2; lancer 0.125 2; lancer 0.3 2
for k in 3 4; do
  for lr in 0.1 0.125 0.15 0.175 0.2 0.3; do lancer $lr $k; done
done
wait
echo fini
