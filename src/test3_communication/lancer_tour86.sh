#!/bin/bash
# Tour 86 : (1) enregistrements longs (300 000 pas) de l'estimateur C/rho, x = 1e-4, lr 0,05 a 0,3, departs ks[0..4], pour la derive au-dela de la salve 8 ;
# (2) relectures avec le jacobien de quatre salves tardives (groupes 50, 200, 500) a lr 0,15 et 0,3, departs ks[0] et ks[1], pour valider C.
# Six processus au plus a la fois.
# Usage (depuis src/test3_communication) : bash lancer_tour86.sh
export TOUR78_ECHELLE=1
export TOUR80_NIVEAU=1
export TOUR83_DETAIL=1
export TOUR84_ROBUSTE=1
n=0
for lr in 0.05 0.1 0.15 0.2 0.25 0.3; do
  for k in 0 1 2 3 4; do
    python verifier_tour86_longue.py enregistrer $lr $k > /d/tmp/t86_enreg_lr${lr}_k${k}.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo enregistrements finis
export TOUR85_NSAL=4
n=0
for lr in 0.15 0.3; do
  for k in 0 1; do
    for g0 in 50 200 500; do
      eps=$(awk "BEGIN{printf \"%.4e\", 2e-9*$lr}")
      TOUR86_G0=$g0 TOUR84_PAS=3 TOUR81_LR=$lr TOUR81_K=$k python verifier_tour78_lr.py reduction reduction_numba_tour71_lrvar $eps 1e-4 > /d/tmp/t86_relecture_lr${lr}_k${k}_g${g0}.log 2>&1 &
      n=$((n+1))
      if [ $((n % 6)) -eq 0 ]; then wait; fi
    done
  done
done
wait
echo fini
