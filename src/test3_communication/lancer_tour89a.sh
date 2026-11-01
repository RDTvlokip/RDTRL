#!/bin/bash
# Tour 89 (essai) : structure des groupes seule (f, calme median, duree mediane ; aucune moyenne de 1/rho imprimee) pour quatre configurations, un depart, sept lr :
# b1 = 0,8 et 0,95 a x = 1e-4 ; x = 1e-5 et 1e-3 a b1 = 0,9. Six processus au plus a la fois.
# Usage (depuis src/test3_communication) : bash lancer_tour89a.sh
n=0
for cfg in "0.8 1e-4" "0.95 1e-4" "0.9 1e-5" "0.9 1e-3"; do
  set -- $cfg
  for lr in 0.05 0.1 0.15 0.2 0.25 0.3 0.4; do
    python verifier_tour89_poids.py essai $1 $2 $lr > /d/tmp/t89_essai_b$1_x$2_lr$lr.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo fini
