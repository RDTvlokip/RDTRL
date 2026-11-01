#!/bin/bash
# Tour 89 : enregistrements (300 000 pas, 5 departs ks[0..4]) avec M, T, T_osc, T_calme, f, calme et duree medians :
#  - configuration de base (b1 = 0,9, x = 1e-4), grille du tour 88 plus 0,05, 0,075, 0,325, 0,35 ;
#  - x = 1e-3 : lr 0,1 a 0,4 (pas 0,025) ; x = 1e-5 : lr 0,05, 0,075, puis 0,1 a 0,25 (pas 0,0125) ;
#  - b1 = 0,95, x = 1e-4 : lr 0,02 a 0,14 (pas 0,01), 0,16, 0,2.
# b1 = 0,8 s'effondre des lr 0,15 (voir l'essai) et n'est pas scanne. Six processus au plus a la fois.
# Usage (depuis src/test3_communication) : bash lancer_tour89b.sh
n=0
lancer() {
  b1=$1; x=$2; lr=$3
  for k in 0 1 2 3 4; do
    python verifier_tour89_poids.py enregistrer $b1 $x $lr $k > /d/tmp/t89_enreg_b${b1}_x${x}_lr${lr}_k${k}.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
}
for lr in 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35; do lancer 0.9 1e-4 $lr; done
for lr in 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4; do lancer 0.9 1e-3 $lr; done
for lr in 0.05 0.075 0.1 0.1125 0.125 0.1375 0.15 0.1625 0.175 0.1875 0.2 0.2125 0.225 0.2375 0.25; do lancer 0.9 1e-5 $lr; done
for lr in 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.16 0.2; do lancer 0.95 1e-4 $lr; done
wait
echo fini
