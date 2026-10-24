#!/bin/bash
# Tour 87 : (1) relecture, avec l'avance de franchissement imprimee, des memes 60 salves aleatoires par depart que le tour 86 (departs ks[0..4]) ;
# (2) jeu de confirmation : departs ks[5..9], enregistrement (npz) puis echantillon aleatoire de 60 salves, lr 0,1 a 0,3. x = 1e-4. Six processus au plus a la fois.
# Usage (depuis src/test3_communication) : bash lancer_tour87.sh
export TOUR78_ECHELLE=1
export TOUR80_NIVEAU=1
export TOUR83_DETAIL=1
export TOUR84_ROBUSTE=1
export TOUR85_NSAL=1000
lecture() {
  lr=$1; k=$2
  eps=$(awk "BEGIN{printf \"%.4e\", 2e-9*$lr}")
  liste=$(python verifier_tour86_longue.py liste $k 60)
  TOUR86_GLIST=$liste TOUR84_PAS=3 TOUR81_LR=$lr TOUR81_K=$k python verifier_tour78_lr.py reduction reduction_numba_tour71_lrvar $eps 1e-4 > /d/tmp/t87_alea_lr${lr}_k${k}.log 2>&1
}
n=0
for lr in 0.3 0.25 0.2 0.15 0.1; do
  for k in 0 1 2 3 4; do
    lecture $lr $k &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo premiere partie finie
n=0
for lr in 0.1 0.15 0.2 0.25 0.3; do
  for k in 5 6 7 8 9; do
    python verifier_tour86_longue.py enregistrer $lr $k > /d/tmp/t87_enreg_lr${lr}_k${k}.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo enregistrements finis
n=0
for lr in 0.3 0.25 0.2 0.15 0.1; do
  for k in 5 6 7 8 9; do
    lecture $lr $k &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo fini
