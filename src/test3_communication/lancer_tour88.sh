#!/bin/bash
# Tour 88 : (1) variantes de protocole, lectures directes des memes 60 salves aleatoires par depart (ks[0..2]), dernier franchissement, lr 0,1 a 0,3 :
#     pas de balayage 1 et 6 (fenetre 140), fenetre 210 (TOUR88_FEN=105, pas 3) ; reference = t87d_alea_* (pas 3, fenetre 140) ;
# (2) lr 0,125 / 0,175 / 0,225 / 0,275, departs ks[0..4] : enregistrement (npz) puis lecture de reference (pas 3, fenetre 140, dernier franchissement).
# x = 1e-4. Six processus au plus a la fois.
# Usage (depuis src/test3_communication) : bash lancer_tour88.sh
export TOUR78_ECHELLE=1
export TOUR80_NIVEAU=1
export TOUR83_DETAIL=1
export TOUR84_ROBUSTE=1
export TOUR85_NSAL=1000
export TOUR87_DERNIER=1
lecture() {
  # lecture <nom> <lr> <k> <pas> <demi-fenetre>
  nom=$1; lr=$2; k=$3; pas=$4; fen=$5
  eps=$(awk "BEGIN{printf \"%.4e\", 2e-9*$lr}")
  liste=$(python verifier_tour86_longue.py liste $k 60)
  TOUR88_FEN=$fen TOUR86_GLIST=$liste TOUR84_PAS=$pas TOUR81_LR=$lr TOUR81_K=$k python verifier_tour78_lr.py reduction reduction_numba_tour71_lrvar $eps 1e-4 > /d/tmp/t88_${nom}_alea_lr${lr}_k${k}.log 2>&1
}
n=0
for variante in "pas1 1 70" "pas6 6 70" "fen210 3 105"; do
  set -- $variante
  for lr in 0.3 0.25 0.2 0.15 0.1; do
    for k in 0 1 2; do
      lecture $1 $lr $k $2 $3 &
      n=$((n+1))
      if [ $((n % 6)) -eq 0 ]; then wait; fi
    done
  done
done
wait
echo variantes finies
n=0
for lr in 0.125 0.175 0.225 0.275; do
  for k in 0 1 2 3 4; do
    python verifier_tour86_longue.py enregistrer $lr $k > /d/tmp/t88_enreg_lr${lr}_k${k}.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo enregistrements finis
n=0
for lr in 0.125 0.175 0.225 0.275; do
  for k in 0 1 2 3 4; do
    lecture ref $lr $k 3 70 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo fini
