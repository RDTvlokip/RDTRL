#!/bin/bash
# Tour 59 (29/09/2026) : H59-24, taux d'echappement lambda(delta) a eps 1e-10.
# Pour chaque phase k (chauffe_eps = 20 000 + k) et chaque delta : un run de PAS pas
# (defaut 30 000, censure a droite), masque `tout` eps 1e-10 (= Adam standard).
# Etape 1 : creer les 10 caches de phase (pas = 0), en parallele.
# Etape 2 : 10 phases x 6 delta, 12 en parallele.
# Sortie : /d/tmp/rdtrl_t59_hz_k<k>_1e-10_<offset>.txt
# Usage : bash lancer_tour59_hasard.sh ["<offsets>"] [pas]
cd "$(dirname "$0")"
export PLI=0.0134372100660973
export PAS=${2:-30000}
OFFS=${1:-"6.6e-9 6.8e-9 7e-9 7.5e-9 8e-9 1e-8"}
KS=${KS:-"0 1 2 3 5 8 13 21 34 55"}
# prefixe des sorties (hz par defaut) : /d/tmp/rdtrl_t59_${HZ_TAG}_k<k>_1e-10_<offset>.txt
HZ_TAG=${HZ_TAG:-hz}

for k in $KS; do
  python verifier_tour59_masque_eps.py mur23 3 4 "$PLI" 0 masque=tout eps=1e-10 chauffe_eps=$((20000 + k)) > /dev/null &
done
wait

un_run() {
  local K=$1 O=$2
  local D
  D=$(python -c "print($PLI + float('$O'))")
  python verifier_tour59_masque_eps.py mur23 3 4 "$D" "$PAS" masque=tout eps=1e-10 chauffe_eps=$((20000 + K)) \
    > "/d/tmp/rdtrl_t59_${HZ_TAG}_k${K}_1e-10_${O}.txt"
}
export HZ_TAG
export -f un_run

for k in $KS; do
  for O in $OFFS; do
    echo "$k $O"
  done
done | xargs -P 12 -L 1 bash -c 'un_run $0 $1'
