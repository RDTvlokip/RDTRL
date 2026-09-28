#!/bin/bash
# Tour 59 (28/09/2026) : grille eps x delta autour du pli, mur 23, etat chauffe
# partage (creer d'abord les caches par un appel a pas=0, cf.
# verifier_tour59_delta_c_dynamique.py).
# Sortie : une ligne par run dans D:/tmp/rdtrl_t59_grille_<eps>_<offset>.txt
# Usage : bash lancer_tour59_grille_eps_delta.sh "<eps...>" "<offsets...>" <pas>
cd "$(dirname "$0")"
export PLI=0.0134372100660973
export PAS=${3:-60000}
EPS_LIST=${1:-"1e-8 3e-8 1e-7"}
OFFS=${2:-"-1e-8 -6e-9 -3e-9 -1e-9 1e-9 3e-9 6e-9 1e-8 2e-8 5e-8"}

un_run() {
  local E=$1 O=$2
  local D
  D=$(python -c "print($PLI + float('$O'))")
  python verifier_tour59_delta_c_dynamique.py mur23 3 4 "$D" "$PAS" eps="$E" chauffe=20000 chauffe_eps=20000 \
    > "/d/tmp/rdtrl_t59_grille_${E}_${O}.txt"
}
export -f un_run

for E in $EPS_LIST; do
  for O in $OFFS; do
    echo "$E $O"
  done
done | xargs -P 12 -L 1 bash -c 'un_run $0 $1'
