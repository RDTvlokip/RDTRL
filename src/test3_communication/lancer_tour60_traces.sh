#!/bin/bash
# Tour 60 (29/09/2026) : traces pas a pas (d3, R_b, r4) du reseau complet, mur 23,
# pour la question de dipankarsarkar sur le biais converti en unites de delta.
# Pour chaque eps, chaque offset (delta = pli + offset) et chaque phase k
# (chauffe 20 000 + chauffe_eps = 20 000 + k) : un run trace de PAS pas
# (verifier_tour59_delta_c_dynamique.py, option trace).
# Sortie : /d/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta<delta>_pas<PAS>_eps<eps>_chauffe20000_<20000+k>.txt
# Plafond de 6 processus (NPROC), un seul lanceur a la fois.
# Usage : EPS="1e-10" KS="89 144" bash lancer_tour60_traces.sh "<offsets>" <pas>
cd "$(dirname "$0")"
export PLI=0.0134372100660973
export PAS=${2:-40000}
OFFS=${1:?offsets}
KS=${KS:-"89 144 233 377 610 987"}
export EPS=${EPS:-1e-10}
export NPROC=${NPROC:-6}

un_run() {
  local K=$1 O=$2
  local D
  D=$(python -c "print($PLI + float('$O'))")
  python verifier_tour59_delta_c_dynamique.py mur23 3 4 "$D" "$PAS" trace eps="$EPS" chauffe=20000 chauffe_eps=$((20000 + K)) > /dev/null
}
export -f un_run

for k in $KS; do
  for O in $OFFS; do
    echo "$k $O"
  done
done | xargs -P "$NPROC" -L 1 bash -c 'un_run $0 $1'
