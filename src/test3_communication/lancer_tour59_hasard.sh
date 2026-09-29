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

# NPROC : nombre maximal de processus python a la fois (6 : plafond fixe par Theo, 29/09/2026 ;
# ne jamais lancer deux lanceurs en meme temps, le plafond vaut pour le total)
export NPROC=${NPROC:-6}
# EPS : eps d'Adam (1e-10 par defaut ; le nom des fichiers en depend) ; CAS A B : collision (mur23 3 4 par defaut)
export EPS=${EPS:-1e-10}
export CAS=${CAS:-mur23}
export A=${A:-3}
export B=${B:-4}
# LRA : taux d'apprentissage d'Adam (0,05 par defaut) ; TAG des sorties a changer si LRA change
export LRA=${LRA:-0.05}
for k in $KS; do
  echo "$k"
done | xargs -P "$NPROC" -I{} bash -c 'python verifier_tour59_masque_eps.py "$CAS" "$A" "$B" "$PLI" 0 masque=tout eps="$EPS" chauffe_eps=$((20000 + {})) lr="$LRA" > /dev/null'

un_run() {
  local K=$1 O=$2
  local D
  D=$(python -c "print($PLI + float('$O'))")
  python verifier_tour59_masque_eps.py "$CAS" "$A" "$B" "$D" "$PAS" masque=tout eps="$EPS" chauffe_eps=$((20000 + K)) lr="$LRA" \
    > "/d/tmp/rdtrl_t59_${HZ_TAG}_k${K}_${EPS}_${O}.txt"
}
export HZ_TAG
export -f un_run

for k in $KS; do
  for O in $OFFS; do
    echo "$k $O"
  done
done | xargs -P "$NPROC" -L 1 bash -c 'un_run $0 $1'
