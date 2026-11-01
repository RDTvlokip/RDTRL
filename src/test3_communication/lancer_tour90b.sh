#!/bin/bash
# Tour 90 : enregistrements (300 000 pas, 5 departs ks[0..4], x = 1e-4) pour six modules :
#  b1 = 0,92 et 0,85 (b2 = 0,999) ; b2 = 0,9995 (attente 40 000), 0,9999 (attente 80 000), 0,995, 0,99 (b1 = 0,9).
# Six processus au plus a la fois. Usage (depuis src/test3_communication) : bash lancer_tour90b.sh
n=0
lancer() {
  mod=$1; att=$2; lr=$3
  for k in 0 1 2 3 4; do
    TOUR90_ATT=$att python verifier_tour90_memoire.py enregistrer reduction_numba_tour90_$mod 1e-4 $lr $k > /d/tmp/t90_enreg_${mod}_lr${lr}_k${k}.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
}
for lr in 0.05 0.075 0.1 0.125 0.15 0.175 0.1875 0.2 0.2125 0.225 0.2375 0.25 0.275 0.3 0.325; do lancer b1_092 10000 $lr; done
for lr in 0.05 0.075 0.1 0.125 0.15 0.1625 0.175 0.1875 0.2 0.2125 0.225 0.2375; do lancer b1_085 10000 $lr; done
for lr in 0.1 0.125 0.15 0.175 0.2 0.2125 0.225 0.2375 0.25 0.2625 0.275; do lancer b2_09995 40000 $lr; done
for lr in 0.05 0.075 0.1 0.125 0.1375 0.15 0.1625 0.175 0.1875 0.2; do lancer b2_09999 80000 $lr; done
for lr in 0.002 0.004 0.006 0.008 0.01 0.0125 0.015 0.0175 0.02 0.025 0.03 0.04; do lancer b2_0995 10000 $lr; done
for lr in 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.01 0.0125 0.015; do lancer b2_099 10000 $lr; done
wait
echo fini
