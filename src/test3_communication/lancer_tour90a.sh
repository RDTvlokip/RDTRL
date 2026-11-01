#!/bin/bash
# Tour 90 (essai) : structure des groupes seule (f, calme median, duree mediane ; aucune moyenne de 1/rho imprimee), un depart, x = 1e-4,
# pour b1 = 0,85 et 0,92 (b2 = 0,999) et pour b2 = 0,99 / 0,995 / 0,9995 / 0,9999 (b1 = 0,9).
# Memoire de v = 1 / (1 - b2) : 100, 200, 1 000, 2 000, 10 000 pas. Attente avant l'enregistrement : 10 000 pas, sauf 40 000 (b2 = 0,9995) et 80 000 (b2 = 0,9999).
# Six processus au plus a la fois. Usage (depuis src/test3_communication) : bash lancer_tour90a.sh
n=0
for cfg in "reduction_numba_tour90_b1_085 10000" "reduction_numba_tour90_b1_092 10000" "reduction_numba_tour90_b2_099 10000" "reduction_numba_tour90_b2_0995 10000" "reduction_numba_tour90_b2_09995 40000" "reduction_numba_tour90_b2_09999 80000"; do
  set -- $cfg
  for lr in 0.05 0.1 0.15 0.2 0.25 0.3 0.4; do
    TOUR90_ATT=$2 python verifier_tour90_memoire.py essai $1 1e-4 $lr > /d/tmp/t90_essai_${1#reduction_numba_tour90_}_lr$lr.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo fini
