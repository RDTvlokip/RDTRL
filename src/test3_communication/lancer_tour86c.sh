#!/bin/bash
# Tour 86 (suite) : la derive des huit premieres salves depend-elle de l'attente avant l'enregistrement ? Attente 30 000 et 100 000 pas (protocole du tour 85 : 10 000), lr 0,2 et 0,3, departs ks[0..4], 300 000 pas enregistres.
# Usage (depuis src/test3_communication) : bash lancer_tour86c.sh ; puis python verifier_tour86_longue.py attente
n=0
for att in 30000 100000; do
  for lr in 0.2 0.3; do
    for k in 0 1 2 3 4; do
      TOUR86_ATT=$att python verifier_tour86_longue.py enregistrer $lr $k > /d/tmp/t86_att${att}_lr${lr}_k${k}.log 2>&1 &
      n=$((n+1))
      if [ $((n % 6)) -eq 0 ]; then wait; fi
    done
  done
done
wait
echo fini
