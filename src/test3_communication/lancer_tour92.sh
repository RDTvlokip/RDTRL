#!/bin/bash
# Tour 92 : enregistrements (300 000 pas, 5 departs, x = 1e-4, base) aux sept lr 0,1 a 0,25, ecrits en npz pour verifier_tour92_budget.py.
# Six processus au plus a la fois. Usage (depuis src/test3_communication) : bash lancer_tour92.sh
n=0
for lr in 0.1 0.125 0.15 0.175 0.2 0.225 0.25; do
  for k in 0 1 2 3 4; do
    python verifier_tour92_budget.py enregistrer $lr $k > /d/tmp/t92_enreg_lr${lr}_k${k}.log 2>&1 &
    n=$((n+1))
    if [ $((n % 6)) -eq 0 ]; then wait; fi
  done
done
wait
echo fini
