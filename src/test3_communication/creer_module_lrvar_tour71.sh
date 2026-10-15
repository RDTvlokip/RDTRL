#!/bin/sh
# Tour 71 : copie de reduction_numba_tour59.py ou LR devient un argument de run() et run_obs() (module.LRV, modifiable a l'execution) :
# evite une compilation numba par valeur de lr (bissection sur lr). run_until_up et crit_delta ne sont pas utilises ici.
f=reduction_numba_tour71_lrvar.py
{ echo "# Copie de reduction_numba_tour59.py avec lr en argument (tour 71, creer_module_lrvar_tour71.sh) : poser module.LRV avant avancer*."; cat reduction_numba_tour59.py; } > "$f"
sed -i 's/^def run(x, m, v, Se, Sed, t0, delta, eps, nsteps,/def run(x, m, v, Se, Sed, t0, delta, eps, lr_, nsteps,/; s/^def run_obs(x, m, v, Se, Sed, t0, delta, eps, nsteps,/def run_obs(x, m, v, Se, Sed, t0, delta, eps, lr_, nsteps,/' "$f"
sed -i 's/LR \* (m\[/lr_ * (m[/' "$f"
sed -i 's/float(delta), e, int(nsteps), float(cross)/float(delta), e, float(LRV), int(nsteps), float(cross)/' "$f"
sed -i 's/^_REC0 = np.zeros(0)/_REC0 = np.zeros(0)\nLRV = 0.05/' "$f"
grep -n "^def run\|lr_ \* \|LRV\|float(delta), e" "$f" | head -30
