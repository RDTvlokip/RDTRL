"""Tour 62 (07/10/2026) : ou la dynamique de la reduction casse-t-elle reellement, au
voisinage du pli IDEAL ? Le d4 = 1 - s4 de la reduction est ~1,8e-12 (3e4 fois son noeud
3e-17), ce qui decale le pli effectif de (1+delta) d4 / (beta |G_delta|) ~ +9e-13.
Offsets au-dessus du pli ideal : des echappements a +2e-13 signifieraient que le pli ideal
gouverne ; une absence d'echappement jusqu'a ~+1e-12 que le pli effectif gouverne.
12 phases, 2 000 000 pas apres 10 000 de mise en regime, reduction numba.
"""

import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
import reduction_numba_tour59 as redlib
from hasard_reduction_tour59 import base_state, warmed
from verifier_tour60_exposant_reduction import PLI

mp.dps = 40

if __name__ == "__main__":
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 12)]
    s1 = base_state()
    print("offset (depuis le pli ideal)   phases echappees / 12    temps de premiere montee (pas) des phases echappees   d4 median au debut")
    for off in sys.argv[1:]:
        rng = np.random.default_rng(11)
        delta = float(PLI + mpf(off))
        esc, temps, d4s = 0, [], []
        for k in ks:
            w = warmed(s1, k, rng, 1e-9)
            r0 = redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)), cross=0.9, stop=True)
            z4 = float(w["x"][2] - w["x"][3])
            d4s.append(26.0 * np.exp(-z4) / (1.0 + 26.0 * np.exp(-z4)))
            first = r0[0]
            if first < 0:
                first, maxrb, rb, done = redlib.avancer_obs(w, delta, 2_000_000, np.zeros((1, 7)), cross=0.9, stop=True)
                if first >= 0:
                    first += 10000
            if first >= 0:
                esc += 1; temps.append(first)
        print(f"{off:>8s}                        {esc:2d} / 12                  {sorted(temps)[:8]}   {np.median(d4s):.3e}", flush=True)
