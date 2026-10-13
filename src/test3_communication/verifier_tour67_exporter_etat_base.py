"""Tour 67 : exporte l'etat de base de la reduction (x, m, v, Se, Sed, t) en JSON dans le depot.
dipankarsarkar ne pouvait pas rejouer mes runs : base_state() charge D:/tmp/rdtrl_tour59_phase1_mur23_3_4_20000.pt
(38 Ko, hors depot), et Se et Sed y sont calcules. Le JSON contient exactement ce que base_state() renvoie
(floats ecrits avec repr, donc exacts). Usage : python verifier_tour67_exporter_etat_base.py
Chargement ensuite : etat_base_json() dans verifier_tour67_zero_dd.py.
"""

import json
import sys

sys.path.insert(0, ".")
from hasard_reduction_tour59 import base_state

SORTIE = "etat_base_reduction_mur23.json"

if __name__ == "__main__":
    s = base_state()
    out = dict(x=[float(a) for a in s["x"]], m=[float(a) for a in s["m"]], v=[float(a) for a in s["v"]],
               Se=float(s["Se"]), Sed=float(s["Sed"]), t=float(s["t"]),
               origine="phase 1 du reseau complet (20 000 pas Adam a pli-1e-6, mur 23, graine 77777 k=3, message 10, referents 3/4), "
                        "reduit par reduction_numba_tour59.reduire ; coordonnees (e3,o3,e4,o4,l3,l4)")
    with open(SORTIE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(SORTIE, out["x"], out["Se"], out["Sed"], out["t"])
