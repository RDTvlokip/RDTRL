"""Agent (role dipankarsarkar, tour 59 simule) : la ligne 3 co-oscille-t-elle
AVANT la salve visible, pendant la croissance du mode instable, avec la meme
pente que pendant la salve ? Si oui, le mode couple existe des le
franchissement de 38 (~84 pas avant le pic), et c'est pendant cette phase
que la ligne 3 avance la salve.

Lit les traces du tour 58. Pente = regression des differences premieres
dX sur dgap (derive retiree par la mediane sur la fenetre), fenetres avant
le pic.
"""
import sys
sys.path.insert(0, '.')
from verifier_reponse_dipankar_tour58_evenements_alignes import charger, series, evenements, DEBUT

FENETRES = [(-100, -80), (-80, -60), (-60, -40), (-40, -25), (-25, -12)]

if __name__ == "__main__":
    for delta in (0.013026615, 0.0):
        t = charger(delta)
        s = series(t)
        dX = s["X"][1:] - s["X"][:-1]
        dg = s["gap_r"][1:] - s["gap_r"][:-1]
        print(f"=== delta={delta} ===")
        res = {f: [] for f in FENETRES}
        for p, _ in evenements(s):
            i = p - DEBUT - 1
            if i + FENETRES[0][0] < 0:
                continue
            for a, b in FENETRES:
                x = dX[i + a:i + b]
                g = dg[i + a:i + b]
                x = x - x.median()
                g = g - g.median()
                res[(a, b)].append((((x * g).sum() / (g * g).sum()).item(), g.abs().max().item()))
        med = lambda v: sorted(v)[len(v) // 2]
        for f, v in res.items():
            print(f"  fenetre pic{f[0]:+d}..pic{f[1]:+d} : pente dX/dgap mediane={med([a for a, _ in v]):+.4e}  "
                  f"min/max={min(a for a, _ in v):+.4e}/{max(a for a, _ in v):+.4e}  "
                  f"max|dgap| median={med([b for _, b in v]):.2e}  (n={len(v)})")
