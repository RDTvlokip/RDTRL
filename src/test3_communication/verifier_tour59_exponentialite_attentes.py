"""Tour 59 (29/09/2026) : mes ajustements MLE de lambda(delta) supposaient un
taux CONSTANT sur toute la duree d'un run (temps d'attente exponentiels).
Le second agent style dipankar affirme que c'est faux au-dessus d'un genou
vers 7,3e-9 (sd/moyenne = 0,74 a 7,0e-9, KS p < 1e-3) et vrai en dessous
(sd/moyenne 0,87-0,96, KS p 0,11-0,85). Verification avec mon code, sur ses
fichiers haz_*.json (reduction, protocole P-A) :
  sd/moyenne des temps d'echappement (exponentielle : 1) ; parts censurees ;
  p de type Lilliefors par bootstrap parametrique (echantillons exponentiels
  de meme taille et de meme moyenne ajustee, 2000 tirages) sur la statistique
  de Kolmogorov-Smirnov -- le p de scipy.kstest avec une moyenne ajustee est
  trop optimiste.
Seuls les offsets dont la censure est < 10 % sont testes (sinon la troncature
biaise sd/moyenne vers le bas).
"""

import glob
import json
import re
import numpy as np
from scipy import stats


def ks_stat(x, scale):
    return stats.kstest(x, "expon", args=(0, scale)).statistic


def lilliefors_p(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = ks_stat(x, x.mean())
    n = len(x)
    plus_grand = 0
    for _ in range(n_boot):
        s = rng.exponential(1.0, n)
        if ks_stat(s, s.mean()) >= obs:
            plus_grand += 1
    return (plus_grand + 1) / (n_boot + 1)


if __name__ == "__main__":
    lignes = []
    for f in glob.glob("D:/tmp/agent_dk61/haz_*.json"):
        d = json.load(open(f))
        res = d["res"]
        n_rep = len(res)
        t = np.array([first + 1 for _, first, _, _ in res if first >= 0], dtype=float)
        cens = 1 - len(t) / n_rep
        lignes.append((d["off"], re.search(r"haz_(.+?)_[0-9.]+\.json$", f).group(1), n_rep, len(t), cens, t))
    print("offset(1e-9) fichier     n_rep  echapp.  censure   moyenne     sd/moyenne   p(Lilliefors)")
    for off, tag, n_rep, n, cens, t in sorted(lignes, key=lambda z: (z[0], z[1])):
        if n < 10:
            print(f"{off:8.2f}    {tag:8s} {n_rep:6d} {n:7d}   {cens:5.2f}    (trop peu d'evenements)")
            continue
        if cens >= 0.10:
            print(f"{off:8.2f}    {tag:8s} {n_rep:6d} {n:7d}   {cens:5.2f}    censure >= 10 % : non teste")
            continue
        p = lilliefors_p(t)
        print(f"{off:8.2f}    {tag:8s} {n_rep:6d} {n:7d}   {cens:5.2f}   {t.mean():10.4g}   {t.std(ddof=1) / t.mean():7.3f}      {p:6.4f}")
