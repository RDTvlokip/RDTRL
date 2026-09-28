"""Tour 59 (29/09/2026) : verification de l'affirmation de l'agent style
dipankar (point 2) : la variance de u = log(r3/r4) sur les traces a
pli-1e-6 est-elle essentiellement RAPIDE (au-dessus de 1/100 pas^-1) ?
Puissance spectrale (FFT, fenetre = 30 000 derniers pas) au-dessus de la
frequence 1/100, et au-dessus de 1/1000 pour comparaison. Il annonce
98,8 / 99,9 / 100 / 100 % pour eps 1e-10 / 1e-8 / 3e-8 / 1e-7.
"""

import glob
import re
import numpy as np

if __name__ == "__main__":
    fichiers = glob.glob("D:/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta0.0134362100660973_pas40000_eps*_chauffe20000_20000.txt")
    for f in sorted(fichiers, key=lambda s: float(re.search(r"_eps([0-9.e+-]+)_chauffe", s).group(1))):
        eps = re.search(r"_eps([0-9.e+-]+)_chauffe", f).group(1)
        r4 = np.loadtxt(f)[-30000:, 3]
        u = np.log((1 - r4) / r4)
        x = u - u.mean()
        P = np.abs(np.fft.rfft(x)) ** 2
        fr = np.fft.rfftfreq(len(x))
        tot = P[1:].sum()
        if tot == 0 or not np.isfinite(tot) or tot < 1e-30:
            print(f"eps={eps:6s} variance nulle (pas de salves)")
            continue
        print(f"eps={eps:6s} part de var(u) au-dessus de 1/100 : {100 * P[fr > 1 / 100].sum() / tot:6.2f} %   "
              f"au-dessus de 1/1000 : {100 * P[fr > 1 / 1000].sum() / tot:6.2f} %   var(u)={x.var():.3e}")
