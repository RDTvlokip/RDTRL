"""Tour 59 (28/09/2026) : version SIMPLE de H59-11 (moyenne de Kapitza).

Si le point fixe moyenne sur les salves verifie <G(u)> = 0, avec G la
residuelle de la fermeture recepteur de dipankarsarkar,
    G(u, d) = -u - ((1+d) s4 - (1-d) s3) / beta,  u = log(r3/r4),
alors autour du pli (G'(u*) = 0) :
    G_d (d - d_c) + 1/2 G'' (sigma_u^2 + m^2) = 0
    =>  Delta = - G'' (sigma_u^2 + m^2) / (2 G_d),
de signe fixe (celui de -G''/G_d) et proportionnel a la variance de u.
Ici : G'' et G_d au pli (mpmath, differences finies a 40 chiffres),
sigma_u^2 lu sur les traces (u = log((1-r4)/r4), r4 = r[msg,b]).
Prediction du carnet : Delta_simple depasse Delta observe d'un facteur
30 a 300, et son signe est le meme pour tout eps.
"""

import glob
import re
import numpy as np
from mpmath import mp, mpf, exp, log, diff

mp.dps = 40
B = mpf("0.02")
C = 26


def G(u, d):
    r3 = 1 / (1 + exp(-u)); r4 = 1 / (1 + exp(u))
    s3 = 1 / (1 + C * exp(-(1 - d) * r3 / B)); s4 = 1 / (1 + C * exp(-(1 + d) * r4 / B))
    return -u - ((1 + d) * s4 - (1 - d) * s3) / B


if __name__ == "__main__":
    u_star, d_c = mpf("-1.47814609082"), mpf("0.0134372100660973")
    Gpp = diff(lambda x: G(x, d_c), u_star, 2)
    Gp = diff(lambda x: G(x, d_c), u_star, 1)
    Gd = diff(lambda x: G(u_star, x), d_c, 1)
    print(f"au pli : G'(u*)={float(Gp):+.2e} (doit etre ~0)   G''={float(Gpp):+.4f}   G_d={float(Gd):+.3f}")
    signe = -float(Gpp) / float(Gd)
    print(f"Delta = -G'' (sigma^2+m^2)/(2 G_d) = {signe / 2:+.4f} * (sigma_u^2 + m^2)   signe {'+' if signe > 0 else '-'}")
    observe = {"1e-10": 6.5e-9, "1e-08": 9.1e-9, "3e-08": 3.2e-10, "1e-07": -5.5e-9, "1e-06": 0.0}
    fichiers = glob.glob("D:/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta0.0134362100660973_pas40000_eps*_chauffe20000_20000.txt")
    for f in sorted(fichiers, key=lambda s: float(re.search(r"_eps([0-9.e+-]+)_chauffe", s).group(1))):
        eps = re.search(r"_eps([0-9.e+-]+)_chauffe", f).group(1)
        T = np.loadtxt(f)[-30000:]
        r4 = T[:, 3]
        u = np.log((1 - r4) / r4)
        # m^2 = (moyenne - u*)^2 est surtout la distance DETERMINISTE nœud-pli a pli-1e-6
        # (~3e-5, identique pour tous les eps, meme a eps 1e-6 sans salves) : ce n'est pas
        # un effet des salves et n'entre pas dans le decalage. Seul sigma_u^2 est le terme de Kapitza.
        m2 = (u.mean() - float(u_star)) ** 2
        var = u.var()
        dp = signe / 2 * var
        ref = observe.get(eps)
        rap = f"   rapport Delta_kapitza/Delta_obs = {dp / ref:+.1f}" if ref else ""
        print(f"eps={eps:6s} sigma_u^2={var:.3e} (m^2={m2:.2e}, deterministe, exclu)   "
              f"Delta_kapitza={dp:+.3e}   Delta_obs={ref:+.2e}{rap}")
