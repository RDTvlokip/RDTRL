"""Agent (role dipankarsarkar, tour 59 simule) : pourquoi le test A (ligne 3
gelee sur [59600, 59900) puis relachee) avance la salve de 52 pas au lieu de
ne rien faire. Imprime X, m/(sqrt v + eps) sur e3[10] et dgap autour de la
relache, pour la base et pour A ; puis |dgap| de la base de 59940 au pic
59989 (a quel pas la base atteint-elle l'amplitude que la relache injecte ?).
delta reel seulement.
"""
import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import BETA
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS

AUT = [j for j in range(27) if j != 10]
POINTS = (59700, 59800, 59890, 59899, 59900, 59901, 59902, 59905, 59910, 59915,
          59920, 59925, 59930, 59937)


def pas_adam(e, r, opt, poids):
    j, _ = objectif_pondere(e, r, BETA, poids)
    opt.zero_grad()
    (-j).backward()
    opt.step()


if __name__ == "__main__":
    torch.set_num_threads(1)
    for gel in (False, True):
        e, r, opt, poids = reprendre(0.013026615)
        for _ in range(59600 - 54000):
            pas_adam(e, r, opt, poids)
        pe, pr = e.p[0], r.p[0]
        gp = (pr[10, 4] - pr[10, 3]).item()
        print("=== gel [59600,59900)" if gel else "=== base",
              f" X(59600)={(pe[3, 10] - pe[3, AUT].mean()).item():.6f}")
        for et in range(59600, 59990):
            pas_adam(e, r, opt, poids)
            st = opt.state[pe]
            m, rv = st["exp_avg"][3], st["exp_avg_sq"][3].sqrt()
            rho = (m[10] / (rv[10] + ADAM_EPS)).item()
            if gel and et < 59900:
                with torch.no_grad():
                    pe[3] += LR * m / (rv + ADAM_EPS)
            g = (pr[10, 4] - pr[10, 3]).item()
            dg, gp = g - gp, g
            X = (pe[3, 10] - pe[3, AUT].mean()).item()
            if et in POINTS or (not gel and et >= 59940 and (abs(dg) > 1e-4 or et % 5 == 0)):
                print(f"  pas={et} X={X:.6f} m/(sqrt v+eps) e3[10]={rho:+.4f} dgap={dg:+.3e}")
            if gel and et > 59940:
                break
