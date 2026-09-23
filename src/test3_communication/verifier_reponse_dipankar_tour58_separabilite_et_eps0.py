"""Tour 58 (23/09/2026, VRAIE critique de dipankarsarkar) : deux points
laisses sales par l'ablation d'eps sur la ligne 3.

1. H58.5 (separabilite) -- mon premier test comparait le gradient
   STATIQUE de la ligne 3 sous deux ponderations. Mal concu : pres d'un
   point stationnaire ce gradient est ~0 et le rapport ne veut rien dire
   (6,33 et 1,00 imprimes). Le bon test est la REPONSE du gradient de la
   ligne 3 a une salve du recepteur (perturbation du gap r4-r3 de 2h),
   au meme etat, sous quatre ponderations :
     W0  : delta=0
     Wr  : delta reel
     W4  : seul poids[4] change (poids[3]=1/N)
     W3  : seul poids[3] change (poids[4]=1/N)
   Prediction de la separabilite (d J / d s[3,m] = poids[3] r[m,3] + ...,
   jamais poids[4]) : reponse(W4)/reponse(W0) = 1 exactement, et
   reponse(W3)/reponse(W0) = reponse(Wr)/reponse(W0) = 1-delta = 0,98697.
   Et la forme fermee de la reponse sur e3[10] :
     d g(-J)/e3[10] = -s3(1-s3) poids[3] d r[10,3],  d r[10,3] = -r3 r4 (2h)
   a comparer au calcul autograd, aux deux etats.

2. eps3=0 donne une pente 0,68 a delta=0, pas ~1. La formule supposait
   des pas normalises egaux (m/sqrt v) sur e3[10] et r10[4] et un pas
   de -1/26 de celui de e3[10] sur chacun des 26 autres. On trace pas a
   pas pendant la salve : rho = m/sqrt(v) sur e3[10], r10[4], r10[3], et
   la moyenne des pas des 26 autres, pour voir quelle hypothese casse.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import BETA
from representable_atteignable_stable import N
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, poids_pour, LR, ADAM_EPS

DELTA_REEL = 0.013026615
H = 1e-5


def ponderation(p3, p4):
    w = torch.full((N,), 1.0 / N, dtype=torch.float64)
    w[3], w[4] = p3, p4
    return w


def grad_e3(e, r, w):
    for p in list(e.p) + list(r.p):
        p.grad = None
    j, _ = objectif_pondere(e, r, BETA, w)
    (-j).backward()
    g = e.p[0].grad[3].clone()
    for p in list(e.p) + list(r.p):
        p.grad = None
    return g


def reponse(e, r, w):
    g0 = grad_e3(e, r, w)
    with torch.no_grad():
        r.p[0][10, 4] += H
        r.p[0][10, 3] -= H
    g1 = grad_e3(e, r, w)
    with torch.no_grad():
        r.p[0][10, 4] -= H
        r.p[0][10, 3] += H
    return g1 - g0


def avancer(e, r, opt, poids, n):
    for _ in range(n):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()


def test_separabilite(delta, pas_cible):
    e, r, opt, poids = reprendre(delta)
    avancer(e, r, opt, poids, pas_cible - 54000)
    d = DELTA_REEL
    W = {"W0": ponderation(1 / N, 1 / N), "Wr": ponderation((1 - d) / N, (1 + d) / N),
         "W4": ponderation(1 / N, (1 + d) / N), "W3": ponderation((1 - d) / N, 1 / N)}
    rep = {k: reponse(e, r, w) for k, w in W.items()}
    print(f"=== etat delta={delta}, pas={pas_cible} ===")
    for k in ("Wr", "W4", "W3"):
        ratio = (rep[k] @ rep["W0"]) / (rep["W0"] @ rep["W0"])
        ecart = ((rep[k] - ratio * rep["W0"]).norm() / rep["W0"].norm()).item()
        print(f"  reponse({k})/reponse(W0) = {ratio.item():.10f}   (residu hors colinearite {ecart:.1e})")
    with torch.no_grad():
        s = torch.softmax(e.p[0][3], 0)
        rr = torch.softmax(r.p[0][10], 0)
        s3, r3, r4 = s[10].item(), rr[3].item(), rr[4].item()
    predit = -s3 * (1 - s3) * (1 / N) * (-r3 * r4 * 2 * H)
    print(f"  reponse(W0) sur e3[10] : autograd={rep['W0'][10].item():+.6e}   forme fermee={predit:+.6e}   "
          f"(s3(1-s3)={s3*(1-s3):.4e}, r3r4={r3*r4:.4f})")
    autres = [j for j in range(N) if j != 10]
    print(f"  reponse sur les 26 autres / reponse sur e3[10] : moyenne={(rep['W0'][autres].mean()/rep['W0'][10]).item():+.6f}"
          f"  (attendu -1/26={-1/26:+.6f}), somme de la ligne={rep['W0'].sum().item():+.2e}")
    return rep["W0"][10].item()


def trace_eps0():
    e, r, opt, poids = reprendre(0.0)
    avancer(e, r, opt, poids, 59519 - 54000)
    p_e, p_r = e.p[0], r.p[0]
    autres = [j for j in range(N) if j != 10]
    print("=== delta=0, eps ligne 3 = 0 a partir de pas=59519 : pas normalises pendant la salve ===")
    lignes = []
    for k in range(700):
        X_av = (p_e[3, 10] - p_e[3, autres].mean()).item()
        gap_av = (p_r[10, 4] - p_r[10, 3]).item()
        e3_av, ob_av = p_e[3, 10].item(), p_e[3, autres].mean().item()
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            st = opt.state[p_e]
            m, rv = st["exp_avg"][3], st["exp_avg_sq"][3].sqrt()
            p_e[3] += LR * m / (rv + ADAM_EPS) - LR * m / rv
            sr = opt.state[p_r]
            rho_e3 = (m[10] / rv[10]).item()
            rho_aut = (m[autres] / rv[autres]).mean().item()
            rho_r4 = (sr["exp_avg"][10, 4] / sr["exp_avg_sq"][10, 4].sqrt()).item()
            rho_r3 = (sr["exp_avg"][10, 3] / sr["exp_avg_sq"][10, 3].sqrt()).item()
            dX = (p_e[3, 10] - p_e[3, autres].mean()).item() - X_av
            dgap = (p_r[10, 4] - p_r[10, 3]).item() - gap_av
            de3 = p_e[3, 10].item() - e3_av
            dob = p_e[3, autres].mean().item() - ob_av
        lignes.append((59519 + k, dgap, dX, de3, dob, rho_e3, rho_aut, rho_r4, rho_r3))
    salve = [l for l in lignes if abs(l[1]) > 0.004]
    for l in salve[:12]:
        print(f"  pas={l[0]} dgap={l[1]:+.3e} dX={l[2]:+.3e} de3={l[3]:+.3e} d(moy autres)={l[4]:+.3e}  "
              f"rho_e3={l[5]:+.4f} rho_autres(moy)={l[6]:+.4f} rho_r4={l[7]:+.4f} rho_r3={l[8]:+.4f}")
    num = sum(l[2] * l[1] for l in salve)
    den = sum(l[1] ** 2 for l in salve)
    print(f"  pente dX/dgap sur les pas de salve = {num/den:.4f}")
    part_e3 = sum(l[3] * l[1] for l in salve) / den
    part_aut = sum(-l[4] * l[1] for l in salve) / den
    print(f"  decomposition : part de e3[10] = {part_e3:.4f}   part de -moy(autres) = {part_aut:.4f}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    a = test_separabilite(0.0, 59519)
    b = test_separabilite(DELTA_REEL, 59929)
    print(f"rapport des reponses sur e3[10], etat delta reel / etat delta=0 = {b/a:.4e}")
    trace_eps0()
