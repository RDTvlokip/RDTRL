"""Tour 68 (vraie critique de dipankarsarkar, 15/10/2026) : sa question. A la fin de la phase 1 du reseau complet
(checkpoint rdtrl_tour59_phase1_mur23_3_4_20000.pt : 20 000 pas Adam, lr 0,05, eps 1e-10, mur 23 a delta = pli - 1e-6),
quelle est la somme des gradients g sur les 25 logits de reference de la ligne 10 du recepteur (hors referents 3 et 4),
contre le +6,99e-13 gele de la reduction ?

Le gradient est celui de l'objectif du reseau complet (objectif_pondere, autograd, float64). Comme la ligne 10 du recepteur
passe par un softmax, la somme des gradients sur ses 27 logits est identiquement nulle : somme(25 de queue) = -(g3 + g4).
On compare aussi : (a) la meme quantite dans la reduction (g[4] + g[5] de run(), formules recopiees) a l'etat reduit de base ;
(b) les moments d'Adam du checkpoint sur les 25 logits (somme de m).
Usage : python verifier_tour68_somme_gradient_queue.py
"""

import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from verifier_tour59_delta_c_dynamique import depart, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere
import reduction_numba_tour59 as redlib
from hasard_reduction_tour59 import base_state, PHASE1

PLI = 0.0134372100660973
MSG = 10


def gradient_pleine(e, r, delta):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[3] = (1.0 - delta) / N
    poids[4] = (1.0 + delta) / N
    for p in parametres(e, r):
        p.grad = None
    j, _ = objectif_pondere(e, r, BETA, poids)
    (-j).backward()  # descente sur -j : le gradient de Adam est celui de -j
    return [p.grad.detach().clone() for p in parametres(e, r)]


def g_reduction(s, delta):
    """g[4], g[5] de run() (formules recopiees), a l'etat reduit s, en float64 numpy."""
    import math
    x, Se, Sed = s["x"], s["Se"], s["Sed"]
    B, C, N_ = 0.02, 26.0, 27.0
    z3 = x[0] - x[1]; z4 = x[2] - x[3]
    ex3 = C * math.exp(-z3); s3 = 1 / (1 + ex3)
    ex4 = C * math.exp(-z4); d4 = ex4 / (1 + ex4); s4 = 1 - d4
    u = x[4] - x[5]; w3 = math.exp(u); Z = w3 + 1 + Se
    r3 = w3 / Z; r4 = 1 / Z; logZ = math.log(Z)
    Hr = -(r3 * (u - logZ) + r4 * (-logZ) + (Sed - logZ * Se) / Z)
    a3 = (1 - delta) * s3; a4 = (1 + delta) * s4; abar = r3 * a3 + r4 * a4
    g4 = r3 * ((a3 - abar) - B * ((u - logZ) + Hr)) / N_
    g5 = r4 * ((a4 - abar) - B * ((-logZ) + Hr)) / N_
    return g4, g5


if __name__ == "__main__":
    torch.set_num_threads(1)
    (e, r), msg = depart("mur23")
    activer(e, r)
    c = torch.load(PHASE1)
    with torch.no_grad():
        for p, v in zip(parametres(e, r), c["params"]):
            p.copy_(v)
    ps = parametres(e, r)
    print(f"parametres : {[tuple(p.shape) for p in ps]} ; message {msg}")
    # indice du tenseur recepteur : celui dont la forme est (27, 27) et le second element de la liste d'Adam
    st = c["opt"]["state"]
    print(f"pas Adam du checkpoint : {float(st[0]['step'])}")
    delta1 = PLI - 1e-6
    grads = gradient_pleine(e, r, delta1)
    gr = grads[1][MSG].numpy()  # ligne 10 du recepteur
    tail = np.delete(gr, [3, 4])
    print("\nreseau complet, gradient (de -J) de la ligne 10 du recepteur, delta = pli - 1e-6, parametres du checkpoint :")
    print(f"   g3 = {gr[3]:+.6e}   g4 = {gr[4]:+.6e}   g3 + g4 = {gr[3] + gr[4]:+.6e}")
    print(f"   somme des 25 logits de queue = {tail.sum():+.6e}   (doit valoir -(g3+g4) = {-(gr[3] + gr[4]):+.6e} ; somme des 27 = {gr.sum():+.2e})")
    print(f"   queue : min {tail.min():+.3e} max {tail.max():+.3e} |.|moyen {np.abs(tail).mean():.3e}")
    m_r = st[1]["exp_avg"][MSG].numpy()
    print(f"   moments d'Adam du checkpoint : m3 = {m_r[3]:+.4e} m4 = {m_r[4]:+.4e} ; somme de m sur la queue = {np.delete(m_r, [3, 4]).sum():+.4e}")

    s = base_state()
    g4, g5 = g_reduction(s, delta1)
    print("\nreduction, etat de base (meme checkpoint), delta = pli - 1e-6 :")
    print(f"   g[4] = {g4:+.6e}   g[5] = {g5:+.6e}   g[4] + g[5] = {g4 + g5:+.6e}")
    print(f"   rapport reseau complet / reduction pour g3 + g4 : {(gr[3] + gr[4]) / (g4 + g5):.6f}")
    # meme lecture a x = 1e-8 (poids differents, meme etat de base)
    for x in (1e-8, 1e-10):
        d = PLI - x
        gx = gradient_pleine(e, r, d)[1][MSG].numpy()
        g4x, g5x = g_reduction(s, d)
        print(f"   x = {x:.0e} (etat du checkpoint, pas relaxe) : reseau complet g3 + g4 = {gx[3] + gx[4]:+.6e}, queue = {np.delete(gx, [3, 4]).sum():+.6e} ; reduction {g4x + g5x:+.6e}")
