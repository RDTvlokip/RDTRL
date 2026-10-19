"""Tour 77 (vraie critique de dipankarsarkar, 22/10/2026) : le bloc 2 x 2 (gap, ligne 3 de l'emetteur) du reseau complet a eps 1e-9, compare a la reduction.

Il dit : mes cinq rapports normalises sont deux nombres libres (d_gap/S_max et d_s/S_max), le reste en decoule ; l'echelle a rate (38,85 contre 38,02) ; et la reduction a deja pris un engagement a eps 1e-9
(partenaire/S_max 0,0441, poids 0,0350, contre 0,0585 et 0,0485 a 1e-10) que le reseau complet, lance a 1e-10 seulement, n'a pas confirme.

Modes :
  arithmetique
  reseau <eps> <pas_debut> <pas_fin>   : reseau complet reconstruit a cet eps (construire_mur23(adam_eps=eps) puis les pas ponderes au delta reel 0,013026615 ; a eps 1e-10 on reprend le checkpoint a 54000),
                                          bloc (g, s) a chaque pas (g = direction du gap, s = projection de M g sur la ligne 3 de l'emetteur, normalisee), niveau haut, partenaire, poids du sender,
                                          fermeture ; une ligne FRANCHISSEMENT quand le niveau haut du bloc passe 38,0 par le bas.
"""

import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")


def mode_arithmetique():
    dg, ds, c = 0.9544, 0.10405, 0.2022
    somme = (dg + ds) / 2 + math.sqrt(((dg - ds) / 2) ** 2 + c ** 2)
    r = (1 - dg) / (1 - ds)
    print(f"(d_gap + d_s)/2 + sqrt(((d_gap - d_s)/2)^2 + c^2) = {somme:.5f} ; partenaire = d_gap + d_s - 1 = {dg + ds - 1:.5f} ; poids du sender = r/(1+r) = {r / (1 + r):.5f} (r = {r:.5f}) ; couplage = sqrt((1-d_gap)(1-d_s)) = {math.sqrt((1 - dg) * (1 - ds)):.4f}")
    print(f"eps 1e-10 / eps 1e-9 : partenaire {0.0585 / 0.0441:.3f} ; poids {0.0485 / 0.0350:.3f}")


def mode_reseau(eps, debut, fin, delta=0.013026615):
    import torch
    from replay_mur23_referent3 import BETA, construire_mur23
    from representable_atteignable_stable import activer, parametres
    from verifier_prior_asymetrique import objectif_pondere
    from sauver_checkpoints_mur23_tour58 import reprendre, poids_pour, LR
    torch.set_num_threads(1)
    t0 = time.time()

    def pas_adam(e, r, opt, poids):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()

    def dot(a, b):
        return sum((x * y).sum() for x, y in zip(a, b))

    def bloc(e, r, opt, poids):
        params = parametres(e, r)
        j, _ = objectif_pondere(e, r, BETA, poids)
        grads = torch.autograd.grad(-j, params, create_graph=True)
        D = [opt.state[p]["exp_avg_sq"].sqrt() + eps for p in params]
        dmh = [d.rsqrt() for d in D]

        def M(w):
            u = [a * b for a, b in zip(dmh, w)]
            hv = torch.autograd.grad(grads, params, grad_outputs=u, retain_graph=True)
            return [a * b for a, b in zip(dmh, hv)]

        d = [torch.zeros_like(p) for p in params]
        d[1][10, 4], d[1][10, 3] = 1.0, -1.0
        d = [a / b for a, b in zip(d, dmh)]
        nd = torch.sqrt(dot(d, d))
        g = [a / nd for a in d]
        Mg = M(g)
        d_gap = dot(g, Mg).item()
        s = [torch.zeros_like(p) for p in params]
        s[0][3, :] = Mg[0][3, :]
        c = torch.sqrt(dot(s, s)).item()
        s = [a / c for a in s]
        Ms = M(s)
        d_s = dot(s, Ms).item()
        c2 = dot(s, Mg).item()
        res = [a - d_gap * x - c2 * y for a, x, y in zip(Mg, g, s)]
        ferm = (torch.sqrt(dot(res, res)) / torch.sqrt(dot(Mg, Mg))).item()
        return LR * d_gap, LR * d_s, LR * c2, ferm

    if eps == 1e-10 and debut >= 54000 and delta == 0.013026615:
        e, r, opt, poids = reprendre(delta)
        depart = 54000
    else:
        poids = poids_pour(delta)
        e, r = construire_mur23(adam_eps=eps)
        activer(e, r)
        opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=eps)
        depart = 0
    for _ in range(debut - depart):
        pas_adam(e, r, opt, poids)
    print(f"reseau complet, eps = {eps:.0e}, delta = {delta}, pas {debut} a {fin} (construction + avance : {time.time() - t0:.0f} s)", flush=True)
    print("  pas     niveau   d_gap    d_s     couplage  partenaire  part./niveau  poids sender  d_s/niveau  d_gap/niveau  fermeture")
    prec = None
    for pas in range(debut, fin):
        d_gap, d_s, c, ferm = bloc(e, r, opt, poids)
        B = np.array([[d_gap, c], [c, d_s]])
        ev, vec = np.linalg.eigh(B)
        top, part = ev[1], ev[0]
        wv = vec[:, 1]
        ligne = f"  {pas}  {top:7.3f}  {d_gap:7.3f} {d_s:7.3f} {c:8.3f}  {part:8.3f}   {part / top:.5f}      {wv[1] ** 2:.5f}       {d_s / top:.5f}    {d_gap / top:.5f}     {ferm:.4f}"
        if prec is not None and prec < 38.0 <= top:
            ligne += "   <== FRANCHISSEMENT 38,0"
        if (pas - debut) % 10 == 0 or "FRANCH" in ligne:
            print(ligne, flush=True)
        prec = top
        pas_adam(e, r, opt, poids)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "arithmetique":
        mode_arithmetique()
    else:
        mode_reseau(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]) if len(sys.argv) > 5 else 0.013026615)
