"""Tour 58 (23/09/2026) : la salve de periode 2 du recepteur est-elle un
bord de stabilite d'Adam au sens de Cohen et al. 2022 (arXiv:2207.14484,
"Adaptive Gradient Methods at the Edge of Stability") ?

Leur abstract (verifie a la source) : "For Adam with step size eta and
beta1 = 0.9, this stability threshold is 38/eta" -- sur la valeur propre
maximale du hessien preconditionne P^-1 H, P = diag(sqrt(v) + eps).
Donc, en unite sans dimension : S = lr * lambda_max(P^-1 H) contre 38.

Prediction (ecrite avant le run) : entre deux salves, v du recepteur
decroit (plancher de v), S monte ; la salve demarre quand S franchit ~38
sur la direction du gap (r10[4]-r10[3]), l'oscillation fait remonter v,
S retombe sous 38, la salve s'eteint. Si S reste loin de 38 au debut de
la salve (disons < 20 ou > 80), la salve n'est pas un bord de stabilite
de ce type et le mecanisme "plancher de v" doit etre decrit autrement.

Deux mesures a chaque pas :
  - S_gap : quotient de Rayleigh preconditionne sur la direction du gap
    dans la ligne 10 du recepteur (coordonnees r10[3], r10[4] seules) ;
  - S_max : lambda_max global de P^-1/2 H P^-1/2 (iteration de
    puissance sur produits hessien-vecteur, tous les parametres), et
    la part de son vecteur propre portee par r10[3], r10[4].

Resultat (23/09) : CONFIRME. delta=0 : S_gap monte de 32,4 (apres une
salve) a 38 vers pas~59040 pendant que sqrt(v r10[4]) decroit de 5,7e-7
a 4,8e-7 ; l'oscillation de periode 2 croit alors de 1e-9 a 4e-2,
S_gap culmine a 39,5, le vecteur propre dominant est a 100,0 % sur le
gap ; la salve remonte v, S_gap retombe a 34, la salve s'eteint. delta
reel : meme scenario, mais le mode instable est COUPLE -- 95,1 % gap,
4,9 % ligne 3 de l'emetteur (e3[10]=0,162) -- et c'est lui (S_max 39,7)
qui franchit 38, pas le gap seul (S_gap plafonne a 37,9). Entre les
salves, une autre direction reste en permanence a S~38 : la LIGNE 5 de
l'emetteur (referent 5, celui a H~ln27).
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import BETA
from representable_atteignable_stable import parametres
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS

FENETRES = {0.0: (58680, 59140), 0.013026615: (59100, 59560)}
N_PUISSANCE = 40


def pas_adam(e, r, opt, poids):
    j, _ = objectif_pondere(e, r, BETA, poids)
    opt.zero_grad()
    (-j).backward()
    opt.step()


def mesurer(e, r, opt, poids, w_init):
    params = parametres(e, r)
    j, _ = objectif_pondere(e, r, BETA, poids)
    grads = torch.autograd.grad(-j, params, create_graph=True)
    D = [opt.state[p]["exp_avg_sq"].sqrt() + ADAM_EPS for p in params]
    dmh = [d.rsqrt() for d in D]

    def M(w):  # P^-1/2 H P^-1/2 w
        u = [a * b for a, b in zip(dmh, w)]
        hv = torch.autograd.grad(grads, params, grad_outputs=u, retain_graph=True)
        return [a * b for a, b in zip(dmh, hv)]

    # direction du gap dans la metrique preconditionnee
    d = [torch.zeros_like(p) for p in params]
    ir = 1  # parametres(e, r) : [e.p[0], r.p[0]]
    d[ir][10, 4], d[ir][10, 3] = 1.0, -1.0
    d = [a / b for a, b in zip(d, dmh)]  # coordonnees ou P^-1/2 s'applique
    nd = torch.sqrt(sum((a * a).sum() for a in d))
    d = [a / nd for a in d]
    s_gap = sum((a * b).sum() for a, b in zip(d, M(d))).item()

    w = w_init if w_init is not None else [torch.randn_like(p) for p in params]
    lam = 0.0
    for _ in range(N_PUISSANCE):
        n = torch.sqrt(sum((a * a).sum() for a in w))
        w = [a / n for a in w]
        mw = M(w)
        lam = sum((a * b).sum() for a, b in zip(w, mw)).item()
        w = [x.detach() for x in mw]
    n = torch.sqrt(sum((a * a).sum() for a in w))
    w = [a / n for a in w]
    part_gap = (w[ir][10, 3] ** 2 + w[ir][10, 4] ** 2).item()
    return LR * s_gap, LR * lam, part_gap, w


def localiser(w):
    """Ou vit le vecteur propre dominant : part sur la ligne 3 de
    l'emetteur, sur l'emetteur entier, sur la ligne 10 du recepteur, et
    ses 4 plus grosses coordonnees ('e(ligne, colonne)' / 'r(...)')."""
    we, wr = w[0], w[1]
    flat = torch.cat([we.flatten(), wr.flatten()])
    top = flat.abs().topk(4)
    n = we.numel()
    noms = [("e" if i < n else "r") + str(divmod(i % n, we.shape[1])) for i in top.indices.tolist()]
    return ((we[3] ** 2).sum().item(), (we ** 2).sum().item(), (wr[10] ** 2).sum().item(),
            list(zip(noms, [round(v, 3) for v in top.values.tolist()])))


def main(delta):
    torch.manual_seed(0)
    debut, fin = FENETRES[delta]
    e, r, opt, poids = reprendre(delta)
    for _ in range(debut - 54000):
        pas_adam(e, r, opt, poids)
    w = None
    print(f"=== delta={delta} : S = lr*lambda(P^-1 H), seuil Cohen et al. = 38 ===")
    gap_prec = (r.p[0][10, 4] - r.p[0][10, 3]).item()
    for pas in range(debut, fin):
        s_gap, s_max, part, w = mesurer(e, r, opt, poids, w)
        pas_adam(e, r, opt, poids)
        gap = (r.p[0][10, 4] - r.p[0][10, 3]).item()
        dgap = gap - gap_prec
        gap_prec = gap
        if (pas - debut) % 20 == 0 or abs(dgap) > 1e-4:
            pe3, pe, pr10, top = localiser(w)
            print(f"  pas={pas}  S_gap={s_gap:8.3f}  S_max={s_max:8.3f}  part gap du vp={part:.3f}  "
                  f"part e3={pe3:.4f} part emetteur={pe:.4f}  "
                  f"sqrt(v r10[4])={opt.state[r.p[0]]['exp_avg_sq'][10, 4].sqrt().item():.4e}  dgap={dgap:+.3e}  "
                  f"top={top}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main(float(sys.argv[1]))
