"""Tour 58 (23/09/2026, VRAIE critique de dipankarsarkar) : test CAUSAL
du mecanisme trouve en faisant son test aligne sur les evenements.

Constat (verifier_reponse_dipankar_tour58_evenements_alignes.py) : la
pente de regression X/gap pendant une salve vaut 0,937 a delta reel et
4,75e-5 a delta=0, constante sur 15 evenements chacun. Mecanisme
propose : le gradient que la salve du recepteur injecte dans la ligne 3
de l'emetteur est proportionnel a s3(1-s3) (le facteur de dipankar) ;
a delta reel sqrt(v) de la ligne 3 (~1,9e-8) est tres au-dessus de
eps=1e-10, Adam normalise et l'invariance d'echelle EFFACE ce facteur ;
a delta=0 sqrt(v) (~9e-15) est tres en dessous d'eps, le pas devient
lr*m/eps, lineaire dans le gradient, et le facteur de saturation passe
tel quel. La saturation agirait donc A TRAVERS le plancher eps d'Adam,
pas a travers le jacobien logit -> probabilite.

Ablation : on rejoue jusqu'a ~60 pas avant une salve, puis on change
l'eps d'Adam sur la SEULE ligne 3 de l'emetteur (correction exacte du
pas apres opt.step(), les corrections de biais sont saturees a 1 a ce
stade), tout le reste identique. Predictions ecrites AVANT de lancer :
  delta=0 : pente ~ 4,75e-5 * (1e-10 + sqrt v)/(eps3 + sqrt v), donc
            ~x10 a 1e-11, ~x99 a 1e-12, ~x900 a 1e-13 (tant que le
            regime reste lineaire) ; a eps3=0 la pente redevient O(1).
  delta reel : pente qui tombe comme ~1/eps3 quand eps3 >> 1,9e-8
            (~1e-2 a 1e-6, ~1e-3 a 1e-5).
Si la pente ne suit pas eps3, le mecanisme est faux.

Mesure la pente sur les differences premieres (la salve est une
oscillation de periode 2 : robuste a la derive lente, retiree par la
mediane des differences avant la salve).

Aussi (H58.5) : au meme etat, le gradient de la ligne 3 calcule avec les
poids delta=0 puis delta reel -- la separabilite predit que seul
poids[3] entre, donc un ecart de ~delta (1,3 %), jamais x20 000.
"""

import sys
sys.path.insert(0, '.')
import copy
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

ADAM_EPS = 1e-10
LR = 0.05
DELTA_REEL = 0.013026615
CONFIGS = {
    0.0: (59519, (1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 0.0)),
    DELTA_REEL: (59929, (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5)),
}
DUREE = 800


def poids_pour(delta):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    return poids


def X_de(e3):
    autres = [j for j in range(N) if j != 10]
    return (e3[10] - e3[autres].mean()).item()


def grad_ligne3(e, r, poids):
    e.p[0].grad = None
    j, _ = objectif_pondere(e, r, BETA, poids)
    (-j).backward()
    g = e.p[0].grad[3].clone()
    e.p[0].grad = None
    for p in r.p:
        p.grad = None
    return g


def pentes_par_salve(gaps, Xs, seuil=0.004):
    """Salves = pas ou |d gap| depasse seuil, fusionnees a 50 pas. Pente
    = regression de dX (moins la mediane hors salve) sur dgap."""
    dg = [gaps[i] - gaps[i - 1] for i in range(1, len(gaps))]
    dx = [Xs[i] - Xs[i - 1] for i in range(1, len(Xs))]
    idx = [i for i, v in enumerate(dg) if abs(v) > seuil]
    salves = []
    for i in idx:
        if salves and i - salves[-1][-1] <= 50:
            salves[-1].append(i)
        else:
            salves.append([i])
    res = []
    for s in salves:
        a, b = max(0, s[0] - 5), min(len(dg), s[-1] + 6)
        hors = sorted(dx[max(0, a - 40):a])
        c = hors[len(hors) // 2] if hors else 0.0
        num = sum((dx[i] - c) * dg[i] for i in range(a, b))
        den = sum(dg[i] ** 2 for i in range(a, b))
        res.append((s[0] + 1, max(abs(dg[i]) for i in range(a, b)), num / den))
    return res


def main(delta):
    torch.set_num_threads(1)
    p_switch, eps3_liste = CONFIGS[delta]
    poids = poids_pour(delta)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    params = parametres(e, r)
    opt = torch.optim.Adam(params, lr=LR, eps=ADAM_EPS)
    for pas in range(p_switch):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()

    sauve_p = [p.detach().clone() for p in params]
    sauve_opt = copy.deepcopy(opt.state_dict())
    p_e, p_r = e.p[0], r.p[0]

    g_a = grad_ligne3(e, r, poids_pour(0.0))
    g_b = grad_ligne3(e, r, poids_pour(DELTA_REEL))
    print(f"=== delta={delta}, bascule a pas={p_switch} ===")
    print(f"H58.5, meme etat : |g_ligne3(poids reels) - g_ligne3(poids delta=0)| / |g_ligne3(delta=0)| "
          f"= {((g_b - g_a).norm() / g_a.norm()).item():.4e}   (delta={DELTA_REEL})")

    for eps3 in eps3_liste:
        with torch.no_grad():
            for p, s in zip(params, sauve_p):
                p.copy_(s)
        opt.load_state_dict(copy.deepcopy(sauve_opt))
        gaps, Xs, sv = [], [], []
        for k in range(DUREE):
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
            if eps3 != ADAM_EPS:
                with torch.no_grad():
                    st = opt.state[p_e]
                    m, rv = st["exp_avg"][3], st["exp_avg_sq"][3].sqrt()
                    p_e[3] += LR * m / (rv + ADAM_EPS) - LR * m / (rv + eps3)
            with torch.no_grad():
                gaps.append((p_r[10, 4] - p_r[10, 3]).item())
                Xs.append(X_de(p_e[3]))
                sv.append(opt.state[p_e]["exp_avg_sq"][3, 10].sqrt().item())
        s3 = torch.softmax(p_e[3], dim=0)[10].item()
        print(f"  eps_ligne3={eps3:.0e} : X debut={Xs[0]:.6f} fin={Xs[-1]:.6f}  1-s3 fin={1-s3:.3e}  "
              f"sqrt(v_e3[10]) debut={sv[0]:.3e}")
        for pas_rel, amp, pente in pentes_par_salve(gaps, Xs):
            print(f"      salve a pas={p_switch + pas_rel}  max|dgap|={amp:.4e}  pente dX/dgap={pente:+.4e}")


if __name__ == "__main__":
    main(float(sys.argv[1]))
