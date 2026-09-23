"""Agent (role dipankarsarkar, tour 59 simule) : spectre EXACT du hessien
preconditionne M = lr * P^-1/2 H P^-1/2 (H = hessien de -J, 1458x1458,
P = sqrt(v)+eps d'Adam), par eigh sur la matrice complete -- pas
d'iteration de puissance, pas de demarrage a chaud.

Pour chaque pas demande :
  - les 6 plus grandes valeurs propres, et pour chacune la part (au carre,
    coordonnees blanchies) sur le gap r10[3],r10[4], sur la ligne 3 et la
    ligne 5 de l'emetteur ;
  - S_gap (quotient de Rayleigh sur la direction du gap, meme definition
    que verifier_tour58_bord_stabilite_salve.py) ;
  - le mode du gap avec et sans la ligne 3 (lignes/colonnes de la ligne 3
    retirees de M) : dS = contribution de la ligne 3 a la valeur propre du
    mode instable ;
  - le rapport X/gap lu sur le vecteur propre en coordonnees brutes
    (x = P^-1/2 w), X = e3[10] - moyenne(autres), gap = r10[4] - r10[3].

Usage : python agent_dk59_hessien_explicite.py <delta> <pas1> <pas2> ...
"""
import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import BETA
from representable_atteignable_stable import N
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS

I_R3, I_R4 = 729 + 10 * 27 + 3, 729 + 10 * 27 + 4
L3 = list(range(3 * 27, 4 * 27))
L5 = list(range(5 * 27, 6 * 27))
E310 = 3 * 27 + 10
AUTRES3 = [3 * 27 + j for j in range(27) if j != 10]


def J_plat(x, poids):
    E = x[:729].view(N, N)
    R = x[729:].view(N, N)
    s, r = torch.softmax(E, 1), torch.softmax(R, 1)
    rec = (poids.unsqueeze(1) * s * r.t()).sum()
    es = -(s * torch.log(s.clamp_min(1e-300))).sum() / N
    er = -(r * torch.log(r.clamp_min(1e-300))).sum() / N
    return rec + BETA * (es + er)


def hessien(e, r, poids):
    x = torch.cat([e.p[0].detach().flatten(), r.p[0].detach().flatten()])
    j_ref, _ = objectif_pondere(e, r, BETA, poids)
    assert abs(J_plat(x, poids).item() - j_ref.item()) < 1e-14
    return x, torch.autograd.functional.hessian(lambda z: -J_plat(z, poids), x)


def v_plat(e, r, opt):
    return torch.cat([opt.state[e.p[0]]["exp_avg_sq"].flatten(),
                      opt.state[r.p[0]]["exp_avg_sq"].flatten()])


def spectre(H, v, eps_l3=None, facteur_lr_l3=1.0):
    P = v.sqrt() + ADAM_EPS
    if eps_l3 is not None:
        P[L3] = v[L3].sqrt() + eps_l3
    P[L3] = P[L3] / facteur_lr_l3
    D = P.rsqrt()
    M = LR * D[:, None] * H * D[None, :]
    M = 0.5 * (M + M.t())
    lam, W = torch.linalg.eigh(M)
    ordre = torch.argsort(lam, descending=True)
    return M, D, lam[ordre], W[:, ordre]


def mode_gap(lam, W, i3=I_R3, i4=I_R4):
    # part ANTISYMETRIQUE (r4 - r3) : exclut le mode nul d'invariance par
    # translation de la ligne 10, qui porte aussi r10[3], r10[4] en blanchi
    parts = 0.5 * (W[i4] - W[i3]) ** 2
    k = int(torch.argmax(parts).item())
    return k, parts


def analyser(e, r, opt, poids, pas, eps_l3=None, facteur_lr_l3=1.0, verbeux=True):
    x, H = hessien(e, r, poids)
    v = v_plat(e, r, opt)
    M, D, lam, W = spectre(H, v, eps_l3, facteur_lr_l3)
    d = torch.zeros_like(x)
    d[I_R4], d[I_R3] = 1.0, -1.0
    d = d / D
    d = d / d.norm()
    s_gap = (d @ M @ d).item()
    k, parts_gap = mode_gap(lam, W)
    w = W[:, k]
    xr = D * w
    X = xr[E310] - xr[AUTRES3].mean()
    gp = xr[I_R4] - xr[I_R3]
    garder = [i for i in range(1458) if i not in set(L3)]
    lam2, W2 = torch.linalg.eigh(M[garder][:, garder])
    ig3, ig4 = garder.index(I_R3), garder.index(I_R4)
    k2 = int(torch.argmax((W2[ig4] - W2[ig3]) ** 2).item())
    print(f"pas={pas}  S_gap={s_gap:.4f}  mode gap: S={lam[k].item():.4f} (rang {k})  "
          f"sans ligne3: S={lam2[k2].item():.4f}  dS(ligne3)={lam[k].item()-lam2[k2].item():+.4f}  "
          f"part gap={parts_gap[k].item():.4f} part L3={(w[L3]**2).sum().item():.4f} "
          f"e3[10]={w[E310].item():+.4f}  X/gap brut={(X/gp).item():+.4e}")
    if verbeux:
        print("   top6: " + "  ".join(
            f"{lam[i].item():.3f}[gap {parts_gap[i].item():.3f} L3 {(W[L3, i]**2).sum().item():.3f} "
            f"L5 {(W[L5, i]**2).sum().item():.3f}]" for i in range(6)))
        print(f"   min eig={lam[-1].item():.3e}   sqrt(v) r10[4]={v[I_R4].sqrt().item():.4e}  "
              f"u=sqrt(v) e3[10]={v[E310].sqrt().item():.4e}  H(r4,r4)={H[I_R4,I_R4].item():.4e} "
              f"H(r4,r3)={H[I_R4,I_R3].item():.4e} H(e310,r4)={H[E310,I_R4].item():+.4e} "
              f"H(e310,e310)={H[E310,E310].item():.4e}")
    sys.stdout.flush()
    return lam[k].item(), lam2[k2].item(), s_gap


def pas_adam(e, r, opt, poids):
    j, _ = objectif_pondere(e, r, BETA, poids)
    opt.zero_grad()
    (-j).backward()
    opt.step()


if __name__ == "__main__":
    torch.set_num_threads(1)
    delta = float(sys.argv[1])
    cibles = sorted(int(a) for a in sys.argv[2:])
    e, r, opt, poids = reprendre(delta)
    pas = 54000
    for c in cibles:
        while pas < c:
            pas_adam(e, r, opt, poids)
            pas += 1
        analyser(e, r, opt, poids, pas)
