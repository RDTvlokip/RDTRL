"""Le 0,5 sur le mur 23/24 (referents 3/4, message 10) est-il un point fixe
dynamique, ou le posterior bayesien d'un message porte par deux referents
en collision, comme le pose dipankarsarkar au tour 48 ?

Son test decisif : casser le PRIOR plutot que le PUSH. Si 0,5 est le
posterior d'un message collide, entrainer avec le referent 4 vu deux fois
plus souvent que le referent 3 (tout le reste egal) doit faire migrer
R[10,4] vers 0,667 = 2/3. Si 0,5 est un point fixe dynamique independant
des comptes, R[10,4] doit rester a 0,5.

Deux verifications bon marche demandees en meme temps :
  - R[10,3] doit aussi valoir environ 0,5 (symetrique).
  - l'entropie du recepteur sur le message 10 doit valoir ln(2) = 0,693147
    si toute la masse est bien partagee entre UNIQUEMENT les referents 3 et 4.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres, N


def objectif_pondere(emetteur, recepteur, beta, poids, reponderer_entropie_s=False):
    """Identique a objectif(), sauf que la recompense pondere chaque
    referent par poids[i] au lieu de 1/N uniforme.

    Par defaut (reponderer_entropie_s=False, le comportement d'origine),
    l'entropie de l'emetteur reste une moyenne NON ponderee -- c'est
    exactement le point que dipankarsarkar releve au tour 49 : la
    recompense du referent 3 baisse avec son poids mais la pression
    d'entropie qui le retient ne bouge pas, un ratio recompense/entropie
    desequilibre plutot qu'un vrai signal d'eviction.

    reponderer_entropie_s=True applique son correctif d'une ligne :
    chaque ligne d'emetteur est ponderee par (N*poids[i]), de sorte que
    chaque referent factorise poids[i]*(recompense_i + beta*H_i) et que
    son propre optimum ne depende plus de delta -- seul le posterior du
    recepteur doit alors bouger."""
    s, r = emetteur.loi(), recepteur.loi()
    recompense = (poids.unsqueeze(1) * s * r.t()).sum()
    if reponderer_entropie_s:
        entropie_s = -((N * poids).unsqueeze(1) * s * torch.log(s.clamp_min(1e-300))).sum() / N
    else:
        entropie_s = -(s * torch.log(s.clamp_min(1e-300))).sum() / N
    entropie_r = -(r * torch.log(r.clamp_min(1e-300))).sum() / N
    return recompense + beta * (entropie_s + entropie_r), recompense


def continuer_sous_prior(e, r, beta, pas, lr, adam_eps, poids, reponderer_entropie_s=False):
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=lr, eps=adam_eps)
    for _ in range(pas):
        j, _ = objectif_pondere(e, r, beta, poids, reponderer_entropie_s=reponderer_entropie_s)
        opt.zero_grad()
        (-j).backward()
        opt.step()


def etat(e, r, msg=10, refs=(3, 4)):
    """Retourne aussi s[3,msg]/s[4,msg] desormais -- demande explicite du
    tour 49 (« worth printing s[3,10] and s[4,10] in etat() »)."""
    with torch.no_grad():
        s_, r_ = e.loi(), r.loi()
        R = {i: (s_[i, msg] * r_[msg, i]).item() for i in refs}
        S = {i: s_[i, msg].item() for i in refs}
        ligne = r_[msg, :]
        H_pleine = -(ligne * torch.log(ligne.clamp_min(1e-300))).sum().item()
        masse_34 = sum(ligne[i].item() for i in refs)
        p3, p4 = ligne[refs[0]].item(), ligne[refs[1]].item()
        H_binaire = -(p3 * torch.log(torch.tensor(max(p3, 1e-300))) +
                      p4 * torch.log(torch.tensor(max(p4, 1e-300)))).item()
        return R, H_pleine, masse_34, H_binaire, S


if __name__ == "__main__":
    torch.set_printoptions(precision=15)
    ADAM_EPS = 1e-10
    PAS_SUITE_ADD = 40000

    print("=== etat de depart (mur 23/24, adam_eps=1e-10, comme publie) ===")
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    R0, H0, m0, Hb0, S0 = etat(e, r)
    print(f"  R[10,4]={R0[4]:.6f}  R[10,3]={R0[3]:.6f}  "
          f"H(recepteur, msg10, 27 voies)={H0:.6f}  masse(3+4)={m0:.9f}  "
          f"H(binaire 3/4)={Hb0:.6f}   ln2={torch.log(torch.tensor(2.0)).item():.6f}")

    print("\n=== continuation sous prior SYMETRIQUE (poids=1 partout), controle ===")
    poids_uniformes = torch.full((N,), 1.0 / N, dtype=torch.float64)
    e_ctrl, r_ctrl = construire_mur23(adam_eps=ADAM_EPS)
    continuer_sous_prior(e_ctrl, r_ctrl, BETA, PAS_SUITE_ADD, 0.05, ADAM_EPS, poids_uniformes)
    Rc, Hc, mc, Hbc, Sc = etat(e_ctrl, r_ctrl)
    print(f"  R[10,4]={Rc[4]:.6f}  R[10,3]={Rc[3]:.6f}  "
          f"H(27 voies)={Hc:.6f}  masse(3+4)={mc:.9f}  H(binaire)={Hbc:.6f}")

    print("\n=== continuation sous prior ASYMETRIQUE (referent 4 pese 2x referent 3) ===")
    poids_asym = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids_asym[4] = 2.0 / N
    e_asym, r_asym = construire_mur23(adam_eps=ADAM_EPS)
    continuer_sous_prior(e_asym, r_asym, BETA, PAS_SUITE_ADD, 0.05, ADAM_EPS, poids_asym)
    Ra, Ha, ma, Hba, Sa = etat(e_asym, r_asym)
    print(f"  R[10,4]={Ra[4]:.6f}  R[10,3]={Ra[3]:.6f}  "
          f"H(27 voies)={Ha:.6f}  masse(3+4)={ma:.9f}  H(binaire)={Hba:.6f}")
    print(f"\n  prediction 'posterior bayesien' : R[10,4] -> 2/3 = {2/3:.6f}")
    print(f"  prediction 'point fixe dynamique' : R[10,4] reste a 0.5")
