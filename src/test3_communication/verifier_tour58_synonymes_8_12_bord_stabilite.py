"""Tour 58 (23/09/2026) : les paires synonymes (referent 8 : messages
23/19 ; referent 12 : messages 1/21) sont-elles tenues au bord de
stabilite d'Adam comme l'orphelin (referent 5,
verifier_tour58_referent5_orphelin.py) ?

Les deux messages de chaque paire decodent vers le meme referent
(r ~ 0,9999999996) : la recompense est plate le long de la repartition
entre les deux, seule l'entropie la fixe, optimum p = 1/2 exactement.
Valeur propre du hessien de -beta H/N le long de la repartition
(u = (e_a - e_b)/sqrt 2) : h_u = beta/(2N). Gradient par coordonnee
g_a = (beta/(4N)) d, d = z_a - z_b. Au bord de stabilite,
lr h_u/(sqrt v + eps) = 38, donc :
PREDICTIONS (ecrites avant le run) :
  sqrt(v) sur chaque coordonnee de la paire = lr beta/(76 N) - eps = 4,873e-7
  (p - 1/2)_rms = d_rms/4 = (lr/19 - 2 eps/h_u)/4 = 6,58e-4
Si une paire n'est PAS au bord (convergee, v encore au-dessus du seuil),
(p-1/2)_rms << 6,58e-4 et sqrt(v) > 4,873e-7. Photos du 21/09 :
|p-1/2| = 6,5e-5 (ligne 8), 2,4e-10 (ligne 12).
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import BETA
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS

N = 27
PAIRES = {8: (23, 19), 12: (1, 21)}


def main():
    torch.set_num_threads(1)
    e, r, opt, poids = reprendre(0.013026615)
    p_e = e.p[0]
    ecarts = {i: [] for i in PAIRES}
    sv = {i: [] for i in PAIRES}
    for k in range(4000):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            S = torch.softmax(p_e, dim=1)
            V = opt.state[p_e]["exp_avg_sq"]
            for i, (a, b) in PAIRES.items():
                pa = S[i, a] / (S[i, a] + S[i, b])
                ecarts[i].append((pa - 0.5).item())
                sv[i].append(((V[i, a].sqrt() + V[i, b].sqrt()) / 2).item())
    h_u = BETA / (2 * N)
    sv_pred = LR * BETA / (76 * N) - ADAM_EPS
    ec_pred = (LR / 19 - 2 * ADAM_EPS / h_u) / 4
    with torch.no_grad():
        R = r.loi()
    for i, (a, b) in PAIRES.items():
        x = ecarts[i]
        rms = (sum(v * v for v in x) / len(x)) ** 0.5
        alt = sum(1 for t in range(1, len(x)) if x[t] * x[t - 1] < 0) / (len(x) - 1)
        s_eff = LR * h_u / (sum(sv[i]) / len(sv[i]) + ADAM_EPS)
        print(f"referent {i} (messages {a}/{b}, r[{a},{i}]={R[a, i].item():.10f}, r[{b},{i}]={R[b, i].item():.10f})")
        print(f"   (p-1/2)_rms = {rms:.4e}   predit si bord de stabilite {ec_pred:.4e}   "
              f"max |p-1/2| = {max(abs(v) for v in x):.4e}   alternance de signe = {alt:.3f}")
        print(f"   <sqrt v> = {sum(sv[i])/len(sv[i]):.4e}   predit {sv_pred:.4e}   S = lr h_u/(sqrt v + eps) = {s_eff:.2f}")
        print(f"   debut/fin : sqrt v {sv[i][0]:.4e} -> {sv[i][-1]:.4e}")


if __name__ == "__main__":
    main()
