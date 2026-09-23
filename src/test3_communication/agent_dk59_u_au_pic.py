"""Agent (role dipankarsarkar, tour 59 simule) : l'ablation d'eps du tour 58
refaite (meme bascule 59929, memes eps3, meme correction apres opt.step(),
meme methode de pente), en enregistrant en plus u = sqrt(v) de e3[10] au
PIC de la salve de chaque run -- pas seulement au debut.

Question : le deficit de 5-7 % de la pente sous la formule aux eps3 = 1e-6
et 1e-5 vient-il de "la salve du recepteur grossit" (lecture du tour 58)
ou de ce que la salve arrive ~95 pas plus tard, quand v de la ligne 3 a
decru de 0,999^95 ?

Usage : python agent_dk59_u_au_pic.py <delta> <bascule> <duree> <eps3_1> <eps3_2> ...
"""
import sys
sys.path.insert(0, '.')
import copy
import torch

from replay_mur23_referent3 import BETA
from representable_atteignable_stable import N, parametres
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS
from verifier_reponse_dipankar_tour58_ablation_eps_ligne3 import pentes_par_salve

AUTRES = [j for j in range(N) if j != 10]


def formule(u, eps):
    return 0.5 * (u / (u + eps) + (u / 26) / (u / 26 + eps))


def main(delta, bascule, duree, eps_liste):
    torch.set_num_threads(1)
    e, r, opt, poids = reprendre(delta)
    params = parametres(e, r)
    for _ in range(bascule - 54000):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    sauve_p = [p.detach().clone() for p in params]
    sauve_opt = copy.deepcopy(opt.state_dict())
    p_e, p_r = e.p[0], r.p[0]
    print(f"=== delta={delta}, bascule a pas={bascule} ===")
    for eps3 in eps_liste:
        with torch.no_grad():
            for p, s in zip(params, sauve_p):
                p.copy_(s)
        opt.load_state_dict(copy.deepcopy(sauve_opt))
        gaps, Xs, us, vr = [], [], [], []
        for i in range(duree):
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
            with torch.no_grad():
                st = opt.state[p_e]
                if eps3 != ADAM_EPS:
                    m, rv = st["exp_avg"][3], st["exp_avg_sq"][3].sqrt()
                    p_e[3] += LR * m / (rv + ADAM_EPS) - LR * m / (rv + eps3)
                gaps.append((p_r[10, 4] - p_r[10, 3]).item())
                Xs.append((p_e[3, 10] - p_e[3, AUTRES].mean()).item())
                us.append(st["exp_avg_sq"][3, 10].sqrt().item())
                vr.append(opt.state[p_r]["exp_avg_sq"][10, 4].sqrt().item())
        dg = [gaps[i] - gaps[i - 1] for i in range(1, len(gaps))]
        for pas_rel, amp, pente in pentes_par_salve(gaps, Xs)[:1]:
            fen = range(max(0, pas_rel - 1 - 5), min(len(dg), pas_rel - 1 + 60))
            ipic = max(fen, key=lambda i: abs(dg[i]))
            u_deb, u_pic = us[pas_rel - 1], us[ipic + 1]
            print(f"  eps3={eps3:.0e}  debut={bascule + pas_rel} pic={bascule + ipic + 1}  "
                  f"pente={pente:.4e}  u(bascule)={us[0]:.4e} u(debut salve)={u_deb:.4e} "
                  f"u(pic+1)={u_pic:.4e}  sqrt(v r10[4]) debut salve={vr[pas_rel - 1]:.4e}  "
                  f"formule(u=1,88e-8)={formule(1.88e-8, eps3):.4e} rapport={pente/formule(1.88e-8, eps3):.4f}  "
                  f"formule(u pic)={formule(u_pic, eps3):.4e} rapport={pente/formule(u_pic, eps3):.4f}  "
                  f"max|dgap|={amp:.4e}")
        sys.stdout.flush()


if __name__ == "__main__":
    delta, bascule, duree = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    main(delta, bascule, duree, [float(a) for a in sys.argv[4:]])
