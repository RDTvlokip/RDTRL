"""Agent (role dipankarsarkar, tour 59 simule) : test precommis (predictions
dans D:/tmp/agent_dk59_predictions_k_ligne3.txt, ecrites avant ce run).

A partir du pas `bascule`, le pas d'Adam de la SEULE ligne 3 de l'emetteur
est multiplie par k (correction exacte apres opt.step() : corrections de
biais = 1 a ce stade). k=1 : trajectoire de base ; k=0 : ligne 3 gelee.
Mesure : debut de la salve suivante (premier |dgap| > 0,004, meme
convention que verifier_reponse_dipankar_tour58_ablation_eps_ligne3.py),
son pic (max |dgap|), et la pente dX/dgap par la meme methode que ce
script (pentes_par_salve).

Usage : python agent_dk59_experience_k_ligne3.py <delta> <bascule> <duree> <k1> <k2> ...
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


def main(delta, bascule, duree, ks):
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
    print(f"=== delta={delta}, bascule a pas={bascule} (etat apres {bascule} pas) ===")
    for k in ks:
        with torch.no_grad():
            for p, s in zip(params, sauve_p):
                p.copy_(s)
        opt.load_state_dict(copy.deepcopy(sauve_opt))
        gaps, Xs = [], []
        for i in range(duree):
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
            if k != 1.0:
                with torch.no_grad():
                    st = opt.state[p_e]
                    m, rv = st["exp_avg"][3], st["exp_avg_sq"][3].sqrt()
                    p_e[3] -= (k - 1.0) * LR * m / (rv + ADAM_EPS)
            with torch.no_grad():
                gaps.append((p_r[10, 4] - p_r[10, 3]).item())
                Xs.append((p_e[3, 10] - p_e[3, AUTRES].mean()).item())
        dg = [gaps[i] - gaps[i - 1] for i in range(1, len(gaps))]
        # dg[i] = variation au pas numero bascule + i + 1 (convention trace/EoS)
        print(f"  k={k}")
        for pas_rel, amp, pente in pentes_par_salve(gaps, Xs):
            # pas_rel = s[0]+1 : meme etiquette que l'ablation du tour 58
            fen = range(max(0, pas_rel - 1 - 5), min(len(dg), pas_rel - 1 + 60))
            ipic = max(fen, key=lambda i: abs(dg[i]))
            print(f"      salve : debut (|dgap|>0,004) pas={bascule + pas_rel}  "
                  f"pic |dgap| pas={bascule + ipic + 1}  max|dgap|={amp:.4e}  pente dX/dgap={pente:+.4e}")
        sys.stdout.flush()


if __name__ == "__main__":
    delta, bascule, duree = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    main(delta, bascule, duree, [float(a) for a in sys.argv[4:]])
