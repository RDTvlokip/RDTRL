"""Agent (role dipankarsarkar, tour 59 simule) : QUAND la ligne 3 avance-t-elle
la salve ? Gel de la ligne 3 (pas d'Adam annule apres opt.step()) sur une
fenetre [a, b) seulement. Predictions : D:/tmp/agent_dk59_predictions_fenetre_gel.txt.

Usage : python agent_dk59_fenetre_gel.py <delta> <depart> <duree> a1:b1 a2:b2 ...
(etiquettes de pas = convention de la trace du tour 58 ; b=0 : jusqu'a la fin)
"""
import sys
sys.path.insert(0, '.')
import copy
import torch

from replay_mur23_referent3 import BETA
from representable_atteignable_stable import parametres
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS


def main(delta, depart, duree, fenetres):
    torch.set_num_threads(1)
    e, r, opt, poids = reprendre(delta)
    params = parametres(e, r)
    for _ in range(depart - 54000):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    sauve_p = [p.detach().clone() for p in params]
    sauve_opt = copy.deepcopy(opt.state_dict())
    p_e, p_r = e.p[0], r.p[0]
    for a, b in [(None, None)] + fenetres:
        with torch.no_grad():
            for p, s in zip(params, sauve_p):
                p.copy_(s)
        opt.load_state_dict(copy.deepcopy(sauve_opt))
        g_prec = (p_r[10, 4] - p_r[10, 3]).item()
        debut, pic, amp = None, None, 0.0
        for i in range(duree):
            etiquette = depart + i
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
            if a is not None and a <= etiquette and (b == 0 or etiquette < b):
                with torch.no_grad():
                    st = opt.state[p_e]
                    m, rv = st["exp_avg"][3], st["exp_avg_sq"][3].sqrt()
                    p_e[3] += LR * m / (rv + ADAM_EPS)
            g = (p_r[10, 4] - p_r[10, 3]).item()
            dg, g_prec = g - g_prec, g
            if debut is None and abs(dg) > 0.004:
                debut = etiquette
            if debut is not None and etiquette <= debut + 40 and abs(dg) > amp:
                amp, pic = abs(dg), etiquette
        nom = "base" if a is None else f"gel [{a},{b if b else 'fin'})"
        print(f"  {nom:22s} debut salve={debut}  pic={pic}  max|dgap|={amp:.4e}")
        sys.stdout.flush()


if __name__ == "__main__":
    delta, depart, duree = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    fen = [tuple(int(x) for x in a.split(":")) for a in sys.argv[4:]]
    main(delta, depart, duree, fen)
