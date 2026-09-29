"""Tour 59 (27/09/2026) : delta_c DYNAMIQUE (Adam, lr 0,05, eps 1e-10,
beta 0,02) compare au pli idealise (0,0134372 pour C=26 concurrents).

Deux collisions :
  mur23 : graine 77777 k=3, message 10, referents 3 (sous-pondere) / 4
          (construire_mur23, egalite poussee) ;
  c814  : graine 12345 k=3, message 8, referents 6 / 14 (collision
          spontanee, replay 30 000 pas, Adam standard). On sous-pondere
          l'un ou l'autre (sens a=6 ou a=14).

Usage : python verifier_tour59_delta_c_dynamique.py <cas> <a> <b> <delta> <pas> [trace]
Sort une ligne : cas a b delta pas R[msg,b] 1-s[a,msg] pas_bascule
(pas_bascule = premier pas ou R[msg,b] > 0,9, -1 sinon).
Avec trace : ecrit 1-s[a,msg], R[msg,b] et r[msg,b] a chaque pas dans D:/tmp.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, replay, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

ADAM_EPS = 1e-10
LR = 0.05


def depart(cas):
    """Etat de depart, mis en cache dans D:/tmp (logits seulement ; Adam
    repart a zero comme dans la bissection du tour 50)."""
    import os
    # c7771 : collision spontanee de la graine 77777 avec k=1 (message 14, referents 5/20), 3e code
    msg = {"mur23": 10, "c814": 8, "c7771": 14}[cas]
    chemin = f"D:/tmp/rdtrl_tour59_depart_{cas}.pt"
    if cas == "mur23":
        e, r = replay(77777, 3, 10)
    elif cas == "c814":
        e, r = replay(12345, 3, 10)
    else:
        e, r = replay(77777, 1, 10)
    activer(e, r)
    params = parametres(e, r)
    if os.path.exists(chemin):
        with torch.no_grad():
            for p, v in zip(params, torch.load(chemin)):
                p.copy_(v)
        return (e, r), msg
    if cas == "mur23":
        e, r = construire_mur23(adam_eps=ADAM_EPS)
    elif cas == "c814":
        e, r = replay(12345, 3, 30000)
    else:
        e, r = replay(77777, 1, 30000)
    activer(e, r)
    torch.save([p.detach().clone() for p in parametres(e, r)], chemin)
    return (e, r), msg


def courir(cas, a, b, delta, pas, trace=False, eps=ADAM_EPS, chauffe=0, chauffe_eps=0, lr=LR):
    """chauffe > 0 : d'abord `chauffe` pas a delta = pli - 1e-6 et eps 1e-10
    (l'etat rejoint la branche de collision, ou les gradients ne sont plus
    minuscules), PUIS bascule vers delta et eps, en gardant l'etat d'Adam.
    Sans chauffe, un eps >= 1e-7 gele la ligne a son depart (1-s ~ 3e-10).
    chauffe_eps > 0 : entre les deux, `chauffe_eps` pas supplementaires a
    pli - 1e-6 avec le NOUVEL eps, pour separer le choc du changement d'eps
    du changement de delta."""
    import os
    torch.set_num_threads(1)
    (e, r), msg = depart(cas)
    activer(e, r)
    params = parametres(e, r)
    opt = torch.optim.Adam(params, lr=lr, eps=ADAM_EPS if chauffe else eps)
    if chauffe:
        # etat chauffe mis en cache : tous les delta d'une meme serie partent du
        # MEME point (parametres + moments d'Adam), ce qui retire la dependance
        # au chemin. Creer le cache par un appel a pas=0 AVANT de lancer en parallele.
        # lr different de 0,05 : le taux est dans le nom (le cache de base n'en a pas).
        tag_lr = "" if lr == LR else f"_lr{lr}"
        ck = f"D:/tmp/rdtrl_tour59_chaud_{cas}_{a}_{b}_eps{eps:.0e}_{chauffe}_{chauffe_eps}{tag_lr}.pt"
        if os.path.exists(ck):
            c = torch.load(ck)
            with torch.no_grad():
                for p, v in zip(params, c["params"]):
                    p.copy_(v)
            opt.load_state_dict(c["opt"])
            opt.param_groups[0]["eps"] = eps
        else:
            pc = torch.full((N,), 1.0 / N, dtype=torch.float64)
            pc[a], pc[b] = (1.0 - (0.0134372100660973 - 1e-6)) / N, (1.0 + (0.0134372100660973 - 1e-6)) / N
            for _ in range(chauffe):
                j, _ = objectif_pondere(e, r, BETA, pc)
                opt.zero_grad()
                (-j).backward()
                opt.step()
            opt.param_groups[0]["eps"] = eps
            for _ in range(chauffe_eps):
                j, _ = objectif_pondere(e, r, BETA, pc)
                opt.zero_grad()
                (-j).backward()
                opt.step()
            torch.save({"params": [p.detach().clone() for p in params], "opt": opt.state_dict()}, ck)
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[a] = (1.0 - delta) / N
    poids[b] = (1.0 + delta) / N
    bascule, lignes = -1, []
    for t in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if bascule < 0 or trace:
            with torch.no_grad():
                S, R = e.loi(), r.loi()
                Rb = (S[b, msg] * R[msg, b]).item()
                if bascule < 0 and Rb > 0.9:
                    bascule = t
                if trace:
                    # colonnes 4-7 : sqrt(exp_avg_sq) d'Adam sur e[a,msg], e[b,msg], r[msg,a], r[msg,b]
                    ve, vr = opt.state[e.p[0]]["exp_avg_sq"], opt.state[r.p[0]]["exp_avg_sq"]
                    sv = (ve[a, msg].sqrt().item(), ve[b, msg].sqrt().item(), vr[msg, a].sqrt().item(), vr[msg, b].sqrt().item())
                    lignes.append(f"{t} {1 - S[a, msg].item():.10e} {Rb:.10f} {R[msg, b].item():.10f} "
                                  + " ".join(f"{x:.6e}" for x in sv))
            if bascule >= 0 and not trace:
                break
    with torch.no_grad():
        S, R = e.loi(), r.loi()
        Rb = (S[b, msg] * R[msg, b]).item()
        d = 1 - S[a, msg].item()
    if trace:
        suffixe = "" if (eps == ADAM_EPS and not chauffe) else f"_eps{eps:.0e}_chauffe{chauffe}_{chauffe_eps}"
        with open(f"D:/tmp/rdtrl_tour59_trace_{cas}_a{a}_b{b}_delta{delta}_pas{pas}{suffixe}.txt", "w") as f:
            f.write("\n".join(lignes))
    print(f"{cas} a={a} b={b} delta={delta!r} eps={eps:.0e} pas={pas} R_b={Rb:.6f} 1-s_a={d:.6e} bascule={bascule}", flush=True)


if __name__ == "__main__":
    # arguments optionnels apres les cinq positionnels : "trace" et/ou "eps=<valeur>"
    # (eps d'Adam pendant la phase ponderee ; l'etat de depart reste celui a 1e-10)
    cas, a, b, delta, pas = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    opts = sys.argv[6:]
    eps = next((float(o[4:]) for o in opts if o.startswith("eps=")), ADAM_EPS)
    chauffe = next((int(o[8:]) for o in opts if o.startswith("chauffe=")), 0)
    chauffe_eps = next((int(o[12:]) for o in opts if o.startswith("chauffe_eps=")), 0)
    lr = next((float(o[3:]) for o in opts if o.startswith("lr=")), LR)
    courir(cas, a, b, delta, pas, trace="trace" in opts, eps=eps, chauffe=chauffe, chauffe_eps=chauffe_eps, lr=lr)
