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
    msg = {"mur23": 10, "c814": 8}[cas]
    chemin = f"D:/tmp/rdtrl_tour59_depart_{cas}.pt"
    if cas == "mur23":
        e, r = replay(77777, 3, 10)
    else:
        e, r = replay(12345, 3, 10)
    activer(e, r)
    params = parametres(e, r)
    if os.path.exists(chemin):
        with torch.no_grad():
            for p, v in zip(params, torch.load(chemin)):
                p.copy_(v)
        return (e, r), msg
    e, r = construire_mur23(adam_eps=ADAM_EPS) if cas == "mur23" else replay(12345, 3, 30000)
    activer(e, r)
    torch.save([p.detach().clone() for p in parametres(e, r)], chemin)
    return (e, r), msg


def courir(cas, a, b, delta, pas, trace=False, eps=ADAM_EPS):
    torch.set_num_threads(1)
    (e, r), msg = depart(cas)
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[a] = (1.0 - delta) / N
    poids[b] = (1.0 + delta) / N
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=eps)
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
                    lignes.append(f"{t} {1 - S[a, msg].item():.10e} {Rb:.10f} {R[msg, b].item():.10f}")
            if bascule >= 0 and not trace:
                break
    with torch.no_grad():
        S, R = e.loi(), r.loi()
        Rb = (S[b, msg] * R[msg, b]).item()
        d = 1 - S[a, msg].item()
    if trace:
        with open(f"D:/tmp/rdtrl_tour59_trace_{cas}_a{a}_b{b}_delta{delta}_pas{pas}.txt", "w") as f:
            f.write("\n".join(lignes))
    print(f"{cas} a={a} b={b} delta={delta!r} eps={eps:.0e} pas={pas} R_b={Rb:.6f} 1-s_a={d:.6e} bascule={bascule}", flush=True)


if __name__ == "__main__":
    # arguments optionnels apres les cinq positionnels : "trace" et/ou "eps=<valeur>"
    # (eps d'Adam pendant la phase ponderee ; l'etat de depart reste celui a 1e-10)
    cas, a, b, delta, pas = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    opts = sys.argv[6:]
    eps = next((float(o[4:]) for o in opts if o.startswith("eps=")), ADAM_EPS)
    courir(cas, a, b, delta, pas, trace="trace" in opts, eps=eps)
