"""Test precommis par l'agent-dipankar (18/09/2026, CARNET.md §7.65) pour
trancher entre deux lectures concurrentes du chaos observe dans la
fenetre H6-direct (pas<=24) :

  (a) "R n'a pas eu le temps de bouger" (ma lecture initiale) ;
  (b) "l'etat interne d'Adam (exp_avg, exp_avg_sq) n'a pas eu le temps
      de se verrouiller sur la direction propre locale du col -- R lui-
      meme importe peu" (lecture de l'agent, appuyee par un calcul de
      jacobienne independant montrant un vrai col hyperbolique reel
      dans le champ CONTINU, valeurs propres +3,2e-6/-1,1e-4).

Protocole precommis (verbatim, par l'agent) : avant le premier pas,
ecraser manuellement l'etat Adam (exp_avg, exp_avg_sq, step~20) de TOUS
les parametres avec des valeurs coherentes avec le gradient local reel
(mesure par un backward() frais a ce point), SANS deplacer s3 ni R
d'un iota -- puis refaire le fit sur 24 pas.

Prediction sous (b), l'etat interne d'Adam explique tout : les
changements de signe de d(s3) devraient tomber vers 0-1.
Prediction sous (a), R lui-meme est le probleme : les changements de
signe devraient rester proches de ceux du cas froid (4, sur pas=1-24),
puisque R n'a toujours pas bouge d'un iota au depart.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta

DELTA = 0.013
ADAM_EPS = 1e-10
S3_H6 = 0.994295
R_H6 = 0.829390
BETA1, BETA2 = 0.9, 0.999
T_INJECTE = 20


def compter_changements_signe(deltas):
    signes = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
    return sum(1 for i in range(1, len(signes))
               if signes[i] != 0 and signes[i - 1] != 0 and signes[i] != signes[i - 1])


def run_froid(pas=24):
    """Cas de reference : Adam demarre a zero (m=0,v=0,t=0), comme dans
    toutes les tentatives precedentes."""
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, S3_H6)
    fixer_r4(r, R_H6)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=ADAM_EPS,
                            betas=(BETA1, BETA2))
    s3_vals = [S3_H6]
    for i in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            s3_vals.append(e.loi()[3, 10].item())
    deltas = [s3_vals[i + 1] - s3_vals[i] for i in range(len(s3_vals) - 1)]
    return s3_vals, deltas


def run_moments_injectes(pas=24, t_injecte=T_INJECTE):
    """Meme point de depart EXACT (s3,R inchanges), mais l'etat interne
    d'Adam est initialise comme s'il avait deja fait t_injecte pas de
    vrai gradient coherent avec la direction locale actuelle."""
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, S3_H6)
    fixer_r4(r, R_H6)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=ADAM_EPS,
                            betas=(BETA1, BETA2))

    # 1) backward frais pour capturer le gradient local reel de CHAQUE parametre
    j, _ = objectif_pondere(e, r, BETA, poids)
    opt.zero_grad()
    (-j).backward()
    grads_captures = {p: p.grad.clone() for p in parametres(e, r)}

    # 2) un pas reel pour que torch cree les entrees state (dtype/format corrects)
    opt.step()

    # 3) remettre s3 et R EXACTEMENT sur la cible (annule le deplacement du pas 2)
    fixer_s3(e, S3_H6)
    fixer_r4(r, R_H6)

    # 4) ecraser exp_avg/exp_avg_sq/step avec des valeurs "comme si" verrouillees
    #    depuis t_injecte vrais pas, coherentes avec le gradient local capture
    for p in parametres(e, r):
        st = opt.state[p]
        g = grads_captures[p]
        with torch.no_grad():
            st['exp_avg'].copy_(g)
            st['exp_avg_sq'].copy_(g * g)
        if torch.is_tensor(st['step']):
            st['step'].fill_(float(t_injecte))
        else:
            st['step'] = t_injecte

    # 5) le fit de 24 pas, exactement comme le cas froid
    s3_vals = [S3_H6]
    for i in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            s3_vals.append(e.loi()[3, 10].item())
    deltas = [s3_vals[i + 1] - s3_vals[i] for i in range(len(s3_vals) - 1)]
    return s3_vals, deltas


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    print("=== CAS FROID (m=0,v=0,t=0), pas 1-24 ===")
    s3_froid, d_froid = run_froid(24)
    print(f"  s3: {s3_froid[0]:.6f} -> {s3_froid[-1]:.6f}")
    print(f"  changements de signe de d(s3) : {compter_changements_signe(d_froid)}")
    print(f"  deltas: {[f'{d:+.2e}' for d in d_froid]}")

    print()
    print(f"=== MOMENTS INJECTES (m,v,t={T_INJECTE} coherents avec le gradient local), pas 1-24 ===")
    s3_inj, d_inj = run_moments_injectes(24, T_INJECTE)
    print(f"  s3: {s3_inj[0]:.6f} -> {s3_inj[-1]:.6f}")
    print(f"  changements de signe de d(s3) : {compter_changements_signe(d_inj)}")
    print(f"  deltas: {[f'{d:+.2e}' for d in d_inj]}")

    print()
    print("=== Verdict ===")
    print(f"  froid : {compter_changements_signe(d_froid)} changements de signe")
    print(f"  injecte : {compter_changements_signe(d_inj)} changements de signe")
    print("  Sous (b) [etat Adam est LA cause] : injecte devrait tomber vers 0-1.")
    print("  Sous (a) [R lui-meme est la cause] : injecte devrait rester proche de froid.")
