"""Mesure du residu du recepteur sous perturbation, sur le mur idx5.

Consolide les scripts jetables des 08-09/09/2026 (dialogue avec
dipankarsarkar, CARNET.md 7.60bis/7.60ter/7.60quater) : on pousse le
referent 0 (le mur, jamais engage) au-dela de son seuil sur le message 0,
et on regarde ce que devient la masse que le message 0 envoie a tout ce
qui n'est pas le referent 0 -- le "residu".

Deux regimes tres differents :
  - a convergence (20000 pas de plus), le residu est fige quel que soit
    eps (24 a 100) : ~4.6e-12, spread relatif <1%.
  - PENDANT le transfert (pas=6000 a 8000), le residu bouge de 0.5 a
    ~1e-9 sur neuf ordres de grandeur -- la queue figee n'est que la fin
    d'un effondrement, pas un signal sur le mecanisme.

r (recepteur) et R=s*r (cellule de recompense) ne coincident QUE quand
le referent est deja sature (1-s proche de 0) : R et r sont deux objets
distincts, lies par R=s*r, qui ne s'accordent que par construction une
fois que 1-s sous-passe la precision qui les separe.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_idx5 import replay_idx5, monter_avec_eps_adam, BETA
from representable_atteignable_stable import monter


def residu_a_convergence(eps_liste=(24.0, 26.0, 30.0, 40.0, 60.0, 100.0), pas_suite=20000):
    """Residu et fuites une fois le transfert termine (referent 0 sature)."""
    torch.set_printoptions(precision=15)
    print(f"=== residu a convergence ({pas_suite} pas post-perturbation) ===")
    for eps_test in eps_liste:
        e, r = replay_idx5(40000)
        with torch.no_grad():
            e.p[0][0, 0] += eps_test
        monter(e, r, BETA, pas_suite, lr=0.05)
        with torch.no_grad():
            s_, r_ = e.loi(), r.loi()
            Rcell = (s_[0, 0] * r_[0, 0]).item()
            colonne = s_[:, 0] * r_[0, :]
            total = colonne.sum().item()
            residu = total - Rcell
            un_moins_R = 1.0 - Rcell
            un_moins_r = 1.0 - r_[0, 0].item()
            un_moins_s = 1.0 - s_[0, 0].item()
        print(f"eps={eps_test:6.1f}  R[0,0]={Rcell:.15f}  residu={residu:.6e}  "
              f"1-R={un_moins_R:.6e}  1-r={un_moins_r:.6e}  1-s={un_moins_s:.6e}")


def residu_pendant_transfert(eps_test=24.0, pas_liste=(6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000)):
    """Residu et fuites au fil du VRAI transitoire (pas de convergence forcee)."""
    torch.set_printoptions(precision=15)
    print(f"=== residu pendant le transfert (eps={eps_test}) ===")
    for pas_suite in pas_liste:
        e, r = replay_idx5(40000)
        with torch.no_grad():
            e.p[0][0, 0] += eps_test
        monter(e, r, BETA, pas_suite, lr=0.05)
        with torch.no_grad():
            s_, r_ = e.loi(), r.loi()
            Rcell = (s_[0, 0] * r_[0, 0]).item()
            colonne = s_[:, 0] * r_[0, :]
            total = colonne.sum().item()
            residu = total - Rcell
            un_moins_r = 1.0 - r_[0, 0].item()
            un_moins_s = 1.0 - s_[0, 0].item()
        print(f"pas={pas_suite:6d}  R[0,0]={Rcell:.6f}  residu={residu:.6e}  "
              f"1-r={un_moins_r:.6e}  1-s={un_moins_s:.6e}")


if __name__ == "__main__":
    residu_a_convergence()
    print()
    residu_pendant_transfert()
