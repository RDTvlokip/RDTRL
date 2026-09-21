"""Suite tour 56 (21/09/2026), sur consigne de Theo ("continue et
continue a creuser" dans l'espace 27x27) : ferme le mecanisme de la
derive de fond de la ligne 10 du recepteur (les 25 entrees hors r3/r4
qui bougent 20x plus en agregat que r3/r4 individuellement,
verifier_derive_fond_ligne10.py).

Chaine de trouvailles du tour :
1. Le classement de derive par referent n'est PAS du bruit (CV~0,011,
   contrairement au "top-6" trompeur de la ligne 3 de l'emetteur qui
   etait pur bruit flottant, CV~1e-11).
2. Le classement correle quasi parfaitement avec la probabilite de
   base de chaque referent (24/25 positions identiques).
3. Le rapport derive/probabilite est quasi constant (~-8,49e9) pour
   24 des 25 referents -- le referent 5 est une vraie anomalie
   (~7,5% d'ecart), reproductible a deux pas de mesure distincts.
4. MECANISME REEL trouve en creusant l'anomalie : ce n'est PAS
   "derive proportionnelle a la probabilite" (une correlation de
   proxy) -- c'est "derive = -lr * m/(sqrt(v)+eps) * n_pas", le pas
   Adam NORMALISE standard, ou m et v sont l'etat Adam propre a
   CHAQUE entree. La correlation avec la probabilite n'etait qu'un
   proxy parce que m et v se trouvent correles a la probabilite pour
   24 des 25 referents -- le referent 5 (et dans une moindre mesure
   9, 18) cassent legerement cette correlation proxy parce que leur
   etat m/v ne suit pas exactement leur probabilite.

C'est le MEME mecanisme plancher-de-v que les gros kicks periodiques
deja caracterises ailleurs dans ce projet (verifier_mecanisme_plancher_v.py)
-- ici v est en permanence au plancher pour ces entrees "au repos"
(sqrt(v)~1,8e-14, eps=1e-10, eps domine largement), donc le pas Adam
normalise se reduit en permanence a m/eps, une DERIVE CONSTANTE (pas
un kick), plutot qu'un evenement periodique -- parce que le gradient
residuel sur ces entrees ne redevient jamais assez grand pour
regonfler v et declencher un vrai kick.

Resultat : ecart de magnitude <0,5% sur les 25 referents, y compris
les deux "anomalies" (5 et 9) qui se resolvent completement une fois
qu'on utilise le vrai m/v de chaque entree plutot qu'un proxy via la
probabilite.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.013026615
LR = 0.05
ADAM_EPS = 1e-10
PAS_DEBUT_FENETRE = 59000
PAS_FIN_FENETRE = 59989


def main():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    params = parametres(e, r)
    opt = torch.optim.Adam(params, lr=LR, eps=ADAM_EPS)
    p_r = r.p[0]

    ligne10_debut = None
    for pas in range(PAS_FIN_FENETRE):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas == PAS_DEBUT_FENETRE - 1:
            with torch.no_grad():
                ligne10_debut = p_r[10, :].clone()

    with torch.no_grad():
        ligne10_fin = p_r[10, :].clone()
    state = opt.state[p_r]
    m = state['exp_avg']
    v = state['exp_avg_sq']

    n_pas = PAS_FIN_FENETRE - PAS_DEBUT_FENETRE
    d_brut = ligne10_fin - ligne10_debut
    autres = [j for j in range(27) if j not in (3, 4)]

    print("referent  d_brut_mesure     predit(-lr*m/(sqrt(v)+eps)*n_pas)   ecart%")
    for j in autres:
        mesure = d_brut[j].item()
        mj = m[10, j].item()
        vj = v[10, j].item()
        pas_normalise = mj / (vj ** 0.5 + ADAM_EPS)
        predit = -LR * pas_normalise * n_pas
        ecart = 100 * abs(predit - mesure) / abs(mesure)
        marque = "  <-- anomalie (ref 5)" if j == 5 else ("  <-- max drift (ref 9)" if j == 9 else "")
        print(f"{j:2d}  {mesure:.6e}   {predit:.6e}   {ecart:.3f}%{marque}")


if __name__ == "__main__":
    main()
