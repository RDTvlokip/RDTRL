"""Sonde de capacite : le GRU peut-il seulement REPRESENTER la politique
diverse-et-valide, ou l'optimisation echoue-t-elle a la trouver ?

Ce script n'est PAS un apprentissage et ne fait partie d'aucune revendication
sur le RL. C'est un diagnostic : on ajuste le meme reseau en supervise vers la
loi uniforme sur les 48 phrases valides, et on regarde ce qu'il atteint.

  - s'il atteint ~100 % de masse grammaticale et ~48 modes effectifs, alors la
    solution existe dans la classe de modeles, et le compromis validite /
    diversite observe en RL est un echec d'OPTIMISATION de REINFORCE ;
  - s'il n'y arrive pas, le compromis est une limite de REPRESENTATION et la
    conclusion serait toute autre.

Cible theorique : pour une loi uniforme sur les 48 phrases valides, il faut
P(les) = P(des) = 12/48 = 0.25 et P(le) = P(la) = P(un) = P(une) = 6/48 = 0.125,
puis uniforme sur les noms compatibles, puis sur les verbes compatibles.
"""

import torch

from grammaire import Grammaire
from rl_grammaire import PolitiqueGRU, analyse_exacte, fixer_graine, distribution_exacte

ETAPES = 3000
LR = 5e-3


def phrases_valides_en_ids(grammaire):
    from itertools import product
    return [c for c in product(range(grammaire.taille), repeat=grammaire.longueur)
            if grammaire.analyser(c)["valide"]]


def ajuster(politique, cibles, etapes=ETAPES, lr=LR):
    """Maximise la log-vraisemblance moyenne des phrases valides, ce qui revient
    a minimiser KL(uniforme sur les 48 || politique)."""
    sequences = torch.tensor(cibles, dtype=torch.long)
    debut = torch.full((len(sequences), 1), politique.token_debut, dtype=torch.long)
    entrees = torch.cat([debut, sequences[:, :-1]], dim=1)
    optimiseur = torch.optim.Adam(politique.parameters(), lr=lr)
    for etape in range(etapes):
        sorties, _ = politique.gru(politique.embedding(entrees))
        log_probas = torch.log_softmax(politique.tete(sorties), dim=-1)
        log_p = log_probas.gather(2, sequences.unsqueeze(-1)).squeeze(-1).sum(1)
        perte = -log_p.mean()
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        if (etape + 1) % 500 == 0:
            print(f"    etape {etape+1:5d} | -log P moyenne = {float(perte):.4f} "
                  f"(optimum theorique = {torch.log(torch.tensor(48.0)):.4f})")
    return politique


if __name__ == "__main__":
    grammaire = Grammaire(longue=False)
    cibles = phrases_valides_en_ids(grammaire)
    print("=" * 78)
    print("SONDE DE CAPACITE — le modele peut-il representer la solution ideale ?")
    print("=" * 78)
    print(f"Cible : loi uniforme sur les {len(cibles)} phrases valides")
    print(f"Optimum theorique de -log P : ln(48) = "
          f"{float(torch.log(torch.tensor(48.0))):.4f}\n")

    for graine in (0, 1, 2):
        print(f"  graine {graine} :")
        fixer_graine(graine)
        politique = PolitiqueGRU(grammaire.taille)
        avant = analyse_exacte(politique, grammaire)
        ajuster(politique, cibles)
        apres = analyse_exacte(politique, grammaire)
        print(f"    avant ajustement : masse valide {avant['masse_valide_pct']:6.2f} % | "
              f"{avant['modes_effectifs']:5.1f} modes")
        print(f"    apres ajustement : masse valide {apres['masse_valide_pct']:6.2f} % | "
              f"{apres['modes_effectifs']:5.1f} modes | "
              f"uniformite {apres['uniformite_pct']:.1f} % | "
              f"sg {apres['repartition_familles']['sg']:.1f} % / "
              f"pl {apres['repartition_familles']['pl']:.1f} %")
        print(f"    P(nom accorde | det) = {apres['moyenne_cond_det']:.4f} | "
              f"P(verbe accorde | nom) = {apres['moyenne_cond_nom']:.4f}")
        print(f"    masse par determinant : {apres['masse_par_determinant']}")
        print(f"    (cible : les/des = 0.25 chacun, le/la/un/une = 0.125 chacun)\n")
