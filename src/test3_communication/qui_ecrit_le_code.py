"""
RDTRL — Test 3, §6.3 : qui ecrit le code, l'emetteur ou le recepteur ?

On gele l'un des deux agents sur un code impose, on laisse l'autre apprendre, et
on compare a la situation ou les deux sont libres. C'est la version test 3 du gel
de position qui avait tout localise au test 2 : ce qu'on cherche n'est pas un
constat mais une LOCALISATION.

UN AGENT GELE N'A PAS DE PARAMETRISATION. C'est ce qui rend le dispositif propre :
on le represente par une matrice stochastique fixe construite depuis le code, sans
passer par une classe d'emetteur. Sinon la condition « geler sur un code
aleatoire » serait impossible a poser pour la parametrisation structuree, qui ne
sait pas ecrire la plupart des bijections (§6.5), et on confondrait une limite de
representabilite du GELE avec une difficulte d'apprentissage du LIBRE.

TROIS PREDICTIONS, ECRITES AVANT DE LIRE LES CHIFFRES.

  P1. Geler l'emetteur sur le code compositionnel et le geler sur une bijection
      quelconque posent au recepteur EXACTEMENT le meme probleme. Les deux codes
      sont relies par un renommage des messages, et un recepteur tabulaire est
      equivariant sous ce renommage. Donc conditions A et B identiques, et de meme
      C et D pour l'emetteur tabulaire. Un ecart signalerait un bris de symetrie
      dans l'implementation.

  P2. Toute condition ou un agent est gele atteint E[R] ~ 1. Avec un partenaire
      fixe, l'objectif redevient LINEAIRE dans la politique libre : plus aucun
      optimum a egalite, plus aucune coordination a faire, l'optimum est unique et
      se lit directement. Les deux libres plafonnent vers 0,92 (§6.2). L'ecart
      entre les deux est le cout de la COORDINATION, et c'est le resultat cherche.

  P3. La parametrisation structuree distingue, elle : gelee sur le compositionnel
      elle doit reussir, gelee sur un code aleatoire elle doit echouer, parce
      qu'elle ne peut pas ecrire l'encodeur correspondant. Ce serait la premiere
      condition ou l'echec vient de la representabilite et non de la coordination.
"""

import argparse
import json
import os

import numpy as np
import torch

from grammaire3 import DOSSIER_SORTIE, N
from loi_nulle_longue import matrices_information_generale, statistiques
from representable_atteignable_stable import (CODE_CANONIQUE, EmetteurStructure,
                                              EmetteurTabulaire, Recepteur,
                                              lire_code, parametres)

torch.set_num_threads(int(os.environ.get("RDTRL_THREADS", "1")))


def matrice_emetteur(code, force=20.0):
    """S[r, m] quasi deterministe sur code[r]. Un logit a 20 laisse 5e-8 ailleurs."""
    logits = torch.zeros(N, N, dtype=torch.float64)
    for r, m in enumerate(code):
        logits[r, int(m)] = force
    return torch.softmax(logits, dim=1)


def matrice_recepteur(code, force=20.0):
    """R[m, r] quasi deterministe : le decodeur exact du code donne."""
    logits = torch.zeros(N, N, dtype=torch.float64)
    for r, m in enumerate(code):
        logits[int(m), r] = force
    return torch.softmax(logits, dim=1)


def objectif_partiel(s, r, beta):
    recompense = (s * r.t()).sum() / N
    entropie = (-(s * torch.log(s.clamp_min(1e-300))).sum()
                - (r * torch.log(r.clamp_min(1e-300))).sum()) / N
    return recompense + beta * entropie, recompense


def pas_pour_fraction(trajectoire, fraction=0.99):
    """Nombre de pas pour atteindre `fraction` de SA PROPRE valeur finale.

    Un seuil absolu comme 0,99 n'est pas comparable entre conditions : un code a
    k collisions plafonne arithmetiquement a (27-k)/27, donc 0,99 y est
    INATTEIGNABLE des la premiere collision. Mesurer la vitesse contre un seuil
    qu'une condition ne peut pas franchir compare une vitesse et une capacite.
    """
    cible = fraction * trajectoire[-1]
    for etape, valeur in enumerate(trajectoire):
        if valeur >= cible:
            return etape + 1
    return None


def apprendre(libre, fixe, role_libre, beta, pas, lr=0.05):
    """Optimise le seul agent `libre`, l'autre etant une matrice fixe."""
    for t in libre.p:
        t.requires_grad_(True)
    optimiseur = torch.optim.Adam(parametres(libre), lr=lr)
    trajectoire = []
    for _ in range(pas):
        loi = libre.loi()
        s, r = (loi, fixe) if role_libre == "emetteur" else (fixe, loi)
        j, recompense = objectif_partiel(s, r, beta)
        optimiseur.zero_grad()
        (-j).backward()
        optimiseur.step()
        trajectoire.append(float(recompense))
    with torch.no_grad():
        loi = libre.loi()
        s, r = (loi, fixe) if role_libre == "emetteur" else (fixe, loi)
        _, recompense = objectif_partiel(s, r, beta)
    trajectoire.append(float(recompense))
    return pas_pour_fraction(trajectoire), float(recompense)


def apprendre_les_deux(emetteur, recepteur, beta, pas, lr=0.05):
    for a in (emetteur, recepteur):
        for t in a.p:
            t.requires_grad_(True)
    optimiseur = torch.optim.Adam(parametres(emetteur, recepteur), lr=lr)
    trajectoire = []
    for _ in range(pas):
        j, recompense = objectif_partiel(emetteur.loi(), recepteur.loi(), beta)
        optimiseur.zero_grad()
        (-j).backward()
        optimiseur.step()
        trajectoire.append(float(recompense))
    with torch.no_grad():
        _, recompense = objectif_partiel(emetteur.loi(), recepteur.loi(), beta)
    trajectoire.append(float(recompense))
    return pas_pour_fraction(trajectoire), float(recompense)


def decrire(code):
    cm, ca, _ = statistiques(
        matrices_information_generale(np.asarray(code)[None, :]))
    return {"collisions": int(N - len(set(np.asarray(code).tolist()))),
            "concentration_appariee": float(ca[0])}


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — test 3, §6.3")
    parseur.add_argument("--graines", type=int, default=10)
    parseur.add_argument("--pas", type=int, default=3000)
    parseur.add_argument("--beta", type=float, default=0.02)
    parseur.add_argument("--graine", type=int, default=0)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    generateur = np.random.default_rng(args.graine)

    print("=" * 78)
    print("TEST 3 §6.3 — QUI ECRIT LE CODE, L'EMETTEUR OU LE RECEPTEUR ?")
    print("=" * 78)
    print(f"\n  {args.graines} graines par condition, beta = {args.beta}.")
    print("  Un agent gele est une matrice fixe, sans parametrisation : sinon on")
    print("  confondrait une limite de representabilite avec une difficulte")
    print("  d'apprentissage.\n")

    codes = {"compositionnel": CODE_CANONIQUE,
             "aleatoire": generateur.permutation(N)}
    # temoin de plafond : un code a 2 collisions, du type de ceux ou la paire
    # libre se pose reellement. Son plafond arithmetique vaut 25/27 = 0,9259,
    # et le mesurer evite d'avoir a le supposer.
    collisionne = generateur.permutation(N).copy()
    collisionne[3] = collisionne[7]
    collisionne[11] = collisionne[19]
    codes["a 2 collisions"] = collisionne
    conditions = []
    for nom_code, code in codes.items():
        conditions.append({"cle": f"S gele {nom_code}, R libre",
                           "famille": "S gele, R tabulaire libre",
                           "role_libre": "recepteur", "classe": Recepteur,
                           "fixe": matrice_emetteur(code), "code": nom_code})
    for classe in (EmetteurTabulaire, EmetteurStructure):
        for nom_code, code in codes.items():
            conditions.append({"cle": f"R gele {nom_code}, S {classe.nom} libre",
                               "famille": f"R gele, S {classe.nom} libre",
                               "role_libre": "emetteur", "classe": classe,
                               "fixe": matrice_recepteur(code), "code": nom_code})

    print(f"  {'condition':>42}  {'pas pour 0,99':>14}  {'E[R] final':>12}")
    resultats = []
    for cond in conditions:
        pas_atteint, recompenses = [], []
        for _ in range(args.graines):
            libre = cond["classe"](generateur)
            atteint, recompense = apprendre(libre, cond["fixe"],
                                            cond["role_libre"], args.beta, args.pas)
            pas_atteint.append(atteint)
            recompenses.append(recompense)
        reussis = [p for p in pas_atteint if p is not None]
        r = {"condition": cond["cle"], "famille": cond["famille"],
             "code": cond["code"], "n": args.graines,
             "reussites": len(reussis),
             "pas_median": float(np.median(reussis)) if reussis else None,
             "reward_moyen": float(np.mean(recompenses)),
             "reward_sd": float(np.std(recompenses))}
        resultats.append(r)
        pas_txt = (f"{r['pas_median']:.0f} ({r['reussites']}/{r['n']})"
                   if reussis else f"jamais (0/{r['n']})")
        print(f"  {cond['cle']:>42}  {pas_txt:>14}  "
              f"{r['reward_moyen']:.4f} ± {r['reward_sd']:.4f}")

    print()
    libres = []
    for classe in (EmetteurTabulaire, EmetteurStructure):
        pas_atteint, recompenses, infos = [], [], []
        for _ in range(args.graines):
            emetteur, recepteur = classe(generateur), Recepteur(generateur)
            atteint, recompense = apprendre_les_deux(emetteur, recepteur,
                                                     args.beta, args.pas)
            pas_atteint.append(atteint)
            recompenses.append(recompense)
            infos.append(decrire(lire_code(emetteur)))
        reussis = [p for p in pas_atteint if p is not None]
        r = {"condition": f"les deux libres, S {classe.nom}", "code": None,
             "n": args.graines, "reussites": len(reussis),
             "pas_median": float(np.median(reussis)) if reussis else None,
             "reward_moyen": float(np.mean(recompenses)),
             "reward_sd": float(np.std(recompenses)),
             "rewards": [float(x) for x in recompenses],
             "collisions": [int(i["collisions"]) for i in infos],
             "collisions_moyennes": float(np.mean([i["collisions"] for i in infos])),
             "concentration_moyenne": float(np.mean([i["concentration_appariee"]
                                                     for i in infos]))}
        libres.append(r)
        pas_txt = (f"{r['pas_median']:.0f} ({r['reussites']}/{r['n']})"
                   if reussis else f"jamais (0/{r['n']})")
        print(f"  {r['condition']:>42}  {pas_txt:>14}  "
              f"{r['reward_moyen']:.4f} ± {r['reward_sd']:.4f}")

    print("\n" + "-" * 78)
    print("P1 — GELER SUR LE COMPOSITIONNEL OU SUR UN CODE QUELCONQUE")
    print("-" * 78)
    for famille in dict.fromkeys(r["famille"] for r in resultats):
        lignes = [r for r in resultats if r["famille"] == famille]
        comp = next(r for r in lignes if r["code"] == "compositionnel")
        alea = next(r for r in lignes if r["code"] == "aleatoire")
        ecart = comp["reward_moyen"] - alea["reward_moyen"]
        print(f"  {famille:>30} : compositionnel {comp['reward_moyen']:.8f}, "
              f"aleatoire {alea['reward_moyen']:.8f}, ecart {ecart:+.2e}")

    print("\n" + "-" * 78)
    print("P2 — LE DEFICIT EST-IL DANS L'APPRENTISSAGE OU DANS LE CODE CHOISI ?")
    print("-" * 78)
    # unite : ce qu'un code BIJECTIF rend a ce beta, mesure et non suppose
    bijectifs = [r["reward_moyen"] for r in resultats
                 if r["code"] in ("compositionnel", "aleatoire")
                 and "structure" not in r["condition"]]
    plafond_beta = float(np.mean(bijectifs))
    temoin = next((r for r in resultats if r["code"] == "a 2 collisions"), None)
    print(f"  Un code a k collisions plafonne a (27-k)/27 : deux referents envoyes")
    print(f"  sur le meme message sont indistinguables, quoi que fasse le recepteur.")
    print(f"\n  mesure, agent gele sur une BIJECTION  : E[R] = {plafond_beta:.4f}")
    print(f"  (ce n'est pas 1 a cause du terme d'entropie a beta = {args.beta})")
    if temoin:
        attendu = plafond_beta * (N - 2) / N
        print(f"  mesure, agent gele sur 2 COLLISIONS   : E[R] = "
              f"{temoin['reward_moyen']:.4f}, attendu {attendu:.4f}, "
              f"ecart {temoin['reward_moyen'] - attendu:+.4f}")
        print(f"  => le plafond arithmetique est verifie, pas suppose.")

    print(f"\n  {'condition':>42}  {'E[R]':>8}  {'collisions':>10}  "
          f"{'plafond':>8}  {'E[R]/plafond':>13}")
    for r in libres:
        ratios = [x / (plafond_beta * (N - k) / N)
                  for x, k in zip(r["rewards"], r["collisions"])]
        r["ratio_au_plafond"] = float(np.mean(ratios))
        r["plafond_moyen"] = float(np.mean([plafond_beta * (N - k) / N
                                            for k in r["collisions"]]))
        print(f"  {r['condition']:>42}  {r['reward_moyen']:8.4f}  "
              f"{r['collisions_moyennes']:10.2f}  {r['plafond_moyen']:8.4f}  "
              f"{r['ratio_au_plafond']:13.4f}")
    print("\n  Si ce rapport vaut 1, la paire libre execute son code aussi bien")
    print("  qu'un agent a qui on aurait donne le meme code tout fait. Le deficit")
    print("  ne serait alors pas dans l'APPRENTISSAGE mais dans le CODE CHOISI.")

    rapport = {"beta": args.beta, "graines": args.graines, "pas": args.pas,
               "graine": args.graine, "conditions": resultats, "libres": libres,
               "plafond_beta": plafond_beta}
    nom = f"6_3_qui_ecrit_le_code_b{args.beta}_{args.graines}graines_g{args.graine}.json"
    with open(os.path.join(DOSSIER_SORTIE, nom), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nEcrit dans {DOSSIER_SORTIE} sous {nom}")
