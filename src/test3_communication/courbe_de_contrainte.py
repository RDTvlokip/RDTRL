"""
RDTRL — Test 3, §6.6 : la courbe qui remplace le verdict.

La recompense etant indifferente a la compositionnalite (§3), celle-ci ne peut
venir que d'une contrainte EXTERIEURE. §6.7 a transforme la liste de recettes du
document en une hypothese unique et testable :

    une contrainte ne peut produire de la compositionnalite que si elle BRISE LA
    SYMETRIE, c'est-a-dire si elle reduit le groupe sous lequel la procedure
    entiere est equivariante, de S_27 (transitif sur les bijections, donc
    imposant l'equiprobabilite) a un groupe qui ne l'est plus.

CE QUE J'AI VERIFIE AVANT D'ECRIRE LE RESTE, ET QUI CONTREDIT MA PROPRE TABLE.

Le document justifiait le bruit de canal ainsi : « un code compositionnel ne perd
qu'un attribut quand un token est corrompu ; un code holistique perd tout ». C'est
FAUX pour cette recompense, et ca se calcule sans entrainer quoi que ce soit. Pour
un emetteur deterministe sur un code c et le decodeur optimal,

    E[R]* = (1/27) somme_{m'} max_r C[c(r), m']

et c etant une bijection sur les 27 messages, max_r C[c(r), m'] = max_m C[m, m'],
INDEPENDANT DE c. Mesure : a tout epsilon, compositionnel et bijections aleatoires
donnent la meme valeur a 1,1e-16 pres. Perdre « un seul attribut » ne sert a rien
quand la recompense est tout-ou-rien sur le referent exact ; il faudrait une
recompense a credit partiel.

MAIS — et c'est toute l'experience — le canal brise quand meme la SYMETRIE. La
matrice C ne commute pas avec une permutation quelconque des messages, seulement
avec les 1 296 qui respectent la structure de produit. Donc :

  - le certificat des optima a egalite continue de dire que RIEN ne distingue les
    bijections en recompense, exactement, a tout epsilon ;
  - l'argument d'equivariance, lui, ne force plus l'equiprobabilite.

C'est le cas le plus pur possible de la these du projet : une selection qui opere
ENTIEREMENT hors de la recompense. Reste a savoir si elle opere, et dans quel sens.

QUATRE PREDICTIONS, ECRITES AVANT LES ENTRAINEMENTS.

  P1. Le canal ne brise pas l'egalite des recompenses. VERIFIE ci-dessus, exact.
  P2. Le canal brise la symetrie : C commute avec les 1 296, pas avec les autres.
      Verifiable exactement, et verifie plus bas.
  P3. Je ne sais pas si la concentration emergente bouge, ni dans quel sens. C'est
      la seule vraie inconnue de cette section, et le theoreme de §6.7 ne dit rien
      ici puisqu'il ne s'applique plus.
  P4. Le renouvellement de population, lui, PRESERVE l'equivariance sous S_27 :
      remplacer un agent tabulaire par un agent tabulaire neuf est une operation
      echangeable. Le theoreme s'applique donc encore, et z doit rester nul QUELLE
      QUE SOIT la periode. Si l'iterated learning produit de la compositionnalite
      dans la litterature, ca ne peut donc pas venir du renouvellement seul.
      (Je n'ai pas fait la revue de litterature ; c'est une prediction sur mon
      dispositif, pas une affirmation sur les travaux des autres.)
"""

import argparse
import json
import os

import numpy as np
import torch

from code_emergent import nulle_du_profil, profil, situer
from grammaire3 import (DOSSIER_SORTIE, INDEX_MESSAGE, MESSAGES, N,
                        N_POSITIONS, N_TOKENS, REFERENTS)
from loi_nulle_longue import matrices_information_generale, statistiques
from representable_atteignable_stable import (CODE_CANONIQUE, EmetteurTabulaire,
                                              Recepteur, lire_code, parametres)

torch.set_num_threads(int(os.environ.get("RDTRL_THREADS", "1")))


def canal(epsilon):
    """C[m, m'] : chaque token est remplace par un token uniforme avec proba eps."""
    par_token = np.full((N_TOKENS, N_TOKENS), epsilon / N_TOKENS)
    np.fill_diagonal(par_token, 1 - epsilon + epsilon / N_TOKENS)
    matrice = np.ones((N, N))
    for i, m in enumerate(MESSAGES):
        for j, mp in enumerate(MESSAGES):
            matrice[i, j] = np.prod([par_token[m[k], mp[k]]
                                     for k in range(N_POSITIONS)])
    return torch.as_tensor(matrice, dtype=torch.float64)


def valeur_optimale(code, matrice_canal):
    """E[R] d'un emetteur deterministe sur `code` avec le decodeur optimal."""
    c = matrice_canal.numpy()
    return float(c[np.asarray(code), :].max(axis=0).sum() / N)


def groupe_structurel():
    """Les 1 296 permutations des messages respectant la structure de produit."""
    from itertools import permutations, product
    elements = []
    for tau in permutations(range(N_POSITIONS)):
        for g in product(permutations(range(N_TOKENS)), repeat=N_POSITIONS):
            pi = np.empty(N, dtype=np.int64)
            for i, message in enumerate(MESSAGES):
                pi[i] = INDEX_MESSAGE[tuple(g[j][message[tau[j]]]
                                            for j in range(N_POSITIONS))]
            elements.append(pi)
    return elements


def objectif_canal(emetteur, recepteur, beta, matrice_canal):
    s, r = emetteur.loi(), recepteur.loi()
    recompense = (s @ matrice_canal @ r).diagonal().sum() / N
    entropie = (-(s * torch.log(s.clamp_min(1e-300))).sum()
                - (r * torch.log(r.clamp_min(1e-300))).sum()) / N
    return recompense + beta * entropie, recompense


def monter_canal(emetteur, recepteur, beta, pas, matrice_canal, lr=0.05,
                 periode_renouvellement=None, generateur=None):
    """Montee exacte a travers le canal, avec renouvellement optionnel du recepteur."""
    for a in (emetteur, recepteur):
        for t in a.p:
            t.requires_grad_(True)
    # DEUX optimiseurs, et pas un seul. Renouveler le recepteur en recreant un
    # optimiseur commun remettrait aussi a zero les moments d'Adam de L'EMETTEUR :
    # la condition « avec renouvellement » differerait alors de la condition sans
    # par deux choses au lieu d'une, et on ne saurait plus laquelle agit.
    opt_e = torch.optim.Adam(parametres(emetteur), lr=lr)
    opt_r = torch.optim.Adam(parametres(recepteur), lr=lr)
    for etape in range(pas):
        if (periode_renouvellement and etape and
                etape % periode_renouvellement == 0):
            neuf = Recepteur(generateur)
            with torch.no_grad():
                for cible, source in zip(recepteur.p, neuf.p):
                    cible.copy_(source)
            for t in recepteur.p:
                t.requires_grad_(True)
            opt_r = torch.optim.Adam(parametres(recepteur), lr=lr)
        j, _ = objectif_canal(emetteur, recepteur, beta, matrice_canal)
        opt_e.zero_grad()
        opt_r.zero_grad()
        (-j).backward()
        opt_e.step()
        opt_r.step()
    with torch.no_grad():
        _, recompense = objectif_canal(emetteur, recepteur, beta, matrice_canal)
    return float(recompense)


def mesurer(emetteur, generateur, cache, nulle_n):
    code = lire_code(emetteur)
    p = profil(code)
    _, ca, _ = statistiques(
        matrices_information_generale(np.asarray(code)[None, :]))
    nulle = nulle_du_profil(p, nulle_n, generateur, cache)
    s = situer(float(ca[0]), nulle["appariee"])
    return {"concentration_appariee": float(ca[0]), "collisions": N - len(p),
            "z": s["z"], "centile": s["centile"]}


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — test 3, §6.6")
    parseur.add_argument("--graines", type=int, default=15)
    parseur.add_argument("--pas", type=int, default=3000)
    parseur.add_argument("--nulle", type=int, default=20000)
    parseur.add_argument("--beta", type=float, default=0.02)
    parseur.add_argument("--epsilons", type=float, nargs="*",
                         default=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50])
    parseur.add_argument("--periodes", type=int, nargs="*",
                         default=[0, 1000, 300, 100])
    parseur.add_argument("--graine", type=int, default=0)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    generateur = np.random.default_rng(args.graine)
    cache = {}
    rapport = {"beta": args.beta, "graines": args.graines, "graine": args.graine}

    print("=" * 78)
    print("TEST 3 §6.6 — LA COURBE DE CONTRAINTE")
    print("=" * 78)

    print("\n" + "-" * 78)
    print("P1 — LE CANAL BRISE-T-IL L'EGALITE DES RECOMPENSES ? NON, ET C'EST EXACT")
    print("-" * 78)
    print("  Ma table de §6.6 justifiait le bruit par « le compositionnel ne perd")
    print("  qu'un attribut ». Faux pour une recompense tout-ou-rien.\n")
    print(f"  {'eps':>6}  {'compositionnel':>15}  {'200 bijections':>22}  {'ecart':>10}")
    tie = []
    for eps in args.epsilons:
        c = canal(eps)
        v = valeur_optimale(CODE_CANONIQUE, c)
        alea = np.array([valeur_optimale(generateur.permutation(N), c)
                         for _ in range(200)])
        tie.append({"epsilon": eps, "compositionnel": v,
                    "aleatoire_moyenne": float(alea.mean()),
                    "aleatoire_sd": float(alea.std()),
                    "ecart": float(v - alea.mean())})
        print(f"  {eps:6.2f}  {v:15.8f}  {alea.mean():13.8f} ± {alea.std():.1e}  "
              f"{v - alea.mean():+10.2e}")
    rapport["egalite_des_recompenses"] = tie
    print("\n  Argument : E[R]* = (1/27) somme_m' max_r C[c(r), m'], et c etant une")
    print("  bijection sur les 27 messages, le max ne depend pas de c.")

    print("\n" + "-" * 78)
    print("P2 — LE CANAL BRISE-T-IL LA SYMETRIE ? OUI, DE 27! A 1 296")
    print("-" * 78)
    c = canal(0.2).numpy()
    groupe = groupe_structurel()
    ecarts_groupe = [np.abs(c[np.ix_(pi, pi)] - c).max() for pi in groupe[::37]]
    ecarts_quelconques = [np.abs(c[np.ix_(pi, pi)] - c).max()
                          for pi in (generateur.permutation(N) for _ in range(200))]
    print(f"  a epsilon = 0,2, ecart max entre C et sa version permutee :")
    print(f"    sur le groupe structurel (echantillon de {len(ecarts_groupe)}) : "
          f"{max(ecarts_groupe):.2e}")
    print(f"    sur 200 permutations quelconques : min {min(ecarts_quelconques):.3f}, "
          f"median {np.median(ecarts_quelconques):.3f}")
    print("  => C commute avec les 1 296 et avec aucune autre. Le theoreme")
    print("     d'equivariance de §6.7 ne s'applique donc plus, alors meme que")
    print("     l'egalite des recompenses, elle, tient toujours.")
    rapport["symetrie"] = {"ecart_max_groupe": float(max(ecarts_groupe)),
                           "ecart_min_quelconque": float(min(ecarts_quelconques))}

    print("\n" + "-" * 78)
    print("P3 — LA COURBE : CONCENTRATION EN FONCTION DU BRUIT")
    print("-" * 78)
    print(f"  {args.graines} graines par epsilon, emetteur TABULAIRE, celui dont")
    print("  §6.1 prouve qu'il ne prefere rien a epsilon = 0.\n")
    print(f"  {'eps':>6}  {'E[R]':>7}  {'collisions':>10}  {'C appariee':>17}  "
          f"{'z':>14}  {'>q99,9':>7}")
    courbe = []
    for eps in args.epsilons:
        matrice = canal(eps)
        mesures, recompenses = [], []
        for _ in range(args.graines):
            e, r = EmetteurTabulaire(generateur), Recepteur(generateur)
            recompenses.append(monter_canal(e, r, args.beta, args.pas, matrice))
            mesures.append(mesurer(e, generateur, cache, args.nulle))
        z = np.array([m["z"] for m in mesures])
        conc = np.array([m["concentration_appariee"] for m in mesures])
        ligne = {"epsilon": eps, "reward_moyen": float(np.mean(recompenses)),
                 "collisions": float(np.mean([m["collisions"] for m in mesures])),
                 "concentration_moyenne": float(conc.mean()),
                 "concentration_sd": float(conc.std(ddof=1)),
                 "z_moyen": float(z.mean()),
                 "z_erreur_type": float(z.std(ddof=1) / np.sqrt(len(z))),
                 "au_dela_q999": int(sum(m["centile"] > 0.999 for m in mesures))}
        courbe.append(ligne)
        print(f"  {eps:6.2f}  {ligne['reward_moyen']:7.4f}  {ligne['collisions']:10.2f}  "
              f"{conc.mean():8.4f} ± {conc.std(ddof=1):.4f}  "
              f"{ligne['z_moyen']:+7.2f} ± {ligne['z_erreur_type']:.2f}  "
              f"{ligne['au_dela_q999']:>3}/{args.graines:<3}", flush=True)
    rapport["courbe_bruit"] = courbe

    print("\n" + "-" * 78)
    print("P4 — RENOUVELLEMENT DE POPULATION : PREDIT SANS EFFET")
    print("-" * 78)
    print("  Remplacer un recepteur tabulaire par un neuf est echangeable, donc")
    print("  l'equivariance sous S_27 SURVIT, donc le theoreme de §6.7 s'applique")
    print("  encore et z doit rester nul quelle que soit la periode.\n")
    print(f"  {'periode':>8}  {'E[R]':>7}  {'collisions':>10}  {'C appariee':>17}  "
          f"{'z':>14}")
    renouvellement = []
    for periode in args.periodes:
        mesures, recompenses = [], []
        for _ in range(args.graines):
            e, r = EmetteurTabulaire(generateur), Recepteur(generateur)
            recompenses.append(monter_canal(
                e, r, args.beta, args.pas, canal(0.0),
                periode_renouvellement=periode or None, generateur=generateur))
            mesures.append(mesurer(e, generateur, cache, args.nulle))
        z = np.array([m["z"] for m in mesures])
        conc = np.array([m["concentration_appariee"] for m in mesures])
        ligne = {"periode": periode, "reward_moyen": float(np.mean(recompenses)),
                 "collisions": float(np.mean([m["collisions"] for m in mesures])),
                 "concentration_moyenne": float(conc.mean()),
                 "z_moyen": float(z.mean()),
                 "z_erreur_type": float(z.std(ddof=1) / np.sqrt(len(z)))}
        renouvellement.append(ligne)
        print(f"  {periode if periode else 'aucun':>8}  {ligne['reward_moyen']:7.4f}  "
              f"{ligne['collisions']:10.2f}  {conc.mean():8.4f} ± {conc.std(ddof=1):.4f}  "
              f"{ligne['z_moyen']:+7.2f} ± {ligne['z_erreur_type']:.2f}", flush=True)
    rapport["renouvellement"] = renouvellement

    nom = f"6_6_courbe_de_contrainte_{args.graines}graines_g{args.graine}.json"
    with open(os.path.join(DOSSIER_SORTIE, nom), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nEcrit dans {DOSSIER_SORTIE} sous {nom}")
