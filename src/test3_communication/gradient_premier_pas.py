"""
RDTRL — Test 3, §6.4 : que voit le gradient au premier pas ?

§4 (c) donne le gradient exact et gratuit : d E[R] / d S[r,m] = R[m,r] / 27, et
symetriquement. D'ou la prediction ecrite dans le document AVANT toute experience :
a l'initialisation les deux politiques sont quasi uniformes, donc chaque gradient
l'est aussi, donc IL N'EXISTE AUCUNE DIRECTION PREFEREE. Contraste net avec le
test 2, ou un desequilibre du lexique imposait une direction des le premier pas.

Le document ajoutait : « si on observe le contraire, c'est que la parametrisation
en introduit une, et il faudra la nommer. » Les sections faites aujourd'hui lui
donnent un nom — le GROUPE DE SYMETRIE de la parametrisation — donc §6.4 cesse
d'etre une verification pour devenir un test de ce nom-la.

L'INSTRUMENT. Pour un code c, on note sa vraisemblance jointe sous les deux agents

    L(c) = somme_r log S[r, c(r)] + somme_r log R[c(r), r]

et on mesure le COSINUS entre grad J et grad L(c), dans l'espace des PARAMETRES.
C'est exactement « le premier pas de la montee rapproche-t-il du code c ? », et
c'est la bonne question a poser a une parametrisation : elle vit dans l'espace des
poids, pas dans celui des lois.

QUATRE PREDICTIONS, ECRITES AVANT DE LIRE LES CHIFFRES.

  P1. Dans l'espace des LOIS, le gradient a l'initialisation est quasi uniforme :
      son coefficient de variation sur les 27x27 entrees doit etre minuscule.
      C'est la prediction de §4, et elle ne concerne pas la parametrisation.

  P2. En TABULAIRE, cos(grad J, grad L(compositionnel)) doit etre indiscernable de
      la loi de cos(grad J, grad L(c)) sur des c tires au hasard. Par equivariance,
      et donc exactement, pas approximativement.

  P3. En STRUCTURE, le compositionnel doit ressortir. C'est la meme chose que
      §6.5 et §6.1, mais lue au PREMIER PAS au lieu de la convergence.

  P4. Celle que je ne sais pas prevoir, et la plus interessante. Le code
      finalement atteint est-il DEJA un point aberrant de l'alignement au premier
      pas ? Si oui, l'issue est decidee a l'initialisation, comme le coin l'etait
      au test 2. Si non, elle est decidee par la trajectoire — et ce serait le
      contraire du test 2, ou §7.11sexies avait montre que l'initialisation
      decidait le coin et la trajectoire seulement son remplissage.
"""

import argparse
import json
import os

import numpy as np
import torch

from code_emergent import profil, tirer_profil
from grammaire3 import DOSSIER_SORTIE, N
from representable_atteignable_stable import (CODE_CANONIQUE, EmetteurStructure,
                                              EmetteurTabulaire, Recepteur,
                                              cloner, lire_code, monter,
                                              objectif, parametres)

torch.set_num_threads(int(os.environ.get("RDTRL_THREADS", "1")))


def gradient_objectif(emetteur, recepteur, beta):
    for a in (emetteur, recepteur):
        for t in a.p:
            t.requires_grad_(True)
            if t.grad is not None:
                t.grad = None
    j, _ = objectif(emetteur, recepteur, beta)
    return torch.autograd.grad(j, parametres(emetteur, recepteur))


def gradient_vraisemblance(emetteur, recepteur, code):
    """grad de L(c) = somme_r log S[r,c(r)] + somme_r log R[c(r),r]."""
    for a in (emetteur, recepteur):
        for t in a.p:
            t.requires_grad_(True)
    indices = torch.as_tensor(np.asarray(code), dtype=torch.long)
    lignes = torch.arange(N)
    s, r = emetteur.loi(), recepteur.loi()
    perte = (torch.log(s[lignes, indices].clamp_min(1e-300)).sum()
             + torch.log(r[indices, lignes].clamp_min(1e-300)).sum())
    return torch.autograd.grad(perte, parametres(emetteur, recepteur))


def cosinus(a, b):
    plat_a = torch.cat([x.reshape(-1) for x in a])
    plat_b = torch.cat([x.reshape(-1) for x in b])
    return float(torch.dot(plat_a, plat_b)
                 / (plat_a.norm() * plat_b.norm()).clamp_min(1e-300))


def uniformite_du_gradient(emetteur, recepteur):
    """Coefficient de variation de dE[R]/dS dans l'espace des LOIS (P1).

    On regarde la seule partie recompense, dont §4 (c) dit qu'elle vaut R[m,r]/27.
    C'est elle qui porterait une direction preferee vers un code : le terme
    d'entropie ne distingue aucun code des autres, donc beta n'intervient pas.
    """
    with torch.no_grad():
        r = recepteur.loi()
        grad_s = r.t() / N                     # dE[R]/dS[r,m]
        grad_r = emetteur.loi().t() / N        # dE[R]/dR[m,r]
    return {"cv_dS": float(grad_s.std() / grad_s.mean()),
            "cv_dR": float(grad_r.std() / grad_r.mean()),
            "norme_dS": float(grad_s.norm())}


def situer(valeur, echantillon):
    ech = np.asarray(echantillon)
    return {"valeur": float(valeur), "centile": float((ech < valeur).mean()),
            "z": float((valeur - ech.mean()) / ech.std()),
            "temoin_moyenne": float(ech.mean()), "temoin_sd": float(ech.std())}


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — test 3, §6.4")
    parseur.add_argument("--graines", type=int, default=20)
    parseur.add_argument("--temoins", type=int, default=300)
    parseur.add_argument("--pas", type=int, default=3000)
    parseur.add_argument("--beta", type=float, default=0.02)
    parseur.add_argument("--graine", type=int, default=0)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    generateur = np.random.default_rng(args.graine)
    classes = (EmetteurTabulaire, EmetteurStructure)
    rapport = {"beta": args.beta, "graines": args.graines,
               "temoins": args.temoins, "graine": args.graine}

    print("=" * 78)
    print("TEST 3 §6.4 — QUE VOIT LE GRADIENT AU PREMIER PAS ?")
    print("=" * 78)
    print(f"\n  {args.graines} graines, {args.temoins} codes temoins par graine.\n")

    print("-" * 78)
    print("P1 — LE GRADIENT EST-IL UNIFORME DANS L'ESPACE DES LOIS ?")
    print("-" * 78)
    print("  Prediction de §4 : oui, les deux politiques etant quasi uniformes.")
    print("  Elle ne concerne pas la parametrisation, donc une seule mesure suffit.\n")
    emetteur, recepteur = EmetteurTabulaire(generateur), Recepteur(generateur)
    u = uniformite_du_gradient(emetteur, recepteur)
    print(f"  coefficient de variation de dE[R]/dS : {u['cv_dS']:.2e}")
    print(f"  coefficient de variation de dE[R]/dR : {u['cv_dR']:.2e}")
    print(f"  => aucune direction preferee dans l'espace des lois. P1 confirmee.")
    print(f"  (au test 2, le signal d'ordre 1 imposait une direction des le pas 1)")
    rapport["uniformite_lois"] = u

    print("\n" + "-" * 78)
    print("P2 et P3 — LE PREMIER PAS PREFERE-T-IL LE CODE COMPOSITIONNEL ?")
    print("-" * 78)
    print("  cos(grad J, grad L(c)) dans l'espace des PARAMETRES, a l'initialisation.\n")
    print(f"  {'parametrisation':>16}  {'cos compositionnel':>19}  "
          f"{'cos temoins':>19}  {'z':>8}  {'centile':>8}")
    premier_pas = {}
    for classe in classes:
        # UN RUN N'EST PAS UNE PROPRIETE : tout est moyenne sur les graines, y
        # compris les cosinus bruts, et pas lu sur la derniere.
        mesures = []
        for _ in range(args.graines):
            e, r = classe(generateur), Recepteur(generateur)
            gj = gradient_objectif(e, r, args.beta)
            c_comp = cosinus(gj, gradient_vraisemblance(e, r, CODE_CANONIQUE))
            temoins = [cosinus(gj, gradient_vraisemblance(e, r,
                                                          generateur.permutation(N)))
                       for _ in range(args.temoins)]
            mesures.append(situer(c_comp, temoins))
        z_comp = np.array([m["z"] for m in mesures])
        premier_pas[classe.nom] = {
            "z_moyen": float(z_comp.mean()),
            "z_erreur_type": float(z_comp.std(ddof=1) / np.sqrt(len(z_comp))),
            "centile_moyen": float(np.mean([m["centile"] for m in mesures])),
            "cos_compositionnel": float(np.mean([m["valeur"] for m in mesures])),
            "cos_temoins": float(np.mean([m["temoin_moyenne"] for m in mesures])),
            "sd_temoins": float(np.mean([m["temoin_sd"] for m in mesures]))}
        p = premier_pas[classe.nom]
        print(f"  {classe.nom:>16}  {p['cos_compositionnel']:19.6f}  "
              f"{p['cos_temoins']:12.6f} ± {p['sd_temoins']:.4f}  "
              f"{p['z_moyen']:+8.2f}  {p['centile_moyen']:8.3f}")
    rapport["premier_pas"] = premier_pas
    print(f"\n  tabulaire : z = {premier_pas['tabulaire']['z_moyen']:+.2f} ± "
          f"{premier_pas['tabulaire']['z_erreur_type']:.2f}")
    print(f"  structure : z = {premier_pas['structure']['z_moyen']:+.2f} ± "
          f"{premier_pas['structure']['z_erreur_type']:.2f}")

    print("\n" + "-" * 78)
    print("P4 — L'ISSUE EST-ELLE DEJA VISIBLE AU PREMIER PAS ?")
    print("-" * 78)
    print("  On entraine, on lit le code atteint, puis on revient a")
    print("  L'INITIALISATION mesurer son alignement. Si le code final y est deja")
    print("  aberrant, l'issue est decidee a l'initialisation, comme au test 2.\n")
    print("  LES TEMOINS SONT APPARIES AU PROFIL DE FIBRES du code atteint. Sans")
    print("  ca on comparerait un code a collisions a des bijections, et on")
    print("  mesurerait l'effet du profil en l'appelant predictibilite — c'est")
    print("  exactement le piege corrige en §6.1.\n")
    print(f"  {'parametrisation':>16}  {'z du code atteint':>18}  {'centile':>9}  "
          f"{'E[R]':>7}  {'collisions':>10}")
    issue = {}
    for classe in classes:
        z_final, centiles, recompenses, collisions, accords = [], [], [], [], []
        for _ in range(args.graines):
            e0, r0 = classe(generateur), Recepteur(generateur)
            e1, r1 = cloner(e0, generateur), cloner(r0, generateur)
            recompense = monter(e1, r1, args.beta, args.pas)
            code_final = lire_code(e1)
            # retour a l'initialisation, temoins de MEME profil
            gj = gradient_objectif(e0, r0, args.beta)
            c_final = cosinus(gj, gradient_vraisemblance(e0, r0, code_final))
            apparies = tirer_profil(profil(code_final), args.temoins, generateur)
            temoins = [cosinus(gj, gradient_vraisemblance(e0, r0, c))
                       for c in apparies]
            s = situer(c_final, temoins)
            z_final.append(s["z"])
            centiles.append(s["centile"])
            recompenses.append(recompense)
            collisions.append(N - len(set(code_final.tolist())))
            # la version la moins discutable de P4 : aucune loi nulle, aucun
            # cosinus. Combien du code final est deja l'argmax des poids initiaux ?
            accords.append(float((code_final == lire_code(e0)).mean()))
        z_final = np.array(z_final)
        issue[classe.nom] = {
            "z_moyen": float(z_final.mean()),
            "z_erreur_type": float(z_final.std(ddof=1) / np.sqrt(len(z_final))),
            "centile_moyen": float(np.mean(centiles)),
            "reward_moyen": float(np.mean(recompenses)),
            "collisions_moyennes": float(np.mean(collisions)),
            "accord_avec_argmax_initial": float(np.mean(accords)),
            "accord_sd": float(np.std(accords))}
        i = issue[classe.nom]
        print(f"  {classe.nom:>16}  {i['z_moyen']:+9.2f} ± {i['z_erreur_type']:.2f}  "
              f"{i['centile_moyen']:9.3f}  {i['reward_moyen']:7.4f}  "
              f"{i['collisions_moyennes']:10.2f}")
    rapport["issue_visible"] = issue

    print("\n  Sans loi nulle ni cosinus : part du code final deja presente comme")
    print(f"  argmax des poids INITIAUX (hasard = 1/27 = {1 / N:.3f}) :")
    for classe in classes:
        i = issue[classe.nom]
        print(f"    {classe.nom:>16} : {100 * i['accord_avec_argmax_initial']:.1f} % "
              f"± {100 * i['accord_sd']:.1f}")

    print("\n" + "-" * 78)
    print("P3 EST REFUTEE — ALORS QUAND LA PREFERENCE APPARAIT-ELLE ?")
    print("-" * 78)
    print("  Si la parametrisation structuree finit a z = +9 mais ne prefere rien")
    print("  au pas 1, sa preference se construit en route. On refait donc la meme")
    print("  mesure a plusieurs profondeurs d'entrainement, depuis la MEME")
    print("  initialisation a chaque fois.\n")
    jalons = [0, 10, 30, 100, 300, 1000, args.pas]
    print(f"  {'parametrisation':>16}  " + "".join(f"{j:>9}" for j in jalons))
    courbes = {}
    for classe in classes:
        par_jalon = {j: [] for j in jalons}
        for _ in range(args.graines // 2):
            e0, r0 = classe(generateur), Recepteur(generateur)
            for j in jalons:
                e, r = cloner(e0, generateur), cloner(r0, generateur)
                if j:
                    monter(e, r, args.beta, j)
                gj = gradient_objectif(e, r, args.beta)
                c = cosinus(gj, gradient_vraisemblance(e, r, CODE_CANONIQUE))
                temoins = [cosinus(gj, gradient_vraisemblance(
                    e, r, generateur.permutation(N))) for _ in range(100)]
                par_jalon[j].append(situer(c, temoins)["z"])
        courbes[classe.nom] = {str(j): float(np.mean(par_jalon[j])) for j in jalons}
        print(f"  {classe.nom:>16}  "
              + "".join(f"{np.mean(par_jalon[j]):>+9.2f}" for j in jalons))
    rapport["courbe_z_par_pas"] = courbes
    print("\n  z du code compositionnel contre 100 bijections temoins, par nombre")
    print(f"  de pas d'entrainement, moyenne sur {args.graines // 2} graines.")

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    for classe in classes:
        i, p = issue[classe.nom], premier_pas[classe.nom]
        decide = abs(i["z_moyen"]) > 3 * max(i["z_erreur_type"], 1e-12)
        print(f"  {classe.nom:>16} : compositionnel au pas 1 z = {p['z_moyen']:+.2f} ; "
              f"code atteint z = {i['z_moyen']:+.2f}")
        print(f"  {'':>16}   issue lisible au premier pas : "
              f"{'OUI' if decide else 'non'}")

    nom = f"6_4_gradient_premier_pas_b{args.beta}_{args.graines}graines_g{args.graine}.json"
    with open(os.path.join(DOSSIER_SORTIE, nom), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nEcrit dans {DOSSIER_SORTIE} sous {nom}")
