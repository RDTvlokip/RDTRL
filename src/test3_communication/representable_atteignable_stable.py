"""
RDTRL — Test 3, §6.5 : representable, atteignable, stable, les trois separement.

Au test 2 ces trois reponses etaient DIFFERENTES, et c'est ce qui a fait basculer
le verdict : le modele pouvait representer l'optimum, pouvait a peu pres s'y
maintenir, et ne pouvait pas l'atteindre. Trois echecs qui portent le meme nom et
appellent des remedes opposes.

§6.7 a change ce que cette etape doit mesurer. Il y est etabli que sous une
parametrisation TABULAIRE, la montee est equivariante sous le renommage des
messages, lequel agit transitivement sur les 27! bijections. Donc :

    PREDICTION ENREGISTREE AVANT MESURE. En tabulaire, le code compositionnel se
    comporte EXACTEMENT comme une bijection tiree au hasard — meme vitesse
    d'ajustement supervise, meme stabilite. Pas « a peu pres » : a la precision
    machine, puisque les deux codes sont relies par un renommage et que la
    dynamique commute avec les renommages.

C'est ce qui rend l'etape decisive plutot que descriptive : si l'ecart est nul en
tabulaire et non nul en structure, alors le sens de l'ecart est le mecanisme que
§6.2 cherche, et il est mesurable SANS entrainer quoi que ce soit depuis zero.

LE CONTRASTE EST CONSTRUIT POUR N'AVOIR QU'UNE SEULE DIFFERENCE. Meme objectif,
meme optimiseur, meme recepteur tabulaire, et surtout MEME EXPRESSIVITE : la
parametrisation factorisee p(m1) p(m2|m1) p(m3|m1,m2) est tabulaire a chaque
etage, donc elle represente exactement les memes lois que la matrice 27x27. Seule
la carte des parametres change. C'est la lecon de §6.7 : ce qui compte n'est pas
ce que la parametrisation peut ecrire, c'est le groupe qui agit sur ses poids.
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from grammaire3 import (DOSSIER_SORTIE, INDEX_MESSAGE, N, N_ATTRIBUTS,
                        N_POSITIONS, N_TOKENS, N_VALEURS, REFERENTS,
                        concentration, concentration_appariee,
                        matrice_information)

torch.set_num_threads(int(os.environ.get("RDTRL_THREADS", "1")))

CODE_CANONIQUE = np.array([INDEX_MESSAGE[ref] for ref in REFERENTS])


class EmetteurTabulaire:
    """S[r, m] = softmax sur les 27 messages. Groupe de symetrie : S_27 entier."""

    nom = "tabulaire"

    def __init__(self, generateur, echelle=0.01):
        g = torch.Generator().manual_seed(int(generateur.integers(1 << 30)))
        self.p = [echelle * torch.randn(N, N, generator=g, dtype=torch.float64)]

    def loi(self):
        return torch.softmax(self.p[0], dim=1)

    def poser_code(self, code, force=8.0):
        with torch.no_grad():
            self.p[0].zero_()
            for r, m in enumerate(code):
                self.p[0][r, m] = force


class EmetteurFactorise:
    """S[r, m] = p(m1|r) p(m2|r,m1) p(m3|r,m1,m2). MEME expressivite que tabulaire.

    Chaque etage est tabulaire, donc toute loi sur les 27 messages est atteignable :
    la difference avec la classe ci-dessus n'est pas ce qui est representable, c'est
    la carte des parametres, donc le groupe de renommages qui agit dessus.
    """

    nom = "factorise"

    def __init__(self, generateur, echelle=0.01):
        g = torch.Generator().manual_seed(int(generateur.integers(1 << 30)))
        formes = [(N, N_TOKENS), (N, N_TOKENS, N_TOKENS),
                  (N, N_TOKENS, N_TOKENS, N_TOKENS)]
        self.p = [echelle * torch.randn(*f, generator=g, dtype=torch.float64)
                  for f in formes]

    def loi(self):
        p1 = torch.softmax(self.p[0], dim=-1)
        p2 = torch.softmax(self.p[1], dim=-1)
        p3 = torch.softmax(self.p[2], dim=-1)
        return (p1[:, :, None, None] * p2[:, :, :, None] * p3).reshape(N, N)

    def poser_code(self, code, force=8.0):
        with torch.no_grad():
            for tenseur in self.p:
                tenseur.zero_()
            for r, m in enumerate(code):
                m1, m2, m3 = m // 9, (m // 3) % 3, m % 3
                self.p[0][r, m1] = force
                self.p[1][r, m1, m2] = force
                self.p[2][r, m1, m2, m3] = force


class EmetteurStructure:
    """Le referent entre par ses ATTRIBUTS, et les poids sont PARTAGES.

    Trouve en construisant ce fichier, et ca corrige §6.7. Les deux classes
    ci-dessus indexent leurs parametres par referent, sans aucun partage : elles
    sont donc equivariantes sous le renommage des REFERENTS, c -> c o rho^-1, qui
    est lui aussi transitif sur les 27! bijections. §6.7 raisonnait sur le cote
    message ; l'equivariance d'un SEUL des deux cotes suffit a egaliser tous les
    codes. Mesure : l'ecart compositionnel/aleatoire y vaut 3,3e-16.

    Pour que la structure puisse compter, il faut donc que le referent cesse
    d'etre un indice opaque. Ici :

        logit du token t en position j = somme_{i,v} W[j,t,i,v] . 1[attribut i = v]

    soit 81 poids partages par les 27 referents, contre 729 libres en tabulaire.
    Un code compositionnel s'ecrit exactement dans cette forme. Une bijection
    quelconque, en general, non : c'est §6.5 question 1 qui cesse d'etre triviale.
    """

    nom = "structure"

    def __init__(self, generateur, echelle=0.01):
        g = torch.Generator().manual_seed(int(generateur.integers(1 << 30)))
        self.p = [echelle * torch.randn(N_POSITIONS, N_TOKENS, N_ATTRIBUTS,
                                        N_VALEURS, generator=g, dtype=torch.float64),
                  echelle * torch.randn(N_POSITIONS, N_TOKENS,
                                        generator=g, dtype=torch.float64)]
        # ENCODAGE[r, i, v] = 1 si l'attribut i du referent r vaut v
        encodage = torch.zeros(N, N_ATTRIBUTS, N_TOKENS, dtype=torch.float64)
        for r, referent in enumerate(REFERENTS):
            for i, v in enumerate(referent):
                encodage[r, i, v] = 1.0
        self.encodage = encodage

    def loi(self):
        logits = torch.einsum("riv,jtiv->rjt", self.encodage, self.p[0]) + self.p[1]
        p = torch.softmax(logits, dim=-1)          # (referents, positions, tokens)
        return (p[:, 0, :, None, None] * p[:, 1, None, :, None]
                * p[:, 2, None, None, :]).reshape(N, N)

    def poser_code(self, code, force=8.0):
        """N'est exact que pour un code compositionnel — c'est precisement le point.

        On pose le code canonique (token j = attribut j). Pour tout autre code, le
        reglage exact n'existe pas en general, et c'est l'ajustement supervise de
        la section 1 qui dit jusqu'ou on peut s'en approcher.
        """
        with torch.no_grad():
            for tenseur in self.p:
                tenseur.zero_()
            if np.array_equal(np.asarray(code), CODE_CANONIQUE):
                for j in range(N_POSITIONS):
                    for v in range(N_TOKENS):
                        self.p[0][j, v, j, v] = force
                return False          # pose exactement, pas besoin d'ajuster
            return True               # a ajuster par descente


class Recepteur:
    """Tabulaire dans tous les cas : la seule difference doit etre l'emetteur."""

    def __init__(self, generateur, echelle=0.01):
        g = torch.Generator().manual_seed(int(generateur.integers(1 << 30)))
        self.p = [echelle * torch.randn(N, N, generator=g, dtype=torch.float64)]

    def loi(self):
        return torch.softmax(self.p[0], dim=1)

    def poser_code(self, code, force=8.0):
        with torch.no_grad():
            self.p[0].zero_()
            for r, m in enumerate(code):
                self.p[0][m, r] = force


def parametres(*agents):
    return [t for a in agents for t in a.p]


def activer(*agents):
    for a in agents:
        for t in a.p:
            t.requires_grad_(True)


def objectif(emetteur, recepteur, beta):
    s, r = emetteur.loi(), recepteur.loi()
    recompense = (s * r.t()).sum() / N
    entropie_s = -(s * torch.log(s.clamp_min(1e-300))).sum() / N
    entropie_r = -(r * torch.log(r.clamp_min(1e-300))).sum() / N
    return recompense + beta * (entropie_s + entropie_r), recompense


def monter(emetteur, recepteur, beta, pas, lr=0.05):
    activer(emetteur, recepteur)
    optimiseur = torch.optim.Adam(parametres(emetteur, recepteur), lr=lr)
    for _ in range(pas):
        j, _ = objectif(emetteur, recepteur, beta)
        optimiseur.zero_grad()
        (-j).backward()
        optimiseur.step()
    with torch.no_grad():
        _, recompense = objectif(emetteur, recepteur, beta)
    return float(recompense)


def ajuster(emetteur, recepteur, code, pas, lr=0.05, cible=0.999):
    """Sonde de capacite : entropie croisee vers un code impose. Purement diagnostique.

    Rend le nombre de pas pour depasser `cible` en E[R], et la recompense finale.
    C'est la vitesse d'ajustement qui porte l'information, pas le fait d'y arriver :
    les deux parametrisations ont la meme expressivite, donc les deux y arrivent.
    """
    activer(emetteur, recepteur)
    optimiseur = torch.optim.Adam(parametres(emetteur, recepteur), lr=lr)
    indices = torch.as_tensor(np.asarray(code), dtype=torch.long)
    lignes = torch.arange(N)
    atteint = None
    for etape in range(pas):
        s, r = emetteur.loi(), recepteur.loi()
        perte = -(torch.log(s[lignes, indices].clamp_min(1e-300)).sum()
                  + torch.log(r[indices, lignes].clamp_min(1e-300)).sum()) / N
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        if atteint is None:
            with torch.no_grad():
                if float((emetteur.loi() * recepteur.loi().t()).sum() / N) > cible:
                    atteint = etape + 1
    with torch.no_grad():
        finale = float((emetteur.loi() * recepteur.loi().t()).sum() / N)
        ecart = float((emetteur.loi()[lignes, indices] - 1.0).abs().max())
    return {"pas_pour_atteindre": atteint, "reward_final": finale,
            "ecart_max_a_la_cible": ecart}


def reinforce(emetteur, recepteur, beta, pas, lot=64, lr=0.01, graine=0):
    """REINFORCE echantillonne, chemin d'avantage float64 (canonique depuis 0.4.0).

    C'est la version qui compte pour la stabilite : au test 2, la separation entre
    exact et echantillonne portait tout le resultat.
    """
    activer(emetteur, recepteur)
    optimiseur = torch.optim.Adam(parametres(emetteur, recepteur), lr=lr)
    g = torch.Generator().manual_seed(graine)
    baseline = 0.0
    for _ in range(pas):
        s, r = emetteur.loi(), recepteur.loi()
        referents = torch.randint(0, N, (lot,), generator=g)
        messages = torch.multinomial(s[referents], 1, generator=g).squeeze(1)
        reconstruits = torch.multinomial(r[messages], 1, generator=g).squeeze(1)
        recompenses = (reconstruits == referents).double()
        moyenne = float(recompenses.mean())
        avantages = torch.tensor([float(x) - baseline for x in recompenses],
                                 dtype=torch.float64)
        baseline = 0.9 * baseline + 0.1 * moyenne
        log_s = torch.log(s[referents, messages].clamp_min(1e-300))
        log_r = torch.log(r[messages, reconstruits].clamp_min(1e-300))
        entropie = -(s * torch.log(s.clamp_min(1e-300))).sum() / N \
                   - (r * torch.log(r.clamp_min(1e-300))).sum() / N
        perte = -((log_s + log_r) * avantages).mean() - beta * entropie
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
    with torch.no_grad():
        _, recompense = objectif(emetteur, recepteur, beta)
    return float(recompense)


def cloner(agent, generateur):
    """Copie d'un agent ajuste, pour lancer deux dynamiques depuis le MEME etat."""
    copie = type(agent)(generateur)
    with torch.no_grad():
        for cible, source in zip(copie.p, agent.p):
            cible.copy_(source)
    return copie


def lire_code(emetteur):
    with torch.no_grad():
        return emetteur.loi().argmax(dim=1).numpy()


def decrire(code):
    """Concentration par le chemin GENERAL : un code atteint n'est pas bijectif."""
    matrice = matrice_information(np.asarray(code))
    return {"bijectif": bool(len(np.unique(code)) == N),
            "collisions": int(N - len(np.unique(code))),
            "concentration_max": concentration(matrice=matrice),
            "concentration_appariee": concentration_appariee(matrice=matrice)}


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — test 3, §6.5")
    parseur.add_argument("--graines", type=int, default=10)
    parseur.add_argument("--pas", type=int, default=3000)
    parseur.add_argument("--pas-reinforce", type=int, default=4000)
    parseur.add_argument("--beta", type=float, default=0.02)
    parseur.add_argument("--graine", type=int, default=0)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    generateur = np.random.default_rng(args.graine)
    classes = (EmetteurTabulaire, EmetteurFactorise, EmetteurStructure)
    rapport = {"beta": args.beta, "graines": args.graines, "pas": args.pas,
               "graine": args.graine}

    print("=" * 78)
    print("TEST 3 §6.5 — REPRESENTABLE, ATTEIGNABLE, STABLE")
    print("=" * 78)
    print(f"\n  beta = {args.beta} (sous le seuil 1/27 de §6.7, donc regime de code)")
    print("  PREDICTION de §6.7 : en tabulaire, compositionnel et aleatoire sont")
    print("  identiques a la precision machine. En factorise, rien ne l'impose.\n")

    codes_temoins = {"compositionnel": CODE_CANONIQUE}
    for i in range(3):
        codes_temoins[f"aleatoire_{i}"] = generateur.permutation(N)

    print("-" * 78)
    print("1. REPRESENTABLE — ajustement supervise vers un code impose")
    print("-" * 78)
    print("  Sonde de capacite, purement diagnostique. Les deux premieres")
    print("  parametrisations ont l'expressivite pleine ; la troisieme, non.\n")
    print(f"  {'parametrisation':>16}  {'code':>16}  {'pas':>6}  {'E[R]':>9}  "
          f"{'ecart max':>10}")
    representable, ajustes = [], {}
    for classe in classes:
        for nom, code in codes_temoins.items():
            emetteur, recepteur = classe(generateur), Recepteur(generateur)
            res = ajuster(emetteur, recepteur, code, args.pas)
            res.update({"parametrisation": classe.nom, "code": nom})
            representable.append(res)
            ajustes[(classe.nom, nom)] = (emetteur, recepteur)
            print(f"  {classe.nom:>16}  {nom:>16}  {str(res['pas_pour_atteindre']):>6}  "
                  f"{res['reward_final']:9.5f}  {res['ecart_max_a_la_cible']:10.2e}")
    rapport["representable"] = representable

    print()
    for classe in classes:
        lignes = [r for r in representable if r["parametrisation"] == classe.nom]
        comp = next(r for r in lignes if r["code"] == "compositionnel")
        alea = [r for r in lignes if r["code"] != "compositionnel"]
        moyenne_alea = float(np.mean([r["reward_final"] for r in alea]))
        print(f"  {classe.nom:>16} : compositionnel {comp['reward_final']:.5f}, "
              f"aleatoire {moyenne_alea:.5f}, ecart {comp['reward_final'] - moyenne_alea:+.2e}")

    print("\n" + "-" * 78)
    print("2. STABLE — on repart de l'etat ajuste, et on regarde si on y reste")
    print("-" * 78)
    print("  Exact puis REINFORCE echantillonne, depuis le MEME etat ajuste.\n")
    print(f"  {'parametrisation':>16}  {'code':>16}  {'E[R] exact':>11}  "
          f"{'E[R] REINFORCE':>15}  {'conserve':>9}")
    stable = []
    for classe in classes:
        for nom, code in codes_temoins.items():
            emetteur, recepteur = ajustes[(classe.nom, nom)]
            e1, r1 = cloner(emetteur, generateur), cloner(recepteur, generateur)
            exact = monter(e1, r1, args.beta, args.pas)
            e2, r2 = cloner(emetteur, generateur), cloner(recepteur, generateur)
            echantillonne = reinforce(e2, r2, args.beta, args.pas_reinforce,
                                      graine=args.graine)
            conserve = bool(np.array_equal(lire_code(e2), np.asarray(code)))
            stable.append({"parametrisation": classe.nom, "code": nom,
                           "reward_exact": exact,
                           "code_conserve_exact": bool(np.array_equal(
                               lire_code(e1), np.asarray(code))),
                           "reward_reinforce": echantillonne,
                           "code_conserve_reinforce": conserve})
            print(f"  {classe.nom:>16}  {nom:>16}  {exact:11.6f}  "
                  f"{echantillonne:15.6f}  {'oui' if conserve else 'NON':>9}")
    rapport["stable"] = stable

    for cle, etiquette in (("reward_exact", "exact"), ("reward_reinforce", "REINFORCE")):
        print()
        for classe in classes:
            lignes = [s for s in stable if s["parametrisation"] == classe.nom]
            comp = next(s for s in lignes if s["code"] == "compositionnel")[cle]
            alea = [s[cle] for s in lignes if s["code"] != "compositionnel"]
            print(f"  {etiquette:>9}, {classe.nom:>16} : compositionnel {comp:.8f}, "
                  f"aleatoire {np.mean(alea):.8f}, ecart {comp - np.mean(alea):+.2e}")

    print("\n" + "-" * 78)
    print("3. ATTEIGNABLE — depuis l'aleatoire, ou se pose-t-on ?")
    print("-" * 78)
    print(f"  {args.graines} graines par parametrisation, montee exacte.\n")
    print(f"  {'parametrisation':>16}  {'E[R] moyen':>11}  {'bijections':>11}  "
          f"{'collisions':>11}  {'concentration appariee':>23}")
    atteignable = []
    for classe in classes:
        resultats = []
        for _ in range(args.graines):
            emetteur, recepteur = classe(generateur), Recepteur(generateur)
            recompense = monter(emetteur, recepteur, args.beta, args.pas)
            info = decrire(lire_code(emetteur))
            info["reward"] = recompense
            info["parametrisation"] = classe.nom
            resultats.append(info)
        atteignable.extend(resultats)
        recompenses = np.array([r["reward"] for r in resultats])
        bijections = sum(r["bijectif"] for r in resultats)
        collisions = np.array([r["collisions"] for r in resultats])
        conc = np.array([r["concentration_appariee"] for r in resultats])
        print(f"  {classe.nom:>16}  {recompenses.mean():11.4f}  "
              f"{bijections:>4} / {len(resultats):<4}  "
              f"{collisions.mean():11.2f}  {conc.mean():13.4f} ± {conc.std():.4f}")
    rapport["atteignable"] = atteignable

    reference = np.array([r["concentration_appariee"] for r in atteignable
                          if r["parametrisation"] == "tabulaire"])
    print()
    for classe in classes[1:]:
        autre = np.array([r["concentration_appariee"] for r in atteignable
                          if r["parametrisation"] == classe.nom])
        print(f"  ecart de concentration {classe.nom} - tabulaire : "
              f"{autre.mean() - reference.mean():+.4f}")
    print(f"\n  rappel, loi nulle appariee SUR DES BIJECTIONS : 0,1168 ± 0,0315")
    print("  (les codes atteints n'etant pas bijectifs, cette ligne n'est PAS une")
    print("  reference valide : c'est la correction annoncee en §6.7, encore due)")

    nom = f"6_5_representable_atteignable_stable_b{args.beta}_g{args.graine}.json"
    with open(os.path.join(DOSSIER_SORTIE, nom), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nEcrit dans {DOSSIER_SORTIE} sous {nom}")
