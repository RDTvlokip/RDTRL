"""
RDTRL — Test 2 : apprendre une grammaire en RL pur.

Difference avec le test 1 : la recompense ne vient plus d'une comparaison a une
phrase cible, mais d'un parser de regles ecrit a la main (grammaire.py) qui
ignore totalement ce que l'agent va produire. Il existe des dizaines de phrases
valides, pas une seule bonne reponse.

Ce que le script mesure :
  - le taux de validite grammaticale atteint ;
  - la diversite des phrases produites (l'agent explore-t-il les 48 solutions
    ou se fige-t-il sur une seule ?) ;
  - la structure conditionnelle apprise (P(nom | det), P(verbe | nom)) ;
  - la generalisation a une combinaison jamais recompensee ;
  - l'effet de la decomposition de la recompense (controle tout-ou-rien),
    a deux tailles d'espace de recherche differentes.
"""

import argparse
import csv
import json
import os
import random
import time
from collections import deque, Counter

import numpy as np
import torch
import torch.nn as nn

# Le nombre de threads change l'ordre des reductions flottantes, donc les
# derniers bits, donc parfois la trajectoire entiere : distribution.sample() est
# un seuil sur un tirage uniforme, il suffit qu'un token bascule une fois. On
# epingle donc a 1 ici, dans le module que tous les scripts importent, plutot
# que fichier par fichier ou depuis le shell. RDTRL_THREADS permet de revenir en
# arriere pour les rares calculs a gros lot qui y gagnent (gradient exact).
torch.set_num_threads(int(os.environ.get("RDTRL_THREADS", "1")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grammaire import Grammaire

# Les sorties vont a la racine du depot, pas a cote du script.
RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOSSIER_SORTIE = os.path.join(RACINE, "results_test2")


class PolitiqueGRU(nn.Module):
    """GRU 1 couche generant un token (un mot entier) a la fois.

    Poids aleatoires, aucun pre-entrainement. L'agent ne recoit jamais les
    traits grammaticaux : un token est un simple index.
    """

    def __init__(self, taille_vocab, taille_cachee=128, taille_embedding=32):
        super().__init__()
        self.taille_vocab = taille_vocab
        self.taille_cachee = taille_cachee
        self.token_debut = taille_vocab
        self.embedding = nn.Embedding(taille_vocab + 1, taille_embedding)
        self.gru = nn.GRU(taille_embedding, taille_cachee, num_layers=1, batch_first=True)
        self.tete = nn.Linear(taille_cachee, taille_vocab)

    def generer(self, longueur, taille_lot=1, greedy=False, masque_fn=None,
                forcer=None, device="cpu"):
        """Deroule une phrase complete.

        masque_fn(t, actions_deja) -> tenseur booleen [lot, vocab], True = autorise.
            Sert a interdire certaines combinaisons pendant l'entrainement.
        forcer : dict {position: index de token} pour imposer un mot a
            l'evaluation (test de generalisation conditionnelle).
        """
        h = torch.zeros(1, taille_lot, self.taille_cachee, device=device)
        entree = torch.full((taille_lot,), self.token_debut, dtype=torch.long, device=device)

        actions, log_probs, entropies, probabilites = [], [], [], []
        for t in range(longueur):
            emb = self.embedding(entree).unsqueeze(1)
            sortie, h = self.gru(emb, h)
            logits = self.tete(sortie.squeeze(1))

            if masque_fn is not None:
                autorise = masque_fn(t, torch.stack(actions, dim=1) if actions else None)
                if autorise is not None:
                    logits = logits.masked_fill(~autorise, float("-inf"))

            distribution = torch.distributions.Categorical(logits=logits)

            if forcer is not None and t in forcer:
                action = torch.full((taille_lot,), forcer[t], dtype=torch.long, device=device)
            elif greedy:
                action = logits.argmax(dim=-1)
            else:
                action = distribution.sample()

            actions.append(action)
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            probabilites.append(distribution.probs)
            entree = action

        return (torch.stack(actions, dim=1),
                torch.stack(log_probs, dim=1),
                torch.stack(entropies, dim=1),
                torch.stack(probabilites, dim=1))


def fixer_graine(graine):
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)


def nouvelle_politique(grammaire, graine, device="cpu"):
    fixer_graine(graine)
    return PolitiqueGRU(grammaire.taille).to(device)


def entrainer(politique, grammaire, max_episodes=20000, type_recompense="graduee",
              lr=1e-3, coef_entropie=0.01, taille_lot=1, fenetre=100,
              masque_fn=None, verbeux=True, periode_log=100, device="cpu",
              etiquette="", chemin_avantage="float32"):
    """REINFORCE avec baseline. La recompense vient du parser de regles.

    chemin_avantage : ou se fait la soustraction recompense - baseline.
        "float32"  recompenses_t est float32 et baseline un flottant Python, donc
                   la promotion arrondit la baseline AVANT de soustraire : deux
                   arrondis. C'est le comportement historique de ce fichier, et
                   celui qui a produit tous les chiffres publies.
        "float64"  soustraction en double precision puis un seul arrondi. Plus
                   exact, et c'est ce que font les quatre autres scripts du
                   depot ; le defaut reste "float32" pour ne rien changer en
                   silence a des resultats deja archives.
        Les deux ne different que dans les derniers bits, mais un seul bit suffit
        a faire basculer un tirage, donc a changer la trajectoire entiere.
    """
    fonction = (grammaire.recompense_graduee if type_recompense == "graduee"
                else grammaire.recompense_tout_ou_rien)
    optimiseur = torch.optim.Adam(politique.parameters(), lr=lr)

    historique_baseline = deque(maxlen=fenetre)
    historique_validite = deque(maxlen=fenetre)
    recompenses_log, moyennes_log, validites_log = [], [], []
    episode = 0
    premier_valide = None
    debut = time.time()

    while episode < max_episodes:
        actions, log_probs, entropies, _ = politique.generer(
            grammaire.longueur, taille_lot=taille_lot, masque_fn=masque_fn, device=device)

        liste = actions.tolist()
        recompenses = [fonction(seq) for seq in liste]
        valides = [grammaire.analyser(seq)["valide"] for seq in liste]

        baseline = (sum(historique_baseline) / len(historique_baseline)
                    if historique_baseline else 0.0)
        if chemin_avantage == "float32":
            recompenses_t = torch.tensor(recompenses, dtype=torch.float32, device=device)
            avantages = (recompenses_t - baseline).detach()
        else:
            avantages = torch.tensor([r - baseline for r in recompenses],
                                     dtype=torch.float32, device=device).detach()

        perte = -(log_probs.sum(dim=1) * avantages).mean() \
                - coef_entropie * entropies.sum(dim=1).mean()

        optimiseur.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(politique.parameters(), 5.0)
        optimiseur.step()

        for i, r in enumerate(recompenses):
            episode += 1
            historique_baseline.append(r)
            historique_validite.append(1.0 if valides[i] else 0.0)
            recompenses_log.append(r)
            moyennes_log.append(float(np.mean(recompenses_log[-100:])))
            validites_log.append(float(np.mean(historique_validite)))
            if premier_valide is None and valides[i]:
                premier_valide = episode
            if verbeux and episode % periode_log == 0:
                print(f"  [{etiquette}] ep {episode:6d} | reward moyen (100) = "
                      f"{moyennes_log[-1]:.4f} | phrases valides (100) = "
                      f"{100*validites_log[-1]:5.1f} %")

    return {
        "etiquette": etiquette,
        "type_recompense": type_recompense,
        "grammaire": "longue" if grammaire.longue else "courte",
        "episodes": episode,
        "premier_valide": premier_valide,
        "reward_final": float(np.mean(recompenses_log[-100:])),
        "validite_finale_pct": 100 * float(np.mean(list(historique_validite))),
        "duree_s": round(time.time() - debut, 1),
        "recompenses_log": recompenses_log,
        "moyennes_log": moyennes_log,
        "validites_log": validites_log,
    }


def evaluer(politique, grammaire, n=500, device="cpu"):
    """Genere n phrases par echantillonnage et mesure validite + diversite."""
    with torch.no_grad():
        actions, _, _, _ = politique.generer(grammaire.longueur, taille_lot=n, device=device)
    phrases = [" ".join(grammaire.tokens[i] for i in seq) for seq in actions.tolist()]
    valides = [p for p, seq in zip(phrases, actions.tolist())
               if grammaire.analyser(seq)["valide"]]

    compteur = Counter(phrases)
    distinctes_valides = set(valides)
    total_valides_possibles = grammaire.compter_phrases_valides()

    # Entropie empirique de la distribution de phrases (en bits)
    probas = np.array(list(compteur.values()), dtype=float) / n
    entropie = float(-(probas * np.log2(probas)).sum())

    return {
        "n": n,
        "validite_pct": 100 * len(valides) / n,
        "validite_hasard_pct": 100 * grammaire.probabilite_hasard(),
        "phrases_distinctes": len(compteur),
        "phrases_valides_distinctes": len(distinctes_valides),
        "phrases_valides_possibles": total_valides_possibles,
        "couverture_pct": 100 * len(distinctes_valides) / total_valides_possibles,
        "entropie_bits": round(entropie, 3),
        "top5": compteur.most_common(5),
        "exemples_valides": sorted(distinctes_valides)[:8],
    }


def distribution_exacte(politique, grammaire, device="cpu"):
    """Probabilite exacte de chacune des taille^longueur phrases possibles.

    Enumerable sur la grammaire courte (8 000 sequences) : on supprime ainsi
    tout bruit d'echantillonnage sur les mesures de diversite. Le scoring se
    fait en teacher forcing, strictement equivalent au deroule autoregressif
    puisque l'entree du pas t est le token reellement present au pas t-1.
    """
    from itertools import product as _product
    sequences = torch.tensor(list(_product(range(grammaire.taille),
                                           repeat=grammaire.longueur)),
                             dtype=torch.long, device=device)
    debut = torch.full((len(sequences), 1), politique.token_debut,
                       dtype=torch.long, device=device)
    entrees = torch.cat([debut, sequences[:, :-1]], dim=1)
    with torch.no_grad():
        sorties, _ = politique.gru(politique.embedding(entrees))
        log_probas = torch.log_softmax(politique.tete(sorties), dim=-1)
        log_p = log_probas.gather(2, sequences.unsqueeze(-1)).squeeze(-1).sum(1)
    return sequences, log_p.exp()


def analyse_exacte(politique, grammaire, device="cpu"):
    """Mesure la repartition reelle de la politique sur les phrases valides.

    La couverture (nombre de phrases distinctes tirees) ne dit rien sur
    l'uniformite : 40 phrases dont une a 60 % donnent la meme couverture qu'une
    distribution plate. On mesure donc directement l'entropie de la politique
    RESTREINTE aux solutions valides, et le nombre de modes effectifs 2^H, qui
    repond a "combien de phrases valides l'agent utilise-t-il vraiment".
    """
    sequences, probas = distribution_exacte(politique, grammaire, device)
    valide = torch.tensor([grammaire.analyser(s)["valide"] for s in sequences.tolist()],
                          device=device)

    masse_valide = float(probas[valide].sum())
    nb_solutions = int(valide.sum())
    p = probas[valide] / max(probas[valide].sum(), torch.tensor(1e-12, device=device))
    entropie = float(-(p * p.clamp_min(1e-12).log2()).sum())
    entropie_max = float(np.log2(nb_solutions))
    modes_effectifs = float(2 ** entropie)

    ordre = torch.argsort(p, descending=True)
    top = [(" ".join(grammaire.tokens[i] for i in sequences[valide][j].tolist()),
            round(float(p[j]), 4)) for j in ordre[:5].tolist()]

    # Conditionnelles exactes, obtenues en marginalisant la loi jointe
    V = grammaire.taille
    jointe = probas.reshape([V] * grammaire.longueur)
    i_det, i_nom, i_verbe = (grammaire.positions["det"], grammaire.positions["nom"],
                             grammaire.positions["verbe"])
    cond_det, cond_nom = {}, {}
    # Entropie conditionnelle H(nom | determinant) : separe la diversite A
    # L'INTERIEUR d'une branche de la couverture ENTRE branches. Un determinant
    # peut etre bien exploite (H proche de son maximum) alors que la politique
    # ne l'emet presque jamais, et inversement.
    entropie_nom_sachant_det, masse_det = {}, {}
    for det in grammaire.tokens_par_categorie["det"]:
        bloc = jointe[grammaire.index[det]]                       # [nom, verbe]
        total = float(bloc.sum())
        traits_det = grammaire.traits(det)
        masse = sum(float(bloc[grammaire.index[n]].sum())
                    for n in grammaire.tokens_par_categorie["nom"]
                    if all(grammaire._compatible(traits_det[t], grammaire.traits(n)[t])
                           for t in ("genre", "nombre")))
        cond_det[det] = round(masse / total, 4) if total > 0 else None

        masse_det[det] = round(total, 5)
        noms_compatibles = [n for n in grammaire.tokens_par_categorie["nom"]
                            if all(grammaire._compatible(traits_det[t], grammaire.traits(n)[t])
                                   for t in ("genre", "nombre"))]
        if total <= 1e-9:
            entropie_nom_sachant_det[det] = None
            continue
        # Deux quantites que l'ancienne version confondait dans un seul ratio.
        # H sur les 8 noms rapporte a log2(noms compatibles) pouvait depasser
        # 100 %, et une valeur > 100 % signalait une FUITE de masse sur des noms
        # incompatibles, donc un echec, alors qu'elle se lisait comme un succes.
        # On separe : la masse accordee dit si l'agent reste valide, la
        # saturation dit combien de noms compatibles il utilise vraiment.
        noms = grammaire.tokens_par_categorie["nom"]
        p_tous = np.array([float(bloc[grammaire.index[n]].sum()) / total for n in noms])
        pt = p_tous[p_tous > 1e-12]
        h_tous = float(-(pt * np.log2(pt)).sum())

        masse_accordee = float(sum(p_tous[i] for i, n in enumerate(noms)
                                   if n in noms_compatibles))
        if masse_accordee > 1e-12:
            p_acc = np.array([p_tous[i] for i, n in enumerate(noms)
                              if n in noms_compatibles]) / masse_accordee
            p_acc = p_acc[p_acc > 1e-12]
            h_accorde = float(-(p_acc * np.log2(p_acc)).sum())
        else:
            h_accorde = 0.0
        h_max = float(np.log2(len(noms_compatibles))) if noms_compatibles else 0.0
        entropie_nom_sachant_det[det] = {
            "H_bits": round(h_tous, 3),
            "H_accorde_bits": round(h_accorde, 3),
            "H_max_bits": round(h_max, 3),
            # Borne a 100 % par construction : calculee sur la conditionnelle
            # RESTREINTE aux noms compatibles puis renormalisee.
            "saturation_pct": round(100 * h_accorde / h_max, 1) if h_max > 0 else None,
            "masse_accordee_pct": round(100 * masse_accordee, 2),
            "noms_compatibles": len(noms_compatibles),
            "masse_du_determinant": round(total, 5),
        }
    for nom in grammaire.tokens_par_categorie["nom"]:
        bloc = jointe[:, grammaire.index[nom], :]                 # [det, verbe]
        total = float(bloc.sum())
        nombre = grammaire.traits(nom)["nombre"]
        masse = sum(float(bloc[:, grammaire.index[v]].sum())
                    for v in grammaire.tokens_par_categorie["verbe"]
                    if grammaire._compatible(grammaire.traits(v)["nombre"], nombre))
        cond_nom[nom] = round(masse / total, 4) if total > 0 else None

    # Repartition de la masse grammaticale entre les familles de solutions.
    # L'uniformite globale peut etre bonne alors que l'agent ignore une famille
    # entiere : les 24 phrases au singulier et les 24 au pluriel forment deux
    # sous-espaces de meme taille, et rien n'oblige a les visiter tous les deux.
    sequences_valides = sequences[valide]
    familles = {"sg": 0.0, "pl": 0.0}
    par_nom = {n: 0.0 for n in grammaire.tokens_par_categorie["nom"]}
    for j, seq in enumerate(sequences_valides.tolist()):
        nom = grammaire.tokens[seq[i_nom]]
        familles[grammaire.traits(nom)["nombre"]] += float(p[j])
        par_nom[nom] += float(p[j])

    valeurs_det = [v for v in cond_det.values() if v is not None]
    valeurs_nom = [v for v in cond_nom.values() if v is not None]

    # La moyenne NON PONDEREE ci-dessus compte les six determinants a egalite,
    # y compris ceux que la politique n'emet jamais. Un softmax n'atteint jamais
    # zero, donc les lignes mortes passent le garde-fou et entrent dans la
    # moyenne : la quantite obtenue est (determinants emis)/6, pas un accord.
    # Deux corrections, et la seconde seule repond a la vraie question.
    def _pondere(cond, masses):
        total = sum(masses.values())
        if total <= 1e-12:
            return None
        return round(sum(masses[k] * v for k, v in cond.items() if v is not None) / total, 4)

    masse_par_nom = {}
    for nom in grammaire.tokens_par_categorie["nom"]:
        masse_par_nom[nom] = float(jointe[:, grammaire.index[nom], :].sum())

    # Information mutuelle I(det ; nom) en bits. C'est la seule des trois qui
    # mesure une DEPENDANCE : une politique restreinte a un genre a un accord
    # parfait sans aucun couplage, parce que son support est un produit. Elle
    # vaut donc 0 pour un produit et > 0 des que le nom depend du determinant.
    p_det_nom = np.array([[float(jointe[grammaire.index[d], grammaire.index[n]].sum())
                           for n in grammaire.tokens_par_categorie["nom"]]
                          for d in grammaire.tokens_par_categorie["det"]])
    p_det_nom = p_det_nom / max(p_det_nom.sum(), 1e-30)
    p_d, p_n = p_det_nom.sum(1, keepdims=True), p_det_nom.sum(0, keepdims=True)
    produit = p_d * p_n
    nz = (p_det_nom > 1e-12) & (produit > 1e-30)
    im_det_nom = float((p_det_nom[nz] * np.log2(p_det_nom[nz] / produit[nz])).sum())

    return {
        "cond_det_pondere": _pondere(cond_det, masse_det),
        "cond_nom_pondere": _pondere(cond_nom, masse_par_nom),
        "information_mutuelle_det_nom_bits": round(im_det_nom, 4),
        "determinants_emis": sum(1 for v in masse_det.values() if v > 0.01),
        "entropie_nom_sachant_det": entropie_nom_sachant_det,
        "masse_par_determinant": masse_det,
        "repartition_familles": {k: round(100 * v, 1) for k, v in familles.items()},
        "repartition_par_nom": {k: round(100 * v, 1) for k, v in par_nom.items()},
        "familles_visitees": sum(1 for v in familles.values() if v > 0.05),
        "masse_valide_pct": round(100 * masse_valide, 3),
        "nb_solutions": nb_solutions,
        "entropie_sur_valides_bits": round(entropie, 3),
        "entropie_max_bits": round(entropie_max, 3),
        "uniformite_pct": round(100 * entropie / entropie_max, 1) if entropie_max > 0 else 0.0,
        "modes_effectifs": round(modes_effectifs, 1),
        "part_du_mode_principal_pct": round(100 * float(p.max()), 2),
        "top5_valides": top,
        "cond_det_vers_nom": cond_det,
        "cond_nom_vers_verbe": cond_nom,
        "moyenne_cond_det": round(float(np.mean(valeurs_det)), 4),
        "moyenne_cond_nom": round(float(np.mean(valeurs_nom)), 4),
    }


def test_conditionnel(politique, grammaire, n=400, device="cpu"):
    """L'agent a-t-il appris des conditionnelles systematiques, ou seulement
    quelques triplets ?

    (a) on impose un determinant et on regarde la masse de probabilite placee
        sur les noms qui s'accordent avec lui ;
    (b) on impose un nom et on regarde la masse placee sur les verbes accordes.
    """
    i_det, i_nom, i_verbe = (grammaire.positions["det"], grammaire.positions["nom"],
                             grammaire.positions["verbe"])
    resultats = {"det_vers_nom": {}, "nom_vers_verbe": {}}

    for det in grammaire.tokens_par_categorie["det"]:
        with torch.no_grad():
            _, _, _, probas = politique.generer(
                grammaire.longueur, taille_lot=n,
                forcer={i_det: grammaire.index[det]}, device=device)
        moyenne = probas[:, i_nom, :].mean(0)
        traits_det = grammaire.traits(det)
        masse = sum(float(moyenne[grammaire.index[nom]])
                    for nom in grammaire.tokens_par_categorie["nom"]
                    if all(grammaire._compatible(traits_det[t], grammaire.traits(nom)[t])
                           for t in ("genre", "nombre")))
        resultats["det_vers_nom"][det] = round(masse, 4)

    for nom in grammaire.tokens_par_categorie["nom"]:
        with torch.no_grad():
            _, _, _, probas = politique.generer(
                grammaire.longueur, taille_lot=n,
                forcer={i_nom: grammaire.index[nom]}, device=device)
        moyenne = probas[:, i_verbe, :].mean(0)
        nombre_nom = grammaire.traits(nom)["nombre"]
        masse = sum(float(moyenne[grammaire.index[v]])
                    for v in grammaire.tokens_par_categorie["verbe"]
                    if grammaire._compatible(grammaire.traits(v)["nombre"], nombre_nom))
        resultats["nom_vers_verbe"][nom] = round(masse, 4)

    resultats["moyenne_det_vers_nom"] = round(
        float(np.mean(list(resultats["det_vers_nom"].values()))), 4)
    resultats["moyenne_nom_vers_verbe"] = round(
        float(np.mean(list(resultats["nom_vers_verbe"].values()))), 4)
    return resultats


def masque_paire_exclue(grammaire, det_exclu, nom_exclu):
    """Interdit le nom seulement lorsque le determinant exclu vient d'etre emis.

    Les deux tokens restent entraines par ailleurs : 'des' avec d'autres noms,
    'fleurs' avec d'autres determinants. Seule leur COMBINAISON n'est jamais
    recompensee, ce qui permet de tester une generalisation compositionnelle.
    """
    i_det, i_nom = grammaire.positions["det"], grammaire.positions["nom"]
    id_det, id_nom = grammaire.index[det_exclu], grammaire.index[nom_exclu]

    def masque(t, actions_deja):
        if t != i_nom or actions_deja is None:
            return None
        autorise = torch.ones(actions_deja.shape[0], grammaire.taille, dtype=torch.bool)
        concerne = actions_deja[:, i_det] == id_det
        autorise[concerne, id_nom] = False
        return autorise
    return masque


def masque_token_exclu(grammaire, token_exclu):
    """Interdit totalement un token en position nom (version litterale de la
    spec : un nom jamais vu pendant l'entrainement)."""
    i_nom = grammaire.positions["nom"]
    id_token = grammaire.index[token_exclu]

    def masque(t, actions_deja):
        if t != i_nom:
            return None
        lot = actions_deja.shape[0] if actions_deja is not None else 1
        autorise = torch.ones(lot, grammaire.taille, dtype=torch.bool)
        autorise[:, id_token] = False
        return autorise
    return masque


def sauvegarder_csv(res, chemin):
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "recompense", "recompense_moyenne_100", "validite_100"])
        for i, (r, m, v) in enumerate(zip(res["recompenses_log"], res["moyennes_log"],
                                          res["validites_log"]), start=1):
            w.writerow([i, f"{r:.6f}", f"{m:.6f}", f"{v:.6f}"])


def tracer(runs, chemin, titre):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    couleurs = plt.cm.tab10.colors
    for k, res in enumerate(runs):
        c = couleurs[k % len(couleurs)]
        eps = range(1, len(res["moyennes_log"]) + 1)
        ax1.plot(eps, res["moyennes_log"], color=c, linewidth=1.5, label=res["etiquette"])
        ax2.plot(eps, [100 * v for v in res["validites_log"]], color=c, linewidth=1.5,
                 label=res["etiquette"])
    ax1.set_ylabel("recompense (moyenne 100)")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)
    ax1.set_title(titre)
    ax2.set_ylabel("% de phrases grammaticalement valides")
    ax2.set_xlabel("episode")
    ax2.set_ylim(-2, 105)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(chemin, dpi=130)
    plt.close()


def tracer_compromis(balayage, chemin, total_valides):
    """Validite et couverture en fonction du coefficient d'entropie.

    Question posee : existe-t-il un regime ou l'agent respecte les regles ET
    explore l'ensemble des solutions, ou faut-il choisir ?
    """
    coefs = [b["coef_entropie"] for b in balayage]
    x = range(len(coefs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))

    ax1.plot(x, [b["masse_valide_exacte_pct"] for b in balayage], "o-", color="#1f77b4",
             label="masse grammaticale (exacte)")
    ax1.plot(x, [b["uniformite_pct"] for b in balayage], "^-", color="#ff7f0e",
             label="uniformite sur les solutions (H / Hmax)")
    ax1.plot(x, [100 * b["modes_effectifs"] / total_valides for b in balayage], "s-",
             color="#d62728", label=f"modes effectifs / {total_valides}")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([str(c) for c in coefs])
    ax1.set_xlabel("coefficient d'entropie")
    ax1.set_ylabel("%")
    ax1.set_ylim(-3, 105)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)
    ax1.set_title("Grammaticalite et diversite selon le bonus d'entropie")

    # Frontiere : chaque point est un regime d'entrainement. L'axe des abscisses
    # est le nombre de solutions REELLEMENT utilisees, pas le nombre apercu.
    ax2.plot([b["modes_effectifs"] for b in balayage],
             [b["masse_valide_exacte_pct"] for b in balayage], "o-", color="#2ca02c")
    for b in balayage:
        ax2.annotate(str(b["coef_entropie"]),
                     (b["modes_effectifs"], b["masse_valide_exacte_pct"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax2.axvline(total_valides, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax2.text(total_valides, 4, f" {total_valides} solutions\n (uniforme)", fontsize=8, alpha=0.7)
    ax2.set_xlabel("modes effectifs 2^H (nombre de phrases valides reellement utilisees)")
    ax2.set_ylabel("masse grammaticale (%)")
    ax2.set_xlim(0, total_valides * 1.15)
    ax2.set_ylim(-3, 105)
    ax2.grid(alpha=0.25)
    ax2.set_title("Frontiere grammaticalite / diversite")

    plt.tight_layout()
    plt.savefig(chemin, dpi=130)
    plt.close()


def afficher_evaluation(titre, ev):
    print(f"  {titre}")
    print(f"    validite            : {ev['validite_pct']:.1f} % "
          f"(hasard : {ev['validite_hasard_pct']:.4f} %)")
    print(f"    phrases distinctes  : {ev['phrases_distinctes']} sur {ev['n']} tirages")
    print(f"    valides distinctes  : {ev['phrases_valides_distinctes']} / "
          f"{ev['phrases_valides_possibles']} possibles "
          f"({ev['couverture_pct']:.1f} % de couverture)")
    print(f"    entropie            : {ev['entropie_bits']} bits")
    print(f"    exemples            : {ev['exemples_valides'][:5]}")


def main():
    p = argparse.ArgumentParser(description="RDTRL — test 2 : grammaire en RL pur")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--coef-entropie", type=float, default=0.01)
    p.add_argument("--balayage", type=float, nargs="+",
                   default=[0.0, 0.01, 0.02, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5],
                   help="coefficients d'entropie a comparer")
    p.add_argument("--device", default="cpu")
    p.add_argument("--rapide", action="store_true")
    args = p.parse_args()
    if args.rapide:
        args.episodes = 3000

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    rapport = {"hyperparametres": vars(args)}

    courte, longue = Grammaire(longue=False), Grammaire(longue=True)
    print("=" * 78)
    print("RDTRL — TEST 2 : APPRENDRE UNE GRAMMAIRE EN RL PUR")
    print("=" * 78)
    print(courte.resume())
    print()
    print(longue.resume())
    print()
    rapport["grammaires"] = {
        "courte": {"espace": courte.taille_espace(), "valides": courte.compter_phrases_valides(),
                   "hasard_pct": 100 * courte.probabilite_hasard()},
        "longue": {"espace": longue.taille_espace(), "valides": longue.compter_phrases_valides(),
                   "hasard_pct": 100 * longue.probabilite_hasard()},
    }

    runs = {}

    def lancer(cle, grammaire, type_recompense, masque=None, etiquette=None,
               verbeux=True, coef_entropie=None):
        print("-" * 78)
        print(f"RUN {cle}")
        print("-" * 78)
        politique = nouvelle_politique(grammaire, args.graine, args.device)
        res = entrainer(politique, grammaire, max_episodes=args.episodes,
                        type_recompense=type_recompense, lr=args.lr,
                        coef_entropie=args.coef_entropie if coef_entropie is None else coef_entropie,
                        masque_fn=masque,
                        verbeux=verbeux, periode_log=2000, device=args.device,
                        etiquette=etiquette or cle)
        print(f"  >> {res['episodes']} episodes en {res['duree_s']}s | "
              f"reward final {res['reward_final']:.4f} | "
              f"validite finale {res['validite_finale_pct']:.1f} % | "
              f"1re phrase valide a l'episode {res['premier_valide']}")
        runs[cle] = res
        sauvegarder_csv(res, os.path.join(DOSSIER_SORTIE, f"run_{cle}.csv"))
        return politique, res

    # Balayage du coefficient d'entropie.
    # Les 48 phrases valides rapportent toutes exactement 1.0 : en trouver UNE
    # est deja un optimum global. Rien dans la recompense ne pousse a la
    # diversite ; le seul contre-poids est le bonus d'entropie. On mesure donc
    # son effet avant d'interpreter quoi que ce soit d'autre.
    print("=" * 78)
    print("BALAYAGE DU COEFFICIENT D'ENTROPIE (grammaire courte, recompense graduee)")
    print("=" * 78)
    balayage, politiques = [], {}
    for coef in args.balayage:
        pol, res = lancer(f"courte_ent{coef}", courte, "graduee",
                          coef_entropie=coef, verbeux=False)
        ev = evaluer(pol, courte, n=500, device=args.device)
        exact = analyse_exacte(pol, courte, device=args.device)
        politiques[coef] = pol
        torch.save(pol.state_dict(), os.path.join(DOSSIER_SORTIE, f"politique_ent{coef}.pt"))
        balayage.append({
            "coef_entropie": coef,
            "validite_pct": round(ev["validite_pct"], 1),
            "couverture_pct": round(ev["couverture_pct"], 1),
            "valides_distinctes": ev["phrases_valides_distinctes"],
            "entropie_echantillon_bits": ev["entropie_bits"],
            # Mesures exactes (enumeration des 8 000 sequences, sans bruit)
            "masse_valide_exacte_pct": exact["masse_valide_pct"],
            "uniformite_pct": exact["uniformite_pct"],
            "modes_effectifs": exact["modes_effectifs"],
            "part_mode_principal_pct": exact["part_du_mode_principal_pct"],
            "repartition_familles": exact["repartition_familles"],
            "repartition_par_nom": exact["repartition_par_nom"],
            "top5_valides": exact["top5_valides"],
            "p_nom_sachant_det": exact["moyenne_cond_det"],
            "p_verbe_sachant_nom": exact["moyenne_cond_nom"],
            "detail_det_vers_nom": exact["cond_det_vers_nom"],
            "detail_nom_vers_verbe": exact["cond_nom_vers_verbe"],
        })

    # Le tableau distingue trois choses que la couverture seule confond :
    # combien de masse est grammaticale, sur combien de solutions elle se
    # repartit reellement (modes effectifs), et a quel point cette repartition
    # est plate (uniformite = H / H_max sur les 48 solutions).
    print()
    print(f"{'coef':>6} {'valide%':>9} {'modes':>7} {'unifor%':>8} {'mode1%':>7} "
          f"{'sg%':>6} {'pl%':>6} {'P(nom|det)':>11} {'P(vb|nom)':>10}")
    print(f"{'':>6} {'(exact)':>9} {'eff./48':>7} {'H/Hmax':>8} {'':>7} "
          f"{'famille':>6} {'famille':>6} {'':>11} {'':>10}")
    for b in balayage:
        f = b["repartition_familles"]
        print(f"{b['coef_entropie']:>6} {b['masse_valide_exacte_pct']:>9.2f} "
              f"{b['modes_effectifs']:>7.1f} {b['uniformite_pct']:>8.1f} "
              f"{b['part_mode_principal_pct']:>7.1f} {f['sg']:>6.1f} {f['pl']:>6.1f} "
              f"{b['p_nom_sachant_det']:>11.3f} {b['p_verbe_sachant_nom']:>10.3f}")
    print("  sg%/pl% = part de la masse grammaticale sur chacune des deux familles")
    print("            de 24 solutions (singulier / pluriel). 50/50 = reparti.")
    rapport["balayage_entropie"] = balayage
    tracer_compromis(balayage, os.path.join(DOSSIER_SORTIE, "compromis_entropie.png"),
                     courte.compter_phrases_valides())

    # On retient le coefficient qui couvre le mieux l'espace des solutions
    # tout en restant grammatical : c'est le seul regime ou les tests de
    # generalisation ont un sens (sinon on mesure l'effondrement de mode).
    # Critere fonde sur les mesures exactes : la masse grammaticale doit rester
    # haute, et on maximise le nombre de modes REELLEMENT utilises, pas le
    # nombre de phrases distinctes apercues sur 500 tirages.
    candidats = [b for b in balayage if b["masse_valide_exacte_pct"] >= 90.0]
    if not candidats:
        candidats = balayage
    meilleur = max(candidats, key=lambda b: b["modes_effectifs"])
    coef_retenu = meilleur["coef_entropie"]
    print(f"\n  -> coefficient retenu pour la suite : {coef_retenu} "
          f"(masse valide {meilleur['masse_valide_exacte_pct']} %, "
          f"{meilleur['modes_effectifs']} modes effectifs sur 48, "
          f"uniformite {meilleur['uniformite_pct']} %)")
    print(f"     phrases les plus probables : {meilleur['top5_valides']}")
    rapport["coef_retenu"] = coef_retenu
    pol_courte = politiques[coef_retenu]
    exact_retenu = analyse_exacte(pol_courte, courte, device=args.device)
    afficher_evaluation("Regime retenu :", evaluer(pol_courte, courte, n=500, device=args.device))
    print(f"    P(nom accorde | det impose)   : {meilleur['detail_det_vers_nom']}")
    print(f"    P(verbe accorde | nom impose) : {meilleur['detail_nom_vers_verbe']}")

    # Diversite A L'INTERIEUR de chaque branche, separee de la couverture ENTRE
    # branches : un determinant peut saturer son entropie tout en n'etant
    # presque jamais emis.
    print("\n  Entropie conditionnelle H(nom | determinant) :")
    print(f"    {'det':>5} {'masse':>8} {'accord%':>8} {'H bits':>8} {'H acc.':>8} "
          f"{'H max':>7} {'satur.%':>8} {'noms ok':>8}")
    for det, e in sorted(exact_retenu["entropie_nom_sachant_det"].items()):
        if e is None:
            print(f"    {det:>5} {'~0':>8} {'n/a':>8}")
            continue
        print(f"    {det:>5} {e['masse_du_determinant']:>8.4f} "
              f"{e['masse_accordee_pct']:>8.2f} {e['H_bits']:>8.3f} "
              f"{e['H_accorde_bits']:>8.3f} {e['H_max_bits']:>7.3f} "
              f"{(e['saturation_pct'] if e['saturation_pct'] is not None else 0):>8.1f} "
              f"{e['noms_compatibles']:>8}")
    print("    accord% = masse placee sur des noms qui s'accordent avec le determinant.")
    print("    satur.% = H sur ces seuls noms, renormalisee, rapportee a log2(noms ok).")
    rapport["regime_retenu"] = exact_retenu
    print()

    # Le choix de branche est-il une loterie d'initialisation ou un bassin
    # structurellement plus large ? Meme coefficient, graines differentes.
    print("  Choix de branche sur 3 graines supplementaires (meme coefficient) :")
    branches = []
    for graine in (1, 2, 3):
        fixer_graine(graine)
        pol_g = PolitiqueGRU(courte.taille).to(args.device)
        entrainer(pol_g, courte, max_episodes=args.episodes, type_recompense="graduee",
                  lr=args.lr, coef_entropie=coef_retenu, verbeux=False,
                  device=args.device, etiquette=f"graine{graine}")
        ex = analyse_exacte(pol_g, courte, device=args.device)
        branches.append({"graine": graine,
                         "masse_valide_pct": ex["masse_valide_pct"],
                         "modes_effectifs": ex["modes_effectifs"],
                         "familles": ex["repartition_familles"]})
        print(f"    graine {graine} : valide {ex['masse_valide_pct']:6.2f} % | "
              f"{ex['modes_effectifs']:5.1f} modes | "
              f"sg {ex['repartition_familles']['sg']:5.1f} % / "
              f"pl {ex['repartition_familles']['pl']:5.1f} %")
    familles_vues = {("sg" if b["familles"]["sg"] > b["familles"]["pl"] else "pl")
                     for b in branches} | {("sg" if meilleur["repartition_familles"]["sg"]
                                            > meilleur["repartition_familles"]["pl"] else "pl")}
    print(f"    -> branches dominantes observees sur 4 graines : {sorted(familles_vues)}")
    print(f"       {'loterie d initialisation' if len(familles_vues) > 1 else 'toujours la meme branche : bassin structurellement favorise'}")
    rapport["etude_branches"] = {"par_graine": branches,
                                 "branches_dominantes": sorted(familles_vues)}
    print()

    # Controle tout-ou-rien, a coefficient d'entropie identique
    pol_tr, _ = lancer("courte_tout_ou_rien", courte, "tout_ou_rien",
                       coef_entropie=coef_retenu, verbeux=False)
    ev_tr = evaluer(pol_tr, courte, n=500, device=args.device)
    afficher_evaluation("Evaluation :", ev_tr)
    exact_tr = analyse_exacte(pol_tr, courte, device=args.device)
    print(f"    masse valide (exacte)  : {exact_tr['masse_valide_pct']:.2f} %")
    print(f"    modes effectifs        : {exact_tr['modes_effectifs']} / 48 "
          f"(uniformite {exact_tr['uniformite_pct']} %)")
    print(f"    P(nom accorde | det)   : {exact_tr['moyenne_cond_det']:.3f}")
    print(f"    P(verbe accorde | nom) : {exact_tr['moyenne_cond_nom']:.3f}")
    torch.save(pol_tr.state_dict(), os.path.join(DOSSIER_SORTIE, "politique_tout_ou_rien.pt"))
    rapport["courte_tout_ou_rien"] = {"evaluation": ev_tr, "exact": exact_tr}
    print()

    tracer([runs[f"courte_ent{coef_retenu}"], runs["courte_tout_ou_rien"]],
           os.path.join(DOSSIER_SORTIE, "courbe_grammaire_courte.png"),
           f"Grammaire courte — espace {courte.taille_espace()}, "
           f"{100*courte.probabilite_hasard():.2f} % de validite au hasard")

    # Meme comparaison sur la grammaire longue : l'espace passe de 8 000 a
    # 28,6 M et la validite au hasard de 0,6 % a 0,001 %. C'est ce qui separe
    # "la logique de la grammaire aide" de "l'espace etait simplement petit".
    print("=" * 78)
    print("PASSAGE A L'ECHELLE — grammaire longue (espace x3576)")
    print("=" * 78)
    pol_lg, _ = lancer("longue_graduee", longue, "graduee",
                       coef_entropie=coef_retenu, verbeux=True)
    ev_lg = evaluer(pol_lg, longue, n=500, device=args.device)
    afficher_evaluation("Evaluation :", ev_lg)
    rapport["longue_graduee"] = {"evaluation": ev_lg}
    print()

    pol_ltr, _ = lancer("longue_tout_ou_rien", longue, "tout_ou_rien",
                        coef_entropie=coef_retenu, verbeux=True)
    ev_ltr = evaluer(pol_ltr, longue, n=500, device=args.device)
    afficher_evaluation("Evaluation :", ev_ltr)
    rapport["longue_tout_ou_rien"] = {"evaluation": ev_ltr}
    print()

    tracer([runs["longue_graduee"], runs["longue_tout_ou_rien"]],
           os.path.join(DOSSIER_SORTIE, "courbe_grammaire_longue.png"),
           f"Grammaire longue — espace {longue.taille_espace()}, "
           f"{100*longue.probabilite_hasard():.4f} % de validite au hasard")

    # Generalisation : une COMBINAISON jamais recompensee.
    print("=" * 78)
    print("TEST DE GENERALISATION — combinaison 'des fleurs' jamais recompensee")
    print("=" * 78)
    print("Les deux tokens sont entraines normalement ('des' avec d'autres noms,")
    print("'fleurs' avec d'autres determinants) ; seule leur combinaison est masquee.")
    pol_paire, _ = lancer("courte_paire_exclue", courte, "graduee",
                          masque=masque_paire_exclue(courte, "des", "fleurs"),
                          coef_entropie=coef_retenu, verbeux=False)
    i_det, i_nom, i_verbe = (courte.positions["det"], courte.positions["nom"],
                             courte.positions["verbe"])
    # Fiabilite du test : si l'agent n'emet quasiment jamais 'des', la mesure
    # de P(fleurs | des) porte sur un prefixe marginal et ne vaut rien.
    exact_paire = analyse_exacte(pol_paire, courte, device=args.device)
    sequences_p, probas_p = distribution_exacte(pol_paire, courte, device=args.device)
    jointe_p = probas_p.reshape([courte.taille] * courte.longueur)
    p_des = float(jointe_p[courte.index["des"]].sum())
    print(f"  masse valide {exact_paire['masse_valide_pct']:.2f} % | "
          f"{exact_paire['modes_effectifs']} modes effectifs | "
          f"P(des en position 1) = {p_des:.4f}")
    if p_des < 0.02:
        print("  ATTENTION : 'des' est quasiment jamais emis, le test qui suit est peu fiable.")
    familles_paire = exact_paire["repartition_familles"]
    branche_morte = [k for k, v in familles_paire.items() if v < 5.0]
    if branche_morte:
        print(f"  PORTEE : la branche {branche_morte} porte moins de 5 % de la masse. "
              f"La conclusion de generalisation ne vaut que pour la branche "
              f"{[k for k in familles_paire if k not in branche_morte]}.")

    with torch.no_grad():
        _, _, _, probas = pol_paire.generer(courte.longueur, taille_lot=500,
                                            forcer={i_det: courte.index["des"]},
                                            device=args.device)
    distribution_nom = probas[:, i_nom, :].mean(0)
    p_fleurs = float(distribution_nom[courte.index["fleurs"]])
    autres_pl = [n for n in courte.tokens_par_categorie["nom"]
                 if courte.traits(n)["nombre"] == "pl" and n != "fleurs"]
    p_autres = {n: round(float(distribution_nom[courte.index[n]]), 4) for n in autres_pl}
    moyenne_autres = float(np.mean(list(p_autres.values())))
    # Reference : la meme paire dans un run ou elle N'ETAIT PAS exclue
    with torch.no_grad():
        _, _, _, probas_ref = pol_courte.generer(courte.longueur, taille_lot=500,
                                                 forcer={i_det: courte.index["des"]},
                                                 device=args.device)
    p_fleurs_ref = float(probas_ref[:, i_nom, :].mean(0)[courte.index["fleurs"]])
    ratio = p_fleurs / moyenne_autres if moyenne_autres > 0 else 0.0
    print(f"  P(fleurs | des)  apres exclusion : {p_fleurs:.4f}   <- jamais recompense")
    print(f"  P(autres noms pluriels | des)    : {p_autres} (moyenne {moyenne_autres:.4f})")
    print(f"  P(fleurs | des)  sans exclusion  : {p_fleurs_ref:.4f}   <- reference")
    print(f"  ratio exclu / autres pluriels    : {ratio:.3f}")
    print(f"  -> {'generalisation compositionnelle' if ratio > 0.5 else 'pas de generalisation : la combinaison exclue reste evitee'}")
    rapport["generalisation_paire"] = {
        "fiabilite": {"p_des": round(p_des, 5),
                      "masse_valide_pct": exact_paire["masse_valide_pct"],
                      "modes_effectifs": exact_paire["modes_effectifs"]},
        "p_fleurs_sachant_des_apres_exclusion": round(p_fleurs, 5),
        "p_autres_pluriels": p_autres,
        "moyenne_autres": round(moyenne_autres, 5),
        "p_fleurs_sachant_des_reference": round(p_fleurs_ref, 5),
        "ratio": round(ratio, 4),
    }
    print()

    # Version litterale de la spec : un nom jamais vu du tout. Son embedding et
    # sa ligne dans la couche de sortie restent a l'initialisation aleatoire :
    # aucune information n'existe sur ce token, le resultat au hasard est attendu.
    print("=" * 78)
    print("TEST DU NOM JAMAIS VU ('fleurs' totalement masque a l'entrainement)")
    print("=" * 78)
    pol_token, _ = lancer("courte_token_exclu", courte, "graduee",
                          masque=masque_token_exclu(courte, "fleurs"),
                          coef_entropie=coef_retenu, verbeux=False)

    def masse_verbe_accorde(politique, nom):
        with torch.no_grad():
            _, _, _, pr = politique.generer(courte.longueur, taille_lot=500,
                                            forcer={i_nom: courte.index[nom]},
                                            device=args.device)
        moyenne = pr[:, i_verbe, :].mean(0)
        nombre = courte.traits(nom)["nombre"]
        return sum(float(moyenne[courte.index[v]])
                   for v in courte.tokens_par_categorie["verbe"]
                   if courte.traits(v)["nombre"] == nombre)

    mesures = {nom: round(masse_verbe_accorde(pol_token, nom), 4)
               for nom in courte.tokens_par_categorie["nom"]}
    vus = {n: v for n, v in mesures.items() if n != "fleurs"}
    print(f"  P(verbe correctement accorde | nom impose) :")
    for nom, v in sorted(mesures.items()):
        marque = "  <- JAMAIS VU a l'entrainement" if nom == "fleurs" else ""
        print(f"    {nom:8s} : {v:.4f}{marque}")
    print(f"  moyenne sur les noms vus : {np.mean(list(vus.values())):.4f}")
    print(f"  hasard (3 verbes accordes sur 6) : 0.5000")
    rapport["nom_jamais_vu"] = {
        "par_nom": mesures,
        "moyenne_noms_vus": round(float(np.mean(list(vus.values()))), 4),
        "nom_jamais_vu": mesures["fleurs"],
        "hasard": 0.5,
    }

    with open(os.path.join(DOSSIER_SORTIE, "rapport.json"), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False, default=str)
    print()
    print(f"Rapport ecrit dans {DOSSIER_SORTIE}")


if __name__ == "__main__":
    main()
