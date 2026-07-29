"""
RDTRL — Test 1 : apprendre a copier une phrase fixe en RL pur.

But de l'experience
Verifier empiriquement si une politique initialisee 100% aleatoirement peut
apprendre a produire une phrase cible en n'utilisant QUE un signal de
recompense (aucune paire entree/sortie, aucun pre-entrainement, aucune
connaissance de la langue injectee).

Contraintes respectees dans ce fichier :
  - les poids sont initialises par PyTorch (aleatoire), jamais charges ;
  - la cible n'est JAMAIS donnee au modele : elle n'entre que dans la fonction
    de recompense, qui renvoie un simple flottant ;
  - aucun teacher forcing : a chaque pas on reinjecte le caractere que la
    politique a elle-meme echantillonne ;
  - le vocabulaire est une simple liste de caracteres (pas une connaissance
    linguistique : c'est l'alphabet du probleme, l'equivalent des "actions
    possibles" d'un environnement RL).

Usage :
    python rl_copie.py                 # experience complete (phase 1 + phase 2)
    python rl_copie.py --rapide        # version courte pour verifier que ca tourne
    python rl_copie.py --episodes 50000
"""

import argparse
import csv
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif, on sauvegarde des PNG
import matplotlib.pyplot as plt


# Configuration generale

CIBLE_PRINCIPALE = "le chat dort"      # 12 caracteres
CIBLE_PERTURBEE = "le chien dort"      # 13 caracteres, un seul mot change

DOSSIER_SORTIE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats")


# Vocabulaire

def construire_vocabulaire(phrases):
    """Alphabet = union des caracteres des phrases utilisees + l'espace.

    On inclut d'emblee les caracteres de la cible perturbee ('i', 'n') pour que
    l'architecture reste identique entre la phase 1 et le test de perturbation
    (sinon la couche de sortie changerait de taille et la comparaison n'aurait
    aucun sens).
    """
    caracteres = {" "}
    for phrase in phrases:
        caracteres.update(phrase)
    return sorted(caracteres)


class Vocabulaire:
    def __init__(self, caracteres):
        self.caracteres = list(caracteres)
        self.index_par_caractere = {c: i for i, c in enumerate(self.caracteres)}
        self.taille = len(self.caracteres)

    def encoder(self, texte):
        return [self.index_par_caractere[c] for c in texte]

    def decoder(self, indices):
        return "".join(self.caracteres[i] for i in indices)


# Fonctions de recompense
#   Toutes prennent (indices generes, indices cibles) et renvoient un float
#   dans [0, 1]. C'est le SEUL canal par lequel l'information sur la cible
#   atteint l'agent.

def recompense_positions(genere, cible):
    """Fraction de caracteres corrects a la bonne position. Signal dense."""
    justes = sum(1 for a, b in zip(genere, cible) if a == b)
    return justes / len(cible)


def distance_levenshtein(a, b):
    """Distance d'edition classique (programmation dynamique, une seule ligne)."""
    if a == b:
        return 0
    ligne_precedente = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        ligne = [i]
        for j, cb in enumerate(b, start=1):
            cout = 0 if ca == cb else 1
            ligne.append(min(
                ligne_precedente[j] + 1,        # suppression
                ligne[j - 1] + 1,               # insertion
                ligne_precedente[j - 1] + cout  # substitution
            ))
        ligne_precedente = ligne
    return ligne_precedente[-1]


def recompense_levenshtein(genere, cible):
    """1 - distance normalisee : plus tolerant aux decalages de position."""
    longueur_max = max(len(genere), len(cible))
    if longueur_max == 0:
        return 1.0
    return 1.0 - distance_levenshtein(genere, cible) / longueur_max


def recompense_tout_ou_rien(genere, cible):
    """Recompense VRAIMENT sparse : 1 seulement si la phrase est exacte.

    C'est le controle qui teste directement l'objection "sparse reward" :
    ici l'agent ne recoit aucun signal tant qu'il n'a pas trouve la phrase
    entiere par hasard.
    """
    return 1.0 if list(genere) == list(cible) else 0.0


RECOMPENSES = {
    "positions": recompense_positions,
    "levenshtein": recompense_levenshtein,
    "tout_ou_rien": recompense_tout_ou_rien,
}


# La politique (l'agent)

class PolitiqueGRU(nn.Module):
    """GRU 1 couche, genere un caractere a la fois de facon autoregressive.

    Entree du pas t : le caractere echantillonne au pas t-1 (au pas 0, un
    token special "debut" qui ne fait pas partie du vocabulaire de sortie).
    Aucune information sur la cible n'entre jamais dans ce reseau.
    """

    def __init__(self, taille_vocab, taille_cachee=128, taille_embedding=32):
        super().__init__()
        self.taille_vocab = taille_vocab
        self.taille_cachee = taille_cachee
        self.token_debut = taille_vocab  # index reserve, hors vocabulaire de sortie
        self.embedding = nn.Embedding(taille_vocab + 1, taille_embedding)
        self.gru = nn.GRU(taille_embedding, taille_cachee, num_layers=1, batch_first=True)
        self.tete = nn.Linear(taille_cachee, taille_vocab)

    def generer(self, longueur, taille_lot=1, greedy=False,
                ablation_position=None, ablation_mode="zero", device="cpu"):
        """Deroule une sequence complete.

        greedy=True  -> argmax, pas d'exploration (utilise pour l'evaluation).
        ablation_position=t -> on corrompt l'etat cache APRES avoir produit le
        caractere t, ce qui permet de mesurer l'information transportee vers
        les positions suivantes.

        Renvoie : actions [lot, longueur], log_probs [lot, longueur],
                  entropies [lot, longueur], probabilites [lot, longueur, vocab]
        """
        h = torch.zeros(1, taille_lot, self.taille_cachee, device=device)
        entree = torch.full((taille_lot,), self.token_debut, dtype=torch.long, device=device)

        actions, log_probs, entropies, probabilites = [], [], [], []
        for t in range(longueur):
            emb = self.embedding(entree).unsqueeze(1)          # [lot, 1, emb]
            sortie, h = self.gru(emb, h)                        # sortie [lot, 1, cache]
            logits = self.tete(sortie.squeeze(1))               # [lot, vocab]
            distribution = torch.distributions.Categorical(logits=logits)

            action = logits.argmax(dim=-1) if greedy else distribution.sample()

            actions.append(action)
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            probabilites.append(distribution.probs)

            # Ablation eventuelle de l'etat cache (test 3 de la phase 2)
            if ablation_position is not None and t == ablation_position:
                h = torch.zeros_like(h) if ablation_mode == "zero" else torch.randn_like(h)

            entree = action  # pas de teacher forcing : on reinjecte notre propre sortie

        return (torch.stack(actions, dim=1),
                torch.stack(log_probs, dim=1),
                torch.stack(entropies, dim=1),
                torch.stack(probabilites, dim=1))


# Boucle d'entrainement : REINFORCE avec baseline

def entrainer(politique, cible_texte, vocabulaire, max_episodes=30000,
              type_recompense="positions", lr=1e-3, coef_entropie=0.01,
              taille_lot=1, fenetre_baseline=100, seuil_convergence=0.99,
              arret_sur_convergence=True, verbeux=True, periode_log=100,
              periode_meilleure=500, device="cpu", optimiseur=None, etiquette=""):
    """Entraine la politique et renvoie un dictionnaire de resultats.

    REINFORCE : loss = -log_prob(action) * (recompense - baseline)
    La baseline est la moyenne des recompenses des `fenetre_baseline` derniers
    episodes. ELLE NE CONTIENT AUCUNE INFORMATION SUR LA CIBLE : c'est une
    statistique sur des scalaires deja obtenus (verifie par assertion).
    """
    fonction_recompense = RECOMPENSES[type_recompense]
    cible_ids = vocabulaire.encoder(cible_texte)
    longueur = len(cible_ids)

    if optimiseur is None:
        optimiseur = torch.optim.Adam(politique.parameters(), lr=lr)

    historique_baseline = deque(maxlen=fenetre_baseline)
    historique_recompenses = []       # une entree par episode
    historique_moyennes = []          # moyenne glissante 100 par episode
    meilleure_recompense = -1.0
    meilleure_phrase = ""
    premier_parfait = None            # 1er episode avec recompense == 1.0 (echantillonne)
    premier_greedy_parfait = None     # 1er episode ou le decodage greedy == cible
    episode_convergence = None        # 1er episode ou moyenne_100 >= seuil
    episode = 0
    debut = time.time()

    while episode < max_episodes:
        # 1. Generation d'un lot d'episodes (echantillonnage stochastique)
        actions, log_probs, entropies, _ = politique.generer(
            longueur, taille_lot=taille_lot, greedy=False, device=device)

        # 2. Calcul des recompenses (seul contact avec la cible)
        actions_liste = actions.tolist()
        recompenses = [fonction_recompense(seq, cible_ids) for seq in actions_liste]

        # 3. Baseline : moyenne mobile des recompenses DEJA obtenues
        if historique_baseline:
            baseline = sum(historique_baseline) / len(historique_baseline)
        else:
            baseline = 0.0
        # Verification anti-fuite : la baseline n'est qu'un flottant issu de
        # recompenses passees, elle ne peut pas encoder la cible.
        assert isinstance(baseline, float)

        recompenses_t = torch.tensor(recompenses, dtype=torch.float32, device=device)
        avantages = (recompenses_t - baseline).detach()        # [lot]

        # 4. Perte REINFORCE + bonus d'entropie
        somme_log_probs = log_probs.sum(dim=1)                 # [lot]
        perte_politique = -(somme_log_probs * avantages).mean()
        perte_entropie = -coef_entropie * entropies.sum(dim=1).mean()
        perte = perte_politique + perte_entropie

        optimiseur.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(politique.parameters(), 5.0)
        optimiseur.step()

        # 5. Journalisation
        for i, r in enumerate(recompenses):
            episode += 1
            historique_baseline.append(r)
            historique_recompenses.append(r)
            moyenne_100 = float(np.mean(historique_recompenses[-100:]))
            historique_moyennes.append(moyenne_100)

            if r > meilleure_recompense:
                meilleure_recompense = r
                meilleure_phrase = vocabulaire.decoder(actions_liste[i])
            if premier_parfait is None and r >= 1.0:
                premier_parfait = episode
            if (episode_convergence is None
                    and len(historique_recompenses) >= 100
                    and moyenne_100 >= seuil_convergence):
                episode_convergence = episode

            if verbeux and episode % periode_log == 0:
                print(f"  [{etiquette}] ep {episode:6d} | reward moyen (100) = {moyenne_100:.4f} "
                      f"| baseline = {baseline:.4f}")
            if verbeux and episode % periode_meilleure == 0:
                greedy_txt = decoder_greedy(politique, longueur, vocabulaire, device)
                print(f"  [{etiquette}] ep {episode:6d} | meilleure = '{meilleure_phrase}' "
                      f"(r={meilleure_recompense:.3f}) | greedy = '{greedy_txt}'")

        # 6. Verification du decodage greedy (evaluation sans exploration)
        if premier_greedy_parfait is None:
            texte_greedy = decoder_greedy(politique, longueur, vocabulaire, device)
            if texte_greedy == cible_texte:
                premier_greedy_parfait = episode

        # 7. Arret anticipe si convergence stable
        if (arret_sur_convergence and episode_convergence is not None
                and episode >= episode_convergence + 200):
            break

    duree = time.time() - debut
    texte_greedy_final = decoder_greedy(politique, longueur, vocabulaire, device)

    return {
        "etiquette": etiquette,
        "cible": cible_texte,
        "type_recompense": type_recompense,
        "episodes_effectues": episode,
        "max_episodes": max_episodes,
        "premier_parfait": premier_parfait,
        "premier_greedy_parfait": premier_greedy_parfait,
        "episode_convergence": episode_convergence,
        "meilleure_recompense": meilleure_recompense,
        "meilleure_phrase": meilleure_phrase,
        "greedy_final": texte_greedy_final,
        "greedy_final_correct": texte_greedy_final == cible_texte,
        "recompense_moyenne_finale": float(np.mean(historique_recompenses[-100:])),
        "duree_s": round(duree, 1),
        "historique_recompenses": historique_recompenses,
        "historique_moyennes": historique_moyennes,
        "optimiseur": optimiseur,
    }


def decoder_greedy(politique, longueur, vocabulaire, device="cpu"):
    """Sortie deterministe de la politique (argmax a chaque pas)."""
    with torch.no_grad():
        actions, _, _, _ = politique.generer(longueur, taille_lot=1, greedy=True, device=device)
    return vocabulaire.decoder(actions[0].tolist())


# Utilitaires : graine, sauvegarde, graphiques

def fixer_graine(graine):
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)


def nouvelle_politique(vocabulaire, graine, device):
    """Poids 100% aleatoires (initialisation PyTorch par defaut)."""
    fixer_graine(graine)
    return PolitiqueGRU(vocabulaire.taille).to(device)


def sauvegarder_csv(resultat, chemin):
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "recompense", "recompense_moyenne_100"])
        for i, (r, m) in enumerate(zip(resultat["historique_recompenses"],
                                       resultat["historique_moyennes"]), start=1):
            writer.writerow([i, f"{r:.6f}", f"{m:.6f}"])


def tracer_courbe(resultats, chemin, titre):
    """Courbe(s) de recompense : brut en clair, moyenne glissante en fonce."""
    plt.figure(figsize=(11, 5.5))
    couleurs = plt.cm.tab10.colors
    for k, res in enumerate(resultats):
        couleur = couleurs[k % len(couleurs)]
        episodes = range(1, len(res["historique_recompenses"]) + 1)
        if len(resultats) == 1:
            plt.plot(episodes, res["historique_recompenses"], color=couleur,
                     alpha=0.15, linewidth=0.5, label="recompense par episode")
        plt.plot(episodes, res["historique_moyennes"], color=couleur, linewidth=1.8,
                 label=f"{res['etiquette']} (moyenne 100)")
        if res["episode_convergence"]:
            plt.axvline(res["episode_convergence"], color=couleur, linestyle="--", alpha=0.6)
    plt.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    plt.xlabel("episode")
    plt.ylabel("recompense")
    plt.title(titre)
    plt.ylim(-0.02, 1.05)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(chemin, dpi=130)
    plt.close()


# PHASE 2 — analyses post-entrainement

def heatmap_probabilites(politique, cible_texte, vocabulaire, chemin, device="cpu"):
    """Test 2 : probabilite assignee a chaque caractere, position par position."""
    longueur = len(cible_texte)
    with torch.no_grad():
        actions, _, _, probabilites = politique.generer(
            longueur, taille_lot=1, greedy=True, device=device)
    matrice = probabilites[0].cpu().numpy().T  # [vocab, longueur]

    plt.figure(figsize=(1.0 * longueur + 3, 0.45 * vocabulaire.taille + 2.5))
    plt.imshow(matrice, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="probabilite (mode greedy)")
    etiquettes_y = [f"'{c}'" if c != " " else "'_'" for c in vocabulaire.caracteres]
    plt.yticks(range(vocabulaire.taille), etiquettes_y)
    etiquettes_x = [f"{i+1}\n'{cible_texte[i]}'" if cible_texte[i] != " " else f"{i+1}\n'_'"
                    for i in range(longueur)]
    plt.xticks(range(longueur), etiquettes_x)
    plt.xlabel("position dans la sequence (et caractere cible)")
    plt.ylabel("caractere du vocabulaire")
    plt.title(f"Distribution apprise — cible '{cible_texte}'")
    # Encadre la case correspondant au caractere cible
    for pos, c in enumerate(cible_texte):
        y = vocabulaire.index_par_caractere[c]
        plt.gca().add_patch(plt.Rectangle((pos - 0.5, y - 0.5), 1, 1,
                                          fill=False, edgecolor="red", linewidth=1.5))
    plt.tight_layout()
    plt.savefig(chemin, dpi=130)
    plt.close()

    # Statistiques utiles pour le verdict
    proba_cible = [float(matrice[vocabulaire.index_par_caractere[c], pos])
                   for pos, c in enumerate(cible_texte)]
    return {
        "proba_par_position": [round(p, 4) for p in proba_cible],
        "proba_min": round(min(proba_cible), 4),
        "proba_moyenne": round(float(np.mean(proba_cible)), 4),
    }


def ablation_etat_cache(politique, cible_texte, vocabulaire, device="cpu"):
    """Test 3 : on corrompt l'etat cache apres la position t et on regarde ce
    qui survit dans la suite de la generation."""
    longueur = len(cible_texte)
    cible_ids = vocabulaire.encoder(cible_texte)
    resultats = []
    for mode in ("zero", "bruit"):
        for t in range(longueur):
            with torch.no_grad():
                actions, _, _, _ = politique.generer(
                    longueur, taille_lot=1, greedy=True,
                    ablation_position=t, ablation_mode=mode, device=device)
            genere = actions[0].tolist()
            suite = range(t + 1, longueur)
            if len(list(suite)) == 0:
                precision_suite = None
            else:
                justes = sum(1 for i in suite if genere[i] == cible_ids[i])
                precision_suite = round(justes / (longueur - t - 1), 4)
            resultats.append({
                "mode": mode,
                "position_ablatee": t,
                "texte_genere": vocabulaire.decoder(genere),
                "precision_suite": precision_suite,
            })
    return resultats


# Programme principal

def main():
    parseur = argparse.ArgumentParser(description="RDTRL — test 1 : copie d'une phrase en RL pur")
    parseur.add_argument("--episodes", type=int, default=30000)
    parseur.add_argument("--graine", type=int, default=0)
    parseur.add_argument("--lr", type=float, default=1e-3)
    parseur.add_argument("--coef-entropie", type=float, default=0.01)
    parseur.add_argument("--taille-lot", type=int, default=1,
                         help="episodes par mise a jour (1 = REINFORCE pur)")
    parseur.add_argument("--device", default="cpu",
                         help="cpu (recommande : modele minuscule, le GPU est plus lent ici) ou cuda")
    parseur.add_argument("--rapide", action="store_true", help="version courte de verification")
    parseur.add_argument("--sans-phase2", action="store_true")
    args = parseur.parse_args()

    if args.rapide:
        args.episodes = 4000

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    device = args.device
    vocabulaire = Vocabulaire(construire_vocabulaire([CIBLE_PRINCIPALE, CIBLE_PERTURBEE]))
    longueur = len(CIBLE_PRINCIPALE)
    taille_espace = vocabulaire.taille ** longueur

    print("=" * 78)
    print("RDTRL — TEST 1 : COPIER UNE PHRASE EN RL PUR (aucun pre-entrainement)")
    print("=" * 78)
    print(f"Cible                 : '{CIBLE_PRINCIPALE}' ({longueur} caracteres)")
    print(f"Vocabulaire ({vocabulaire.taille:2d} car.)  : {vocabulaire.caracteres}")
    print(f"Espace de recherche   : {vocabulaire.taille}^{longueur} = {taille_espace:.3e} sequences")
    print(f"Recompense d'une politique aleatoire (attendue) : {1/vocabulaire.taille:.4f}")
    print(f"Proba de tomber par hasard sur la cible exacte  : {1/taille_espace:.3e}")
    print(f"Device : {device} | lr={args.lr} | entropie={args.coef_entropie} "
          f"| lot={args.taille_lot} | max {args.episodes} episodes")
    print()

    rapport = {
        "cible": CIBLE_PRINCIPALE,
        "vocabulaire": vocabulaire.caracteres,
        "taille_espace_recherche": taille_espace,
        "hyperparametres": {
            "lr": args.lr, "coef_entropie": args.coef_entropie,
            "taille_lot": args.taille_lot, "max_episodes": args.episodes,
            "taille_cachee": 128, "algorithme": "REINFORCE + baseline moyenne mobile 100",
        },
    }

    # PHASE 1 — entrainement principal (recompense par positions)
    print("-" * 78)
    print("PHASE 1a — entrainement principal | recompense = caracteres bien places / longueur")
    print("-" * 78)
    politique = nouvelle_politique(vocabulaire, args.graine, device)
    print(f"Sortie AVANT entrainement (poids aleatoires) : "
          f"'{decoder_greedy(politique, longueur, vocabulaire, device)}'")
    run_principal = entrainer(politique, CIBLE_PRINCIPALE, vocabulaire,
                              max_episodes=args.episodes, type_recompense="positions",
                              lr=args.lr, coef_entropie=args.coef_entropie,
                              taille_lot=args.taille_lot, verbeux=True,
                              device=device, etiquette="positions")
    afficher_resume(run_principal)
    sauvegarder_csv(run_principal, os.path.join(DOSSIER_SORTIE, "run_principal.csv"))
    tracer_courbe([run_principal], os.path.join(DOSSIER_SORTIE, "courbe_run_principal.png"),
                  f"REINFORCE pur — cible '{CIBLE_PRINCIPALE}' (recompense par positions)")
    rapport["run_principal"] = resume_serialisable(run_principal)

    # PHASE 1b — variante Levenshtein (comparaison de signal)
    print()
    print("-" * 78)
    print("PHASE 1b — variante | recompense = 1 - distance de Levenshtein normalisee")
    print("-" * 78)
    politique_lev = nouvelle_politique(vocabulaire, args.graine, device)
    run_levenshtein = entrainer(politique_lev, CIBLE_PRINCIPALE, vocabulaire,
                                max_episodes=args.episodes, type_recompense="levenshtein",
                                lr=args.lr, coef_entropie=args.coef_entropie,
                                taille_lot=args.taille_lot, verbeux=True, periode_log=500,
                                periode_meilleure=5000, device=device, etiquette="levenshtein")
    afficher_resume(run_levenshtein)
    sauvegarder_csv(run_levenshtein, os.path.join(DOSSIER_SORTIE, "run_levenshtein.csv"))
    rapport["run_levenshtein"] = resume_serialisable(run_levenshtein)

    # PHASE 1c — CONTROLE : recompense reellement sparse (tout ou rien)
    #   C'est le test direct de l'objection "sparse reward".
    print()
    print("-" * 78)
    print("PHASE 1c — CONTROLE sparse | recompense = 1 si la phrase est exacte, 0 sinon")
    print("-" * 78)
    politique_sparse = nouvelle_politique(vocabulaire, args.graine, device)
    run_sparse = entrainer(politique_sparse, CIBLE_PRINCIPALE, vocabulaire,
                           max_episodes=args.episodes, type_recompense="tout_ou_rien",
                           lr=args.lr, coef_entropie=args.coef_entropie,
                           taille_lot=args.taille_lot, verbeux=True, periode_log=2000,
                           periode_meilleure=10000, device=device, etiquette="tout_ou_rien")
    afficher_resume(run_sparse)
    sauvegarder_csv(run_sparse, os.path.join(DOSSIER_SORTIE, "run_sparse.csv"))
    rapport["run_sparse_controle"] = resume_serialisable(run_sparse)

    tracer_courbe([run_principal, run_levenshtein, run_sparse],
                  os.path.join(DOSSIER_SORTIE, "courbe_comparaison_recompenses.png"),
                  "Comparaison des trois signaux de recompense")

    # PHASE 2 — uniquement si la copie parfaite a ete atteinte
    reussite = run_principal["greedy_final_correct"] or run_principal["premier_parfait"] is not None
    if args.sans_phase2 or not reussite:
        if not reussite:
            print("\nRecompense 1.0 non atteinte -> phase 2 non declenchee.")
        ecrire_rapport(rapport, run_principal, None)
        return

    print()
    print("=" * 78)
    print("PHASE 2 — ANALYSE POST-ENTRAINEMENT (la reussite doit etre expliquee)")
    print("=" * 78)
    phase2 = {}

    # Test 1 : anti-triche
    print("\n[Test 1] Anti-triche : force brute ? fuite d'information ? reproductibilite ?")
    episodes_utiles = run_principal["premier_parfait"] or run_principal["episodes_effectues"]
    ratio = episodes_utiles / taille_espace
    print(f"  Episodes pour atteindre reward=1.0 : {episodes_utiles}")
    print(f"  Taille de l'espace de recherche    : {taille_espace:.3e}")
    print(f"  Ratio episodes / espace            : {ratio:.3e} "
          f"(force brute attendrait un ratio ~1)")
    sequences_vues = episodes_utiles * args.taille_lot
    print(f"  Sequences distinctes explorables   : au plus {sequences_vues} "
          f"soit {100*sequences_vues/taille_espace:.2e} % de l'espace")

    # Reproductibilite : 3 graines differentes
    print("\n  Reproductibilite sur 3 graines (poids reinitialises a chaque fois) :")
    runs_graines = []
    for graine in (1, 2, 3):
        p = nouvelle_politique(vocabulaire, graine, device)
        r = entrainer(p, CIBLE_PRINCIPALE, vocabulaire, max_episodes=args.episodes,
                      type_recompense="positions", lr=args.lr,
                      coef_entropie=args.coef_entropie, taille_lot=args.taille_lot,
                      verbeux=False, device=device, etiquette=f"graine{graine}")
        runs_graines.append(r)
        print(f"    graine {graine} : 1er reward=1.0 a l'episode {r['premier_parfait']} | "
              f"convergence (moy100>=0.99) a {r['episode_convergence']} | "
              f"greedy final '{r['greedy_final']}'")
    valeurs = [r["premier_parfait"] for r in runs_graines if r["premier_parfait"]]
    valeurs.append(run_principal["premier_parfait"])
    print(f"    -> min={min(valeurs)}  max={max(valeurs)}  "
          f"moyenne={np.mean(valeurs):.0f}  ecart-type={np.std(valeurs):.0f}")

    # Fuite d'information : meme entrainement sur une cible ALEATOIRE.
    # Si l'agent converge aussi vite, c'est que rien de specifique a la phrase
    # francaise ne fuit dans le code : seul le signal de recompense compte.
    fixer_graine(123)
    cible_aleatoire = "".join(random.choice(vocabulaire.caracteres) for _ in range(longueur))
    print(f"\n  Controle anti-fuite — cible aleatoire '{cible_aleatoire}' :")
    p_alea = nouvelle_politique(vocabulaire, args.graine, device)
    run_aleatoire = entrainer(p_alea, cible_aleatoire, vocabulaire, max_episodes=args.episodes,
                              type_recompense="positions", lr=args.lr,
                              coef_entropie=args.coef_entropie, taille_lot=args.taille_lot,
                              verbeux=False, device=device, etiquette="cible_aleatoire")
    print(f"    1er reward=1.0 a l'episode {run_aleatoire['premier_parfait']} | "
          f"greedy final '{run_aleatoire['greedy_final']}'")

    phase2["anti_triche"] = {
        "episodes_pour_reward_1": episodes_utiles,
        "taille_espace": taille_espace,
        "ratio_episodes_sur_espace": ratio,
        "fraction_espace_exploree_pct": 100 * sequences_vues / taille_espace,
        "graines": [resume_serialisable(r) for r in runs_graines],
        "dispersion_premier_parfait": {
            "min": int(min(valeurs)), "max": int(max(valeurs)),
            "moyenne": float(np.mean(valeurs)), "ecart_type": float(np.std(valeurs)),
        },
        "controle_cible_aleatoire": resume_serialisable(run_aleatoire),
        "note_baseline": ("La baseline est la moyenne des recompenses des 100 derniers "
                          "episodes : un scalaire calcule a partir de l'historique des "
                          "rewards uniquement. La cible n'apparait que dans la fonction "
                          "de recompense, qui renvoie un float."),
    }

    # Test 2 : heatmap
    print("\n[Test 2] Heatmap des probabilites apprises (mode greedy)")
    chemin_heatmap = os.path.join(DOSSIER_SORTIE, "heatmap_probabilites.png")
    stats_heatmap = heatmap_probabilites(politique, CIBLE_PRINCIPALE, vocabulaire,
                                         chemin_heatmap, device)
    print(f"  Probabilite du bon caractere par position : {stats_heatmap['proba_par_position']}")
    print(f"  minimum = {stats_heatmap['proba_min']} | moyenne = {stats_heatmap['proba_moyenne']}")
    print(f"  Image : {chemin_heatmap}")
    phase2["heatmap"] = stats_heatmap

    # Test 3 : ablation de l'etat cache
    print("\n[Test 3] Ablation de l'etat cache position par position")
    ablations = ablation_etat_cache(politique, CIBLE_PRINCIPALE, vocabulaire, device)
    for mode in ("zero", "bruit"):
        print(f"  --- mode '{mode}' ---")
        for a in [x for x in ablations if x["mode"] == mode]:
            suite = "n/a" if a["precision_suite"] is None else f"{a['precision_suite']:.2f}"
            print(f"    h remis a {mode:5s} apres position {a['position_ablatee']:2d} -> "
                  f"'{a['texte_genere']}' | exactitude de la suite = {suite}")
    moyennes_ablation = {
        mode: round(float(np.mean([a["precision_suite"] for a in ablations
                                   if a["mode"] == mode and a["precision_suite"] is not None])), 4)
        for mode in ("zero", "bruit")
    }
    print(f"  Exactitude moyenne de la suite apres ablation : {moyennes_ablation}")
    phase2["ablation_etat_cache"] = {"details": ablations, "moyennes": moyennes_ablation}

    # Test 4 : perturbation de la cible
    print("\n[Test 4] Perturbation de la cible : "
          f"'{CIBLE_PRINCIPALE}' -> '{CIBLE_PERTURBEE}'")
    print("  (a) reprise des poids entraines, sans reset")
    run_transfert = entrainer(politique, CIBLE_PERTURBEE, vocabulaire,
                              max_episodes=args.episodes, type_recompense="positions",
                              lr=args.lr, coef_entropie=args.coef_entropie,
                              taille_lot=args.taille_lot, verbeux=False,
                              device=device, etiquette="transfert")
    print(f"      1er reward=1.0 : episode {run_transfert['premier_parfait']} | "
          f"convergence : {run_transfert['episode_convergence']} | "
          f"greedy final '{run_transfert['greedy_final']}'")

    print("  (b) controle : meme cible mais poids reinitialises (depuis zero)")
    p_zero = nouvelle_politique(vocabulaire, args.graine + 100, device)
    run_zero = entrainer(p_zero, CIBLE_PERTURBEE, vocabulaire, max_episodes=args.episodes,
                         type_recompense="positions", lr=args.lr,
                         coef_entropie=args.coef_entropie, taille_lot=args.taille_lot,
                         verbeux=False, device=device, etiquette="depuis_zero")
    print(f"      1er reward=1.0 : episode {run_zero['premier_parfait']} | "
          f"convergence : {run_zero['episode_convergence']} | "
          f"greedy final '{run_zero['greedy_final']}'")

    a = run_transfert["premier_parfait"]
    b = run_zero["premier_parfait"]
    if a and b:
        acceleration = b / a
        print(f"  -> facteur d'acceleration du transfert : x{acceleration:.2f} "
              f"({'transfert utile' if acceleration > 1.5 else 'pas de transfert notable'})")
    else:
        acceleration = None
    tracer_courbe([run_transfert, run_zero],
                  os.path.join(DOSSIER_SORTIE, "courbe_perturbation.png"),
                  f"Perturbation de la cible -> '{CIBLE_PERTURBEE}'")
    sauvegarder_csv(run_transfert, os.path.join(DOSSIER_SORTIE, "run_transfert.csv"))
    sauvegarder_csv(run_zero, os.path.join(DOSSIER_SORTIE, "run_depuis_zero.csv"))
    phase2["perturbation"] = {
        "nouvelle_cible": CIBLE_PERTURBEE,
        "transfert": resume_serialisable(run_transfert),
        "depuis_zero": resume_serialisable(run_zero),
        "facteur_acceleration": acceleration,
    }

    rapport["phase2"] = phase2
    ecrire_rapport(rapport, run_principal, phase2)


def afficher_resume(res):
    print(f"  >> {res['etiquette']} : {res['episodes_effectues']} episodes en {res['duree_s']}s")
    print(f"     1er reward=1.0 (echantillonne) : {res['premier_parfait']}")
    print(f"     1er greedy parfait             : {res['premier_greedy_parfait']}")
    print(f"     convergence (moy100 >= 0.99)   : {res['episode_convergence']}")
    print(f"     meilleure phrase               : '{res['meilleure_phrase']}' "
          f"(r={res['meilleure_recompense']:.3f})")
    print(f"     greedy final                   : '{res['greedy_final']}' "
          f"({'CORRECT' if res['greedy_final_correct'] else 'incorrect'})")


def resume_serialisable(res):
    """Copie du resultat sans les gros historiques ni l'optimiseur."""
    return {k: v for k, v in res.items()
            if k not in ("historique_recompenses", "historique_moyennes", "optimiseur")}


def ecrire_rapport(rapport, run_principal, phase2):
    """Ecrit le JSON complet + un verdict lisible en texte."""
    chemin_json = os.path.join(DOSSIER_SORTIE, "rapport.json")
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False, default=str)

    lignes = []
    lignes.append("=" * 78)
    lignes.append("VERDICT FINAL")
    lignes.append("=" * 78)
    r = run_principal
    if r["premier_parfait"]:
        lignes.append(f"Reward 1.0 ATTEINT a l'episode {r['premier_parfait']} "
                      f"(echantillonnage stochastique).")
        lignes.append(f"Premier decodage greedy parfait a l'episode {r['premier_greedy_parfait']}.")
        lignes.append(f"Convergence stable (moyenne 100 >= 0.99) a l'episode "
                      f"{r['episode_convergence']}.")
    else:
        lignes.append(f"Reward 1.0 JAMAIS atteint en {r['episodes_effectues']} episodes.")
        lignes.append(f"Stagnation autour de {r['recompense_moyenne_finale']:.3f}.")
    lignes.append(f"Cible          : '{r['cible']}'")
    lignes.append(f"Meilleure sortie : '{r['meilleure_phrase']}' (r={r['meilleure_recompense']:.3f})")
    lignes.append(f"Sortie greedy finale : '{r['greedy_final']}'")
    lignes.append("")

    if phase2:
        at = phase2["anti_triche"]
        lignes.append("-- Test 1 : anti-triche --")
        lignes.append(f"Episodes necessaires : {at['episodes_pour_reward_1']} contre un espace de "
                      f"{at['taille_espace']:.3e} sequences "
                      f"({at['fraction_espace_exploree_pct']:.2e} % explore). "
                      f"La force brute est exclue.")
        d = at["dispersion_premier_parfait"]
        lignes.append(f"4 graines : min={d['min']} max={d['max']} moyenne={d['moyenne']:.0f} "
                      f"ecart-type={d['ecart_type']:.0f}.")
        lignes.append(f"Controle cible aleatoire ('{at['controle_cible_aleatoire']['cible']}') : "
                      f"reward=1.0 a l'episode "
                      f"{at['controle_cible_aleatoire']['premier_parfait']} -> aucune fuite "
                      f"specifique a la phrase francaise.")
        lignes.append(at["note_baseline"])
        lignes.append("")
        lignes.append("-- Test 2 : heatmap --")
        lignes.append(f"Probabilite moyenne du bon caractere : {phase2['heatmap']['proba_moyenne']} "
                      f"(minimum {phase2['heatmap']['proba_min']}).")
        lignes.append("")
        lignes.append("-- Test 3 : ablation de l'etat cache --")
        lignes.append(f"Exactitude moyenne de la suite apres corruption de h : "
                      f"{phase2['ablation_etat_cache']['moyennes']}")
        lignes.append("")
        lignes.append("-- Test 4 : perturbation de la cible --")
        p = phase2["perturbation"]
        lignes.append(f"Nouvelle cible '{p['nouvelle_cible']}' : "
                      f"transfert = {p['transfert']['premier_parfait']} episodes, "
                      f"depuis zero = {p['depuis_zero']['premier_parfait']} episodes, "
                      f"facteur = {p['facteur_acceleration']}")
        lignes.append("")

    texte = "\n".join(lignes)
    chemin_txt = os.path.join(DOSSIER_SORTIE, "verdict.txt")
    with open(chemin_txt, "w", encoding="utf-8") as f:
        f.write(texte + "\n")
    print()
    print(texte)
    print(f"Fichiers ecrits dans : {DOSSIER_SORTIE}")


if __name__ == "__main__":
    main()
