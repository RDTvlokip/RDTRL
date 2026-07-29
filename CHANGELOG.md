# Changelog — RDTRL

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versionnage manuel.

## [0.2.0] — 2026-07-29

### Ajouté — Test 2 : apprendre une grammaire en RL pur

**`grammaire.py`** — grammaire formelle écrite à la main, aucune IA. Parser
déterministe vérifiant structure et accords (genre, nombre) sans jamais comparer
à une phrase cible. Deux variantes : courte (`dét nom verbe`, 20 tokens, espace
8 000, 48 phrases valides, 0,600 % au hasard) et longue (`dét adj nom verbe adv`,
31 tokens, espace 28,6 M, 288 valides, 0,001 %). Comptage analytique vérifié par
force brute.

**`rl_grammaire.py`** — politique GRU sur tokens-mots, REINFORCE + baseline,
balayage du coefficient d'entropie, contrôle tout-ou-rien, tests de
généralisation par exclusion de combinaison et de token. L'espace court étant
énumérable, toutes les mesures de diversité sont **exactes** : masse valide,
modes effectifs 2^H, uniformité, répartition entre familles, conditionnelles
obtenues par marginalisation de la loi jointe.

**Scripts de diagnostic** — `balayage_graines.py` (multi-graines),
`sonde_capacite.py` (représentabilité), `optimum_gibbs.py` (optimum en forme
close, taxe de mise en forme), `verifier_dominance.py`, `gradient_exact.py`
(signal d'ordre 1 et gradient analytique), `parametrisation_et_recuit.py`
(tabulaire vs GRU, recuit de β), `stabilite_et_trajectoire.py`,
`trajectoire_et_structure.py` (ACP, ANOVA, fonctionnelles conservées),
`localisation_effondrement.py`.

**`ANALYSE_TEST2.md`** et **`CARNET.md`** — résultats, et le raisonnement avec les
hypothèses réfutées datées.

### Résultats

- **99,9 % de grammaticalité sans aucune règle apprise.** P(nom accordé | dét) =
  0,333 = 2/6 déterminants, P(verbe accordé | nom) = 0,500 = 4/8 noms, pour
  toutes les graines du plateau. L'agent se réfugie dans une sous-langue au
  pluriel où l'accord est vacuellement satisfait.
- **Le contrôle tout-ou-rien réussit** sur la grammaire courte (99,58 %, mieux que
  la graduée) et échoue sur la longue (0 %). La variable qui décide est le taux de
  réussite au hasard, pas la forme du signal.
- **Isolation causale complète de l'effondrement de mode** : ni le bruit
  d'échantillonnage (gradient exact → 12 modes quand même), ni l'objectif
  (tabulaire → 48,0 modes exactement), ni la capacité (sonde → 100 %, 48 modes),
  ni l'instabilité (parti de l'idéal, il s'y maintient). **La factorisation
  autorégressive est la coupable** : les conditionnelles partagent des paramètres.
- **Spectre ANOVA de la récompense** : graduée 76,1 % d'ordre 1, tout-ou-rien
  4,0 % d'ordre 1 et 65,5 % d'ordre 3. Le façonnage déplace la variance vers
  l'ordre 1, seul ordre visible au gradient à politique uniforme.
- **Correctif validé : le recuit de β** (0,2 → 0,01) donne 99,97 % de validité
  ET 45,3 modes sur 48 avec les deux branches, dominant les deux régimes à β
  constant. Reproduit avec un second calendrier.
- **L'échantillonnage par rejet depuis le réseau non entraîné** (100 %, ~47,5
  modes) bat tous les entraînements du plateau sur les deux axes.

### Corrigé

- Première version de `stabilite_et_trajectoire.py` : le token imposé l'était
  après génération, donc la suite restait conditionnée sur un autre token. Test
  refait dans `localisation_effondrement.py`.
- Premier balayage à graine unique : le coefficient retenu (0,08) l'avait été sur
  la graine 0, justement celle qui reste mono-branche à cette valeur. Balayage
  refait sur 3 graines.

## [0.1.0] — 2026-07-29

### Ajouté

**`rl_copie.py` — Test 1 : copier une phrase fixe en RL pur**

Script autonome qui entraîne une politique depuis des poids 100 % aléatoires à
reproduire une phrase cible, avec pour seul signal une récompense scalaire.
Aucun pré-entraînement, aucune paire entrée/sortie, aucune connaissance de la
langue injectée.

- **Politique** : GRU 1 couche, `hidden_size=128`, embedding 32, tête linéaire
  vers le vocabulaire. Génération autorégressive caractère par caractère, sans
  teacher forcing (le caractère échantillonné au pas *t-1* est réinjecté au pas
  *t*). Token de début réservé, hors vocabulaire de sortie.
- **Vocabulaire** : union des caractères des deux cibles (`le chat dort` et
  `le chien dort`) + l'espace, soit 12 caractères. L'union est prise dès le
  départ pour que l'architecture reste identique entre la phase 1 et le test de
  perturbation. Espace de recherche : 12^12 ≈ 8,9 × 10^12.
- **Algorithme** : REINFORCE avec baseline = moyenne mobile des récompenses des
  100 derniers épisodes. `loss = -log_prob(action) * (reward - baseline)`,
  bonus d'entropie (coef 0,01), Adam `lr=1e-3`, clipping du gradient à 5,0.
- **Trois fonctions de récompense** :
  - `positions` — caractères bien placés / longueur (signal dense) ;
  - `levenshtein` — 1 − distance d'édition normalisée (tolérant aux décalages) ;
  - `tout_ou_rien` — 1 seulement si la phrase est exacte. Contrôle qui teste
    directement l'objection « sparse reward ».
- **Critères mesurés** : premier épisode avec récompense 1.0 (échantillonné),
  premier décodage greedy parfait, épisode de convergence (moyenne glissante
  100 ≥ 0,99). Arrêt anticipé 200 épisodes après convergence.
- **Sorties** dans `resultats/` : CSV par run, courbes matplotlib, heatmap,
  `rapport.json` et `verdict.txt`.

**Phase 2 — analyses post-entraînement** (déclenchées uniquement si la copie
parfaite est atteinte)

1. *Anti-triche* : ratio épisodes / taille de l'espace de recherche, fraction de
   l'espace réellement explorée, reproductibilité sur 4 graines, et contrôle
   anti-fuite consistant à réentraîner sur une **cible aléatoire** de même
   longueur — si la convergence est aussi rapide, rien de spécifique à la phrase
   française ne fuit dans le code.
2. *Heatmap* des probabilités apprises (position × caractère, mode greedy), avec
   la case du caractère cible encadrée et les statistiques par position.
3. *Ablation de l'état caché* : `h` est remis à zéro ou bruité après chaque
   position, et on mesure l'exactitude de la suite de la génération.
4. *Perturbation de la cible* : `le chat dort` → `le chien dort`, en reprenant
   les poids entraînés (sans reset), comparé à un contrôle réentraîné depuis
   zéro sur la même nouvelle cible. Le rapport du nombre d'épisodes donne le
   facteur d'accélération du transfert.

**`test4_controle.py` — contrôle du test 4**

Le facteur ×1,74 mesuré au test de perturbation est ambigu : `le chat dort` et
`le chien dort` partagent 5 positions sur 13, donc l'accélération peut n'être que
la réutilisation du préfixe littéral. Ce script rejoue le transfert vers une
cible de même longueur sans aucun caractère commun à la même position, ce qui
sépare les deux interprétations. Sauvegarde aussi les poids entraînés dans
`resultats/politique_le_chat_dort.pt`.

**`bench_device.py`** — mesure du temps par épisode sur CPU et GPU selon la
taille de lot, pour justifier le défaut `--device cpu`.

**`ANALYSE.md`** — résultats et verdict argumenté de l'expérience.

### Résultats

- Copie parfaite atteinte à l'**épisode 1 639** ; greedy parfait à 1 846 ;
  convergence stable à 2 405. Reproductible sur 4 graines (1 360–1 702, σ = 132).
- **Contrôle sparse : échec total.** Récompense tout-ou-rien, 30 000 épisodes,
  récompense exactement nulle du début à la fin, aucun succès.
- Ablation de l'état caché : survie moyenne 0,67 (zéro) / 0,71 (bruit).
  L'agent a appris une table caractère→caractère, `h` ne servant qu'à
  désambiguïser les caractères répétés.
- Transfert : ×1,74 avec préfixe partagé, **×0,91 sans recouvrement** → aucune
  structure abstraite réutilisable, mémorisation position par position.

Conclusion : le succès vient de la décomposabilité de la récompense, pas d'une
capacité du RL à traverser l'espace de recherche. L'objection du sparse reward
est confirmée. Détail dans [ANALYSE.md](ANALYSE.md).

### Notes

- `--device cpu` par défaut : le modèle est trop petit pour que le GPU soit
  rentable à cette taille de lot (surcoût de lancement des noyaux CUDA).
- `--rapide` limite à 4 000 épisodes pour vérifier que le script tourne.
