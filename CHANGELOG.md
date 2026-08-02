# Changelog — RDTRL

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versionnage manuel.

**DOI de concept, toutes versions confondues :**
[10.5281/zenodo.21726216](https://doi.org/10.5281/zenodo.21726216) — c'est celui
du badge et du BibTeX, il résout toujours vers la version la plus récente. Les
DOI de version, propres à une release donnée et figés, sont indiqués sous chaque
entrée ci-dessous.

## [0.4.0] — 2026-07-31

**La version de la revue publique.** Quatre séries de critiques de
[dipankarsarkar](https://orcid.org/0000-0001-5431-6367) ont produit une borne en
forme close que je n'avais pas vue, fait tomber quatre de mes chiffres publiés,
et expliqué une anomalie que j'attribuais au non-déterminisme. Tout est dans
[docs/ARTICLE2.md](docs/ARTICLE2.md) et au carnet §7.10 à §7.13.

### Ajouté — le test de renversement, qui décide si le plafond est une loi

Le plafond de produit du §7.11 est calculé sur **mon** lexique. Une seule
expérience décide s'il s'agit d'une loi ou d'une coïncidence de vocabulaire :
construire une grammaire où les plafonds prennent d'autres **valeurs**, enregistrer
la prédiction avant de lancer, vérifier.

**Ma première version ne testait rien.** Elle déplaçait la neutralité de genre des
déterminants pluriels vers les singuliers, à vocabulaire, espace et coins
identiques. Or noms et verbes sont déjà symétriques en nombre, donc cette
permutation **est** le renommage `sg ↔ pl` : les deux grammaires sont isomorphes,
et 70 graines auraient rendu l'image miroir par construction. Conservée dans le
code comme contre-exemple documenté (`variante="renverse"`).

> **Un renommage peut permuter, il ne peut pas changer un rapport.** Un contrôle
> parfaitement symétrique est souvent parfaitement vide.

**La version qui teste** (`variante="trois_genres"`) : trois genres au lieu de
deux. Les deux coins contiennent 36 phrases valides, mais les plafonds valent
**36 et 12** au lieu de 12 et 24, soit un **rapport de 3 au lieu de 2** — qu'aucun
renommage ne peut produire, le plus grand produit étant un invariant
d'isomorphisme.

| coin | n | plafond prédit | max observé | dépassements | pile au plafond |
|---|---|---|---|---|---|
| singulier | 33 | **36** | **36,0** | **0** | 2 |
| pluriel | 37 | **12** | **12,0** | **0** | 7 |

| grammaire | rapport des plafonds | rapport des moyennes observées |
|---|---|---|
| standard, 2 genres | 2,0 | 1,82 |
| **trois genres** | **3,0** | **3,01** |

La moyenne suit le **rapport**, pas seulement l'ordre. Branche toujours à
pile ou face (33/37, p = 0,72) **malgré l'inversion des deux marginales d'ordre
1**, et `I(dét;nom)` nulle sur les 70.

### Changé — le chemin numérique de la ligne d'avantage

**Le défaut de `entrainer` passe de `float32` à `float64`.** La soustraction
`récompense − baseline` se fait désormais en Python, donc nativement en double,
puis on arrondit **une seule fois**. L'ancien chemin poussait les deux valeurs
dans torch, ce qui arrondissait la baseline **avant** de soustraire : deux
arrondis au lieu d'un.

```
  valeur exacte de r - baseline    0.08333333333333337
  arrondie une fois             -> 0.0833333358168602
  arrondie deux fois            -> 0.08333331346511841
```

Aucun compromis : le nouveau chemin est plus juste **et** mesuré **4× plus
rapide** sur cette ligne (4,6 µs contre 19,5 µs), parce qu'il fait une
soustraction Python au lieu d'une création de tenseur plus un noyau torch plus un
`detach`. Rien n'est stocké en double précision, le tenseur produit reste
float32.

Trouvé par dipankarsarkar. Le dépôt contenait **deux** comportements sans le
dire : `rl_grammaire.py:141` d'un côté, et six autres fichiers de l'autre. Six
fichiers de plus n'ont aucun entraînement échantillonné — gradient exact,
énumérations, formes closes — et ne sont donc pas concernés ; ce sont eux qui
portent le plafond de produit, l'optimum de Gibbs et les marginales d'ordre 1.

**Ce que ça change, et ce que ça ne change pas.** Vérifié sur 70 graines par
chemin : **37 / 33 des deux côtés**, même intervalle de Wilson, même p, **zéro
dépassement du plafond des deux côtés**, `I(dét;nom)` nulle des deux côtés. Les
conclusions du dépôt ne dépendent pas de l'arrondi. Ce qui bouge, c'est le
détail par graine : 21 runs sur 70 gardent le même nombre de modes, corrélation
0,68. **Le coin est décidé par l'initialisation, le remplissage par la
trajectoire.**

⚠️ **Les tableaux des versions 0.3.x ont été produits sur le chemin `float32`.**
Pour les reproduire à l'identique, passer `chemin_avantage="float32"`. Le
paramètre est explicite et conservé pour ça.

### Ajouté

- **`chemin_avantage.py`** — isole les deux chemins, mesure leur divergence sur
  le flux de récompenses réel, et trace les deux avec sonde exacte.
- **`relancer_float64.py`** — relance les scripts concernés en bornant le
  parallélisme, et archive `results_test2/` avant d'écraser.
- **Épinglage des threads dans `rl_grammaire.py`**, que 14 scripts importent, avec
  `RDTRL_THREADS` pour revenir en arrière. Auparavant deux fichiers seulement
  l'épinglaient, les autres dépendaient du shell.

- **`figure_comparaison.py`** — figure de synthèse en quatre panneaux : les deux
  chemins numériques graine par graine, le plafond jamais franchi, la profondeur
  de l'effondrement qui sépare gradient exact et échantillonné, et l'écart
  d'arrêt précoce sur 20 graines. Chaque panneau **affiche le chemin qui l'a
  produit**, calculé depuis la donnée chargée et non écrit en dur.
- **`figure_renversement.py`** — les 70 runs à trois genres contre leur plafond,
  et le test du rapport qu'un renommage ne peut pas passer.
- **`docs/ARTICLE2.md`** — l'article de cette version, avec la conversation citée
  mot pour mot à la demande de son auteur.

### Changé — les tableaux mono-graine deviennent multi-graines

Le balayage d'entropie de l'article 1 était à **3 graines** et sur l'ancien
chemin. Refait à **10 graines par β, 80 runs**, sur le chemin canonique. Les
chiffres bougent, ce qui était attendu : c'est un tableau mono-graine, et le
§4.2 dit depuis le début qu'un balayage mono-graine ne trace pas une frontière.

| β | validité % | modes / 48 | 2 branches | sg / pl |
|---|---|---|---|---|
| 0,0 | 100,0 ± 0,0 | 1,0 ± 0,0 | 0/10 | 6 / 4 |
| 0,01 | 100,0 ± 0,0 | 9,3 ± 5,6 | 0/10 | 4 / 6 |
| 0,02 | 99,7 ± 0,6 | 14,1 ± 5,5 | 0/10 | 3 / 7 |
| 0,05 | 97,0 ± 3,2 | 22,2 ± 2,0 | 0/10 | 4 / 6 |
| 0,08 | 86,4 ± 5,4 | **31,4 ± 10,9** | **5/10** | 4 / 6 |
| 0,12 | 58,6 ± 4,4 | 43,8 ± 1,3 | 10/10 | 7 / 3 |
| 0,2 | 21,4 ± 1,7 | 44,1 ± 1,9 | 10/10 | 3 / 7 |
| 0,35 | 5,4 ± 0,5 | 44,9 ± 1,1 | 10/10 | 4 / 6 |

La colonne sg/pl est nouvelle et confirme sur 80 runs de plus que **le choix de
branche est une pièce équilibrée à tous les β**. Deux chiffres de l'article 1
bougent aussi : le tout-ou-rien court passe de 99,58 % à 99,91 %, et la
**grammaire longue de 6,4 % à 15,8 %**.

### Corrigé

- **Cinq collisions de provenance**, toutes du même défaut : un artefact qui
  n'encode pas la dimension que le run fait varier.
  - poids du balayage `politique_b{β}_g{n}.pt` sans le chemin → **70 politiques
    écrasées** ;
  - motif de fusion `..._b0.02_*.json` ramassant l'autre chemin **et sa propre
    sortie** → 13 fichiers pour 6 tranches, graines comptées deux ou trois fois ;
  - sorties de `chemin_avantage` sans la plage de graines → deux tranches
    parallèles dans un seul fichier ;
  - `rapport.json` et `balayage_graines.json` qu'une relance aurait écrasés →
    attrapé avant, dossier archivé dans `results_test2_float32/` ;
  - étiquette de figure écrite **en dur** à côté d'un chargement avec repli → la
    figure allait annoncer un chemin en traçant les chiffres de l'autre.

  Correctifs : la dimension est dans le nom, le glob se termine par `_[0-9]*`
  pour exclure sa propre sortie, un garde-fou compte les doublons après fusion et
  le signale, et les étiquettes se calculent depuis la donnée réellement lue.

- **Métrique de saturation**, signalée au carnet §3.3 **deux jours** avant d'être
  corrigée. Un défaut noté et non corrigé donne l'impression d'être traité, donc
  plus personne ne le regarde : `H` était calculée sur les 8 noms et `H_max` sur
  les noms compatibles, donc la valeur pouvait dépasser 100 %, et un dépassement
  signalait une **fuite de masse** qui se lisait comme un succès. Séparée en
  `masse_accordee_pct` et `saturation_pct`, cette dernière bornée par
  construction.

- **Écart d'arrêt précoce du §7.9, retiré.** Annoncé à +12,5 modes depuis une
  seule graine, ramené à +5,38 par le changement de chemin, puis à une **médiane
  de +0,00 sur 20 graines**, 3 runs sur 20 seulement au-dessus d'un mode. Ce qui
  survit est conditionnel : les trois sont dans le coin pluriel, 3 sur 8 contre
  **0 sur 12** au singulier.

## [0.3.2] — 2026-07-31

DOI de version : [10.5281/zenodo.21726512](https://doi.org/10.5281/zenodo.21726512)

Version de métadonnées uniquement, aucun changement de code ni de résultat.

### Corrigé

- **Le badge et le BibTeX pointaient sur un DOI de version, pas sur le concept.**
  Zenodo frappe trois numéros ici : `21726216` pour le concept, qui résout
  toujours vers la dernière version et ne change jamais, puis un par release.
  Les archives 0.3.0 et 0.3.1 contiennent donc un README qui présente un DOI de
  version comme s'il représentait le dépôt. Corrigé à la source : le badge, le
  BibTeX et le champ `doi` de `CITATION.cff` portent désormais le concept, les
  deux DOI de version sont conservés en identifiants secondaires, et chaque
  entrée de ce fichier porte le sien.

À partir de cette version, publier une release ne demande plus de toucher au
README : le badge suit automatiquement. Il suffit d'ajouter la ligne « DOI de
version » sous la nouvelle entrée ci-dessous.

## [0.3.1] — 2026-07-31

DOI de version : [10.5281/zenodo.21726387](https://doi.org/10.5281/zenodo.21726387)

Version de métadonnées uniquement, aucun changement de code ni de résultat.

### Ajouté

- **Remerciement à Dipankar Sarkar** ([ORCID
  0000-0001-5431-6367](https://orcid.org/0000-0001-5431-6367)), dans le README,
  dans `CITATION.cff` et comme `contributor` structuré dans `.zenodo.json`. La
  borne de produit qui porte la version 0.3.0 est de lui : il a dérivé les
  marginales d'ordre 1 indépendamment depuis l'article publié, puis montré que
  les deux coins dégénérés contiennent le même nombre de phrases valides mais
  **pas le même plus grand produit**. Trois séries de ses critiques ont aussi
  corrigé une statistique qui mesurait la couverture de déterminants et non
  l'accord, une métrique de saturation qui pouvait dépasser 100 %, et une
  affirmation d'échantillon qui agrégeait 3 graines en 24 lignes.
- **`.zenodo.json`** — métadonnées structurées pour l'archivage, avec le champ
  `contributors` que `CITATION.cff` ne propose pas en version 1.2.0.

## [0.3.0] — 2026-07-31

DOI de version : [10.5281/zenodo.21726217](https://doi.org/10.5281/zenodo.21726217)

Version de référence pour archivage. Elle contient, en plus des tests 1 et 2, un
résultat quantitatif nouveau et **la correction de trois affirmations publiées en
0.2.0**.

### Ajouté — le plafond de produit et la mesure du couplage

**`sonde_ordre1.py`** — marginales `E[R | x_p = t]` en forme close, pour **toutes**
les positions et les deux grammaires, calculables avant tout entraînement. Deux
accidents de lexique orthogonaux, un par trait : le `None` de genre donne +0,0333
au déterminant pluriel, le déséquilibre 4 déterminants singuliers contre 2 donne
+0,0167 au nom singulier. Les marginales se contredisent, d'où une séquence
gloutonne d'ordre 1 **invalide**.

**`produit_et_saturation.py`** — plus grand ensemble **produit** entièrement valide
contenu dans chaque coin, par énumération exhaustive. Une politique sans couplage
dét → nom a un support produit, donc à validité 1 elle est bornée par ce nombre.

| | phrases valides | plus grand produit |
|---|---|---|
| courte, coin pluriel | 24 | **24** |
| courte, coin singulier | 24 | **12** |
| longue, chaque coin | 144 | **72** |

**`balayage_70_graines.py`** — 70 graines à condition unique, avec sauvegarde des
poids et de la masse par déterminant.

**`optimum_produit.py`** — optimisation exacte de `E[R] + β·H` sur la classe des
politiques sans couplage (trois lois indépendantes) contre la classe libre.

**`trajectoire_couplage.py`** — suivi pas à pas de `I(dét ; nom)`.

### Résultats

- **Le plafond de produit n'est jamais franchi** : 0 dépassement sur les 37 runs
  du coin singulier, et le résultat modal **est** le plafond (19 runs exactement
  à 12,0). Les modes effectifs sont des **produits d'entiers** — {2, 4, 6, 8, 12}
  et {6, 8, 12, 16, 18, 24}.
- **Le plafond est un plateau, pas un bassin** : le gradient exact tient 12,00
  modes à `I = 0` pendant mille pas, puis s'échappe vers 24,00 à `I = 0,998`. On
  en sort donc sans aucun bruit.
- **Ce qui sépare gradient exact et échantillonné est la profondeur de
  l'effondrement transitoire** : minimum 10,7 à 11,2 modes pour l'exact, **1,09 à
  1,88** pour l'échantillonné, qui écrase la politique sur une seule phrase avant
  de la reconstruire.
- **Aucun run n'acquiert la conditionnelle** : `I(dét;nom)` médiane 0,0000 bit,
  maximum 0,0377 sur 70, contre 1,0 nécessaire. Mesuré cette fois avec une
  statistique qui peut le dire.
- **Le choix de branche est une pièce équilibrée** : 37 singulier / 33 pluriel,
  Wilson 95 % [0,413 ; 0,641], p = 0,72 contre 1/2.

### Corrigé — affirmations de la version 0.2.0

- **« Isolation causale complète … la factorisation autorégressive est la
  coupable »** : faux tel quel. La cause se scinde en deux régimes. À β ≥ 0,05 le
  gradient exact atteint 48,0 modes et un partage 50/50, donc c'est la procédure
  échantillonnée qui échoue, pas la factorisation.
- **« L'agent se réfugie dans une sous-langue au pluriel »** : vrai d'une graine.
  Sur 70 graines, 37 vont au singulier et 33 au pluriel.
- **`P(nom accordé | dét) = 0,333`** : cette statistique était une moyenne **non
  pondérée** sur les six déterminants et vaut donc (déterminants émis)/6, pas un
  taux d'accord. Remplacée par `I(dét ; nom)`, plus `cond_det_pondere` et
  `determinants_emis`.
- **Métrique de saturation** : `H` était calculée sur les 8 noms et `H_max` sur
  les noms compatibles, donc la valeur pouvait dépasser 100 %, ce qui signalait
  une fuite de masse et se lisait comme un succès. Séparée en
  `masse_accordee_pct` et `saturation_pct`, cette dernière bornée par
  construction.

### Notes

Les hypothèses réfutées sont datées dans [docs/CARNET.md](docs/CARNET.md) §1,
huit à ce jour. Le détail des trois critiques extérieures et de mes erreurs est
en §7.10 à §7.12. L'évaluation de publiabilité, avec ce qui manque, est en §7.12.

## [0.2.0] — 2026-07-29

> **Corrigé en 0.3.0.** Trois affirmations de cette section sont fausses ou mal
> mesurées : l'isolation causale « complète », l'effondrement « au pluriel », et
> `P(nom accordé | dét) = 0,333`. Voir la section 0.3.0 ci-dessus.

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
