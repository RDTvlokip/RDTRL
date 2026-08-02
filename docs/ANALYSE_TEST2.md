# Test 2 — Apprendre une grammaire en RL pur : résultats et verdict

Expérience du 29/07/2026. Grammaire formelle écrite à la main, parser
déterministe, aucune IA en amont, aucune phrase cible. Politique GRU 128,
REINFORCE + baseline, poids aléatoires, aucune donnée.

| grammaire | structure | vocab | espace | phrases valides | validité au hasard |
|---|---|---|---|---|---|
| courte | dét nom verbe | 20 | 8 000 | 48 | 0,600 % |
| longue | dét adj nom verbe adv | 31 | 28 629 151 | 288 | 0,001 % |

L'espace court étant énumérable, **toutes les mesures ci-dessous sont exactes**,
pas estimées sur échantillon.

## Verdict en un paragraphe

L'agent atteint 99,9 % de grammaticalité **sans avoir appris la moindre règle
d'accord** : il se réfugie dans une sous-langue entièrement au pluriel où la
contrainte est vacuellement satisfaite. Le contrôle tout-ou-rien réussit aussi
bien que la récompense graduée sur la grammaire courte, et échoue totalement sur
la longue — la variable qui décide n'est donc pas la forme du signal mais le taux
de réussite au hasard. Enfin, la faible diversité observée n'est **ni** une
propriété de la tâche **ni** un défaut de l'objectif : le modèle sait représenter
et maintenir l'optimum, mais ne l'atteint pas. La cause se scinde en deux
régimes — à faible pression entropique c'est la **factorisation autorégressive**
qui bloque, même avec un gradient exact ; à β=0,05 ce blocage disparaît et c'est
la **procédure échantillonnée** qui échoue. Un recuit du coefficient d'entropie
corrige le problème.

## 1. Les quatre questions du cahier des charges

### 1.1 Taux de validité — oui, et il ne veut rien dire

Balayage du coefficient d'entropie β, 3 graines, 20 000 épisodes, mesures exactes.

**Refait le 31/07/2026 : 10 graines par β au lieu de 3, sur le chemin numérique
canonique (float64).** L'ancienne version à 3 graines venait de l'autre chemin ;
les deux tableaux sont donnés parce que l'écart entre eux est lui-même une
donnée sur la fragilité de ces chiffres.

| β | validité % | modes effectifs / 48 | graines couvrant les 2 branches | sg / pl |
|---|---|---|---|---|
| 0,0 | 100,0 ± 0,0 | 1,0 ± 0,0 | 0/10 | 6 / 4 |
| 0,01 | 100,0 ± 0,0 | 9,3 ± 5,6 | 0/10 | 4 / 6 |
| 0,02 | 99,7 ± 0,6 | 14,1 ± 5,5 | 0/10 | 3 / 7 |
| 0,05 | 97,0 ± 3,2 | 22,2 ± 2,0 | 0/10 | 4 / 6 |
| 0,08 | 86,4 ± 5,4 | **31,4 ± 10,9** | **5/10** | 4 / 6 |
| 0,12 | 58,6 ± 4,4 | 43,8 ± 1,3 | 10/10 | 7 / 3 |
| 0,2 | 21,4 ± 1,7 | 44,1 ± 1,9 | 10/10 | 3 / 7 |
| 0,35 | 5,4 ± 0,5 | 44,9 ± 1,1 | 10/10 | 4 / 6 |

La transition reste à β ≈ 0,08, et c'est là que l'écart-type des modes explose —
**±10,9** — parce que la moitié des graines couvre les deux familles et l'autre
moitié non. La colonne sg/pl confirme sur 80 runs ce que les 70 graines du §7.11
disaient : **le choix de branche est une pièce équilibrée à tous les β**.

<details>
<summary>Ancienne version, 3 graines, chemin float32</summary>

| β | validité % | modes effectifs / 48 | graines couvrant les 2 branches |
|---|---|---|---|
| 0,0 | 100,0 ± 0,0 | 1,0 ± 0,0 | 0/3 |
| 0,01 | 99,9 ± 0,1 | 6,0 ± 2,8 | 0/3 |
| 0,02 | 99,9 ± 0,1 | 14,1 ± 3,2 | 0/3 |
| 0,05 | 96,0 ± 2,6 | 22,5 ± 1,2 | 0/3 |
| 0,08 | 87,6 ± 6,9 | 37,4 ± 9,2 | 2/3 |
| 0,12 | 57,6 ± 1,7 | 45,3 ± 0,6 | 3/3 |
| 0,2 | 21,3 ± 0,9 | 43,6 ± 2,1 | 3/3 |
| 0,35 | 5,6 ± 0,7 | 44,1 ± 1,5 | 3/3 |

</details>

### 1.2 Diversité — l'agent verrouille une sous-famille, pas une phrase

Les 48 solutions se répartissent en deux familles de 24 : les phrases au
singulier et celles au pluriel. Sur tout le plateau (β ≤ 0,05), **aucune graine
ne visite les deux**. Le nombre de modes plafonne autour de 24, soit exactement
la taille d'une branche.

Le choix de branche n'est pas une loterie : il est **prédit par la structure de
la récompense**. Le signal marginal d'ordre 1 vaut 0,2944 pour les noms
singuliers contre 0,2778 pour les pluriels, soit +0,0167 en faveur du singulier.
Cause : le lexique contient **4 déterminants singuliers et seulement 2 pluriels**,
donc un déterminant tiré au hasard s'accorde en nombre avec un nom singulier
4 fois sur 6 contre 2 fois sur 6. C'est un déséquilibre involontaire de ma
conception, calculable avant tout entraînement, et il décide du résultat.

Confirmation : avec gradient exact (sans bruit), les deux graines testées partent
au singulier de façon déterministe. Le bruit d'échantillonnage est précisément ce
qui permet parfois de surmonter ce biais.

### 1.3 Généralisation — non, et le test littéral ne mesure rien

**Combinaison exclue** (`des fleurs` jamais récompensée, les deux tokens
entraînés par ailleurs) : P(`fleurs`|`des`) = 0,2248 après exclusion contre 0,2560
en moyenne pour les autres noms pluriels, ratio 0,878. La référence sans
exclusion donne 0,2861.

Lecture honnête : la dispersion entre noms **non exclus** va de 0,179 (`chiens`)
à 0,309 (`tables`), donc de 0,70 à 1,21 en ratio. **L'effet mesuré est plus petit
que la variabilité naturelle.** Une seule graine, pas de puissance statistique :
on ne peut pas conclure.

**Nom jamais vu** (version littérale de la spec) : P(verbe pluriel | `fleurs`
imposé, jamais entraîné) = 0,9966. Ce chiffre ne mesure rien — l'agent émet un
verbe pluriel **quel que soit le nom**, les noms singuliers vus donnant 0,0003 à
0,0149. La moyenne sur les noms vus est 0,4286, *sous* le hasard. Le test est
confondu par l'effondrement sur une branche.

### 1.4 Contrôle tout-ou-rien — il réussit, et ça invalide la leçon du test 1

| | validité | modes | P(nom accordé\|dét) |
|---|---|---|---|
| graduée, β=0,08 | 94,87 % | 24,4 | 0,632 |
| **tout-ou-rien, β=0,08** | **99,58 %** | 24,0 | 0,334 |

Sur la grammaire courte, le signal tout-ou-rien fait **mieux** que le signal
gradué. Sur la grammaire longue il s'effondre à 0 % quand le gradué tient 6,4 %.

Ce n'est pas la forme de la récompense qui décide, c'est le **taux de réussite au
hasard** : 0,6 % suffit, 0,001 % non. Le mot « sparse » confond deux variables
indépendantes.

## 2. Le résultat central : 99,9 % de grammaticalité sans grammaire

Sur tout le plateau, et pour **toutes** les graines :

```
P(nom accordé | déterminant imposé) = 0,333  = 2/6 déterminants
P(verbe accordé | nom imposé)       = 0,500  = 4/8 noms
```

Ce ne sont pas des scores de qualité, ce sont des **comptages** : l'agent
n'utilise que les 2 déterminants et les 4 noms d'une seule branche, et il est
parfait à l'intérieur. Forcé sur `le`, il produit un nom pluriel.

**On peut donc satisfaire intégralement un vérificateur de règles sans avoir
appris la moindre règle**, en se restreignant à un sous-espace où la contrainte
est sans objet. Aucun bug, aucune triche — un score qui ne mesure pas ce qu'on
croit. Le diagnostic est de forcer l'antécédent et de mesurer le conséquent.

Contre-exemple utile : à β=0,08, les graines 1 et 2 atteignent
P(nom|dét) = 0,911 et 0,837, P(verbe|nom) = 0,977 et 0,916. **La règle est donc
apprenable par REINFORCE — mais de façon non fiable**, et seulement dans la bande
étroite où les deux branches survivent.

## 3. Isolation causale de l'effondrement, et la conclusion que j'ai dû corriger

Ce que l'espace énumérable rend possible et qu'aucun modèle réel ne permet :
parcourir l'arbre des causes en entier.

| suspect | verdict | preuve |
|---|---|---|
| capacité du modèle | **éliminé** | ajustement supervisé → 100 %, 48,0 modes, 3 graines / 3 |
| géométrie de l'objectif | **éliminé** | paramétrisation tabulaire → 48,0 modes exactement, à tout β |
| instabilité de l'optimum | **éliminé** | parti de l'idéal, il reste à 99,9 % / 43 modes après 18 250 ép. |
| factorisation autorégressive | **coupable sous β≈0,05** | β=0,01–0,02 : GRU à gradient exact → 12–24 modes, tabulaire → 48,0 |
| procédure échantillonnée | **coupable au-dessus** | β=0,05 et 0,08 : GRU à gradient exact → **48,0 modes sur 3 graines/3** ; échantillonné → 21–45 |

**Correction d'une conclusion trop forte.** À partir des seuls runs à β=0,01 j'avais écrit « le bruit d'échantillonnage est éliminé, c'est la géométrie ». Les runs à β ≥ 0,05 l'ont démentie, et pas de justesse :

| β | graines | gradient exact |
|---|---|---|
| 0,05 | 0, 1, 2 | 48,0 modes, 50/50, validité 94,60 / 94,60 / 94,59 % |
| 0,08 | 0, 1, 2 | 48,0 modes, 50/50, validité 79,12 / 79,13 / 79,10 % |

Les optima de Gibbs calculés analytiquement pour ces β valent **94,59 %** et **79,12 %**. Le GRU à gradient exact les reproduit à deux décimales sur six runs indépendants : il n'approche pas l'optimum, il l'atteint.

Il y a donc **deux régimes séparés par une transition nette entre β=0,02 et β=0,05** :

- en dessous, la factorisation à paramètres partagés bloque l'optimum même avec un gradient parfait, et aucun estimateur ne peut aider ;
- au-dessus, ce blocage disparaît entièrement, et tout ce qui échoue encore relève de la procédure échantillonnée.

**L'échantillonnage décale d'un facteur 3 à 5 la pression entropique nécessaire** : le gradient exact ouvre les deux familles à β≈0,05, l'échantillonné à β≈0,12 — où l'optimum de l'objectif est déjà tombé à 52 % de validité.

**Confondant à signaler dans ma propre comparaison** : le run à gradient exact optimise le vrai objectif `E[R] + β·H(p)`, alors que le run échantillonné utilise le bonus d'entropie standard, régularisateur aux états visités et estimateur biaisé de ∇H. Les deux n'optimisent pas la même chose. L'écart à β=0,05 confond donc bruit et biais d'estimateur, non séparables avec ce qui a été lancé. La revendication propre reste celle à faible β, où tabulaire et GRU sont comparés sous objectif et gradient identiques.

Détail des mesures :

- **Gradient exact** (`gradient_exact.py`) : optimisation de `J = Σp(s)R(s) + βH(p)`
  par gradient analytique, zéro échantillonnage. GRU → 12,0 modes, 100 % singulier,
  identique sur 2 graines. L'effondrement n'est pas un artefact du bruit.
- **Tabulaire** (`parametrisation_et_recuit.py`) : même objectif, même gradient
  exact, un logit libre par séquence. Résultat : 48,0 modes, 100 % d'uniformité,
  50/50, et une validité de 100,00 / 99,98 / 94,60 % à β = 0,01 / 0,02 / 0,05 —
  soit **exactement l'optimum de Gibbs** calculé analytiquement (100,00 / 99,96 /
  94,59). L'objectif est innocent.
- **Sonde de capacité** (`sonde_capacite.py`) : le même GRU ajusté en supervisé
  atteint `-log P` = 3,8714 contre ln(48) = 3,8712, avec P(dét) à trois décimales
  de la répartition théorique 0,25 / 0,25 / 0,125 × 4.
- **Stabilité** (`stabilite_et_trajectoire.py`) : parti *de* la politique idéale
  (48,0 modes, 49,9/50,1), REINFORCE **quitte l'optimum dès les 250 premiers
  épisodes** — 44,0 modes et 66,7/33,3 — puis oscille autour de l'attracteur à
  45,3 modes, avec des excursions jusqu'à 26,7, et finit à 43,0 après 18 250
  épisodes en gardant les deux branches. Depuis l'aléatoire au même β : 11,5–18,6
  modes, une seule branche.

  **L'optimum n'est donc pas un point fixe.** L'énoncé correct est : *l'optimum
  est instable, mais le bassin dans lequel il retombe (45,3 modes, deux branches)
  est incomparablement meilleur que ce qui est atteignable depuis l'aléatoire.*
  L'attracteur à 45,3 est expliqué exactement en §5.

- **Localisation** (`localisation_effondrement.py`) : le token est imposé
  *pendant* la génération, donc la suite en tient compte, et la position figée
  est exclue du terme REINFORCE.

  | figée | modes | sg % | pl % | P(nom\|dét) | P(verbe\|nom) |
  |---|---|---|---|---|---|
  | aucune | 11,5 | 0,0 | 100,0 | 0,333 | 0,500 |
  | **pos0 (dét)** | **30,3** | **61,9** | **38,1** | **0,999** | **0,924** |
  | pos1 (nom) | 17,7 | 0,2 | 99,8 | 0,005 | 0,875 |
  | pos2 (verbe) | 8,0 | 100,0 | 0,0 | 0,500 | 0,009 |

  **Figer la seule marginale du déterminant suffit à faire apprendre la règle
  d'accord complète** — P(nom accordé | dét) passe de 0,333 à 0,999 pour les six
  déterminants — et les deux branches restent vivantes. **L'effondrement est
  localisé dans la marginale de la position 0.**

  Les lignes pos1 et pos2 sont attendues et non informatives : figer le nom à un
  tirage *indépendant* détruit la dépendance dét→nom par construction, d'où le
  0,005. Idem pour le verbe.

  Réserve de protocole : la validité affichée (3,19 %) est un artefact. La
  position 0 étant exclue du gradient, elle reste à son initialisation aléatoire,
  et l'évaluation la laisse libre — elle émet donc un non-déterminant 71 % du
  temps. Les conditionnelles, normalisées à l'intérieur de chaque déterminant, ne
  sont pas affectées ; ce sont les seuls chiffres à retenir de cette ligne.

**L'échec reste donc d'accessibilité plutôt que de stabilité**, au sens où la
région à deux branches se maintient une fois atteinte alors qu'elle est
inatteignable par entraînement direct. Le mécanisme : dans un GRU les
six conditionnelles `P(· | déterminant)` transitent par des paramètres partagés ;
celle qui reçoit le plus de gradient tôt façonne l'état caché, et les autres
héritent d'une représentation réglée pour elle. C'est un riche-qui-s'enrichit au
niveau de la **représentation**, pas des probabilités.

**Conséquence pratique** : le RL ne crée pas une distribution, il raffine celle
qu'on lui donne. C'est exactement l'ordre du RLHF réel — après pré-entraînement.
La raison habituellement avancée (« le pré-entraînement apporte les
connaissances ») est incomplète : il apporte aussi **la distribution que le RL est
incapable de construire lui-même**.

## 4. Le spectre de la récompense remplace le mot « sparse »

Décomposition ANOVA fonctionnelle sur le cube des 8 000 séquences
(`trajectoire_et_structure.py`) :

| récompense | ordre 1 (marginales) | ordre 2 (paires) | ordre 3 (triplet) |
|---|---|---|---|
| **graduée** | **76,1 %** | 23,9 % | 0,0 % |
| **tout-ou-rien** | **4,0 %** | 30,5 % | **65,5 %** |

À politique uniforme, le gradient de REINFORCE ne voit **que l'ordre 1** : toutes
les interactions sont moyennées. Le façonnage de récompense ne « densifie » donc
rien — il **déplace la variance des ordres élevés vers l'ordre 1**. C'est la
définition opératoire du shaping, et elle est mesurable.

Corollaire structurel : la récompense graduée a un ordre 3 **exactement nul**,
parce qu'elle est une somme de termes par position (ordre 1) et d'accords par
paire (ordre 2). L'indicateur tout-ou-rien, lui, est un produit : il met 65,5 %
de sa variance à l'ordre 3, invisible au départ.

Et le piège de curriculum est réel : la séquence qui maximise le signal d'ordre 1
est `des chat chantent` — **invalide**, R = 0,50. Le premier signal que l'agent
suit ne pointe pas vers une solution.

## 5. Un correctif qui fonctionne : le recuit de β

Le balayage montre deux régimes incompatibles. Personne ne pense à les enchaîner,
parce qu'il faut d'abord savoir que la structure en branches existe.

Recuit géométrique de β sur 30 000 épisodes, deux calendriers testés.

| méthode | validité | modes / 48 | branches |
|---|---|---|---|
| β constant 0,02 | 99,99 % | 18,6 | 1 |
| β constant 0,12 | 57,13 % | 45,9 | 2 |
| **recuit 0,2 → 0,01** | **99,97 %** | **45,3** | **2** (66,7 / 33,3) |
| **recuit 0,12 → 0,02** | **99,96 %** | **45,3** | **2** (65,8 / 34,2) |
| rejet depuis réseau non entraîné | 100 % | ~47,5 | 2 |

**Le recuit domine simultanément les deux régimes constants**, et approche
l'échantillonnage par rejet — avec une seule passe avant à l'inférence au lieu de
167. Les deux calendriers convergent vers le même point (45,3 modes, ~99,97 %),
ce qui écarte le coup de chance. Le diagnostic débouche donc sur une
intervention, pas seulement sur un constat.

Le mécanisme est cohérent avec le reste : à β élevé les deux branches reçoivent
du gradient, donc les conditionnelles des six déterminants sont toutes
entraînées ; quand β redescend, la représentation partagée est déjà formée pour
toutes, et l'interférence qui causait l'effondrement n'a plus lieu d'être. **Le
recuit ne combat pas l'effondrement, il l'empêche de se former.**

### Pourquoi le recuit plafonne à 45,3 et non 48

Les deux recuits convergent vers un partage sg/pl de 66,7/33,3 et 65,8/34,2, et
la trajectoire partie de la politique idéale y tombe aussi (66,7/33,3). Ce n'est
pas du bruit : c'est un attracteur, et il s'explique exactement.

2/3 – 1/3 = 4/6 – 2/6, soit ce qu'on obtient lorsque **P(déterminant) est uniforme
sur les 6 déterminants** — le lexique en comptant 4 singuliers et 2 pluriels.
Le nombre de modes correspondant se calcule :

```
24 phrases sg à masse (2/3)/24, 24 phrases pl à masse (1/3)/24
H = ⅔·log₂(36) + ⅓·log₂(72) = 5,5032 bits
2^H = 45,35 modes effectifs
```

**45,3 — exactement la valeur mesurée.** Le plafond résiduel est donc entièrement
imputable à un décalage de cible : **le bonus d'entropie par token vise
l'uniformité sur les tokens, pas sur les séquences.** Les deux ne coïncident que
si tous les préfixes ont le même nombre de complétions valides, ce qui est faux
ici : `les` en admet 12, `le` seulement 6. L'optimum exige P(`les`) = 0,25 et
P(`le`) = 0,125 ; le bonus pousse vers 1/6 partout.

Correctif prédit, non testé : pondérer l'entropie par le nombre de complétions
valides, ou régulariser l'entropie de séquence plutôt que celle des tokens.

## 6. Comparaison qu'il faut faire et que personne ne fait

L'échantillonnage par rejet depuis le réseau **non entraîné** : 0,60 % de masse
valide et 47,5 modes effectifs à l'initialisation, donc tirer-et-filtrer donne
100 % de validité et ~47,5 modes par construction, pour ~167 tirages par sortie.

Il bat, sur les deux axes à la fois, tous les entraînements REINFORCE du plateau.

Deux précisions qui empêchent d'en tirer trop :
- la métrique de diversité favorise le rejet **par construction** (uniforme sur
  les 48). Ce qui sauve le résultat, c'est que cette même distribution est aussi
  l'optimum de REINFORCE — donc la comparaison mesure un échec d'optimisation,
  pas une supériorité générale du rejet ;
- le rejet exige le vérificateur **à l'inférence**, ce dont on ne dispose pas
  pour du langage réel.

Énoncé qui survit : *quand la cible est déjà atteignable par tirage aléatoire,
entraîner par gradient de politique ne fait pas mieux qu'un filtre et détruit la
diversité que le filtre préserve gratuitement.*

## 6bis. L'optimum est une étape, pas une destination

Même mesure que ci-dessus mais depuis l'aléatoire, à β = 0,02, avec relevé de la
distribution exacte tous les 250 épisodes :

```
maximum de modes effectifs : 24,0 à l'épisode 4 750
KL minimale vers l'idéal   : 1,0000 bit à l'épisode 11 500
état final                 : 11,5 modes, KL 2,0611 bits
```

**La diversité culmine à mi-parcours puis se dégrade.** Un arrêt précoce battrait
la convergence de **+12,5 modes**. Entraîner au-delà de l'épisode ~4 750 détruit
la moitié de ce qui avait été acquis.

**À ne pas citer comme acquis** : une seule graine, et j'ai observé un écart
run-à-run à réglages nominalement identiques (11,5 modes ici contre 18,6 au
balayage), très probablement du non-déterminisme multithread de torch sur CPU. Le
phénomène est cohérent avec le portrait de phase (§6ter) et avec l'écart
initialisation/arrivée, mais il n'est pas répliqué.

## 6ter. Deux résultats nuls, rapportés comme tels

Deux questions que je jugeais prometteuses n'ont rien donné. Les omettre
donnerait une fausse impression de taux de réussite.

**Dimension effective de la trajectoire — hypothèse réfutée.** Je pariais que la
trajectoire d'entraînement, bien que vivant dans un simplexe à 8 000 dimensions,
tiendrait dans 2 ou 3 composantes, et qu'on pourrait donc *dessiner* la dynamique.
ACP sur 81 relevés de distribution exacte × 3 graines :

```
CP1 36,2 %  CP2 20,3 %  CP3 13,5 %  CP4 7,2 %  CP5 4,8 %  CP6 4,3 %
90 % du mouvement → 8 dimensions
99 % → 21 dimensions
99,9 % → 33 dimensions
```

Le plan CP1-CP2 ne capture que 56,5 % du mouvement. Le portrait de phase
(`resultats_test2/portrait_de_phase.png`) reste lisible — trois graines finissant
en trois points bien séparés, passant près de la région de l'optimum avant d'en
diverger, ce qui est cohérent avec le pic de diversité à mi-parcours — mais il ne
peut pas servir de preuve à lui seul.

**Fonctionnelles conservées — aucune.** Zéro fonctionnelle varie de moins de 0,02
sur tout l'entraînement. Les neuf masses catégorie × position bougent toutes
massivement : P(dét en position 0) de 0,289 à 0,9998, P(nom en position 1) de
0,406 à 1,000, P(verbe en position 2) de 0,302 à 1,000, et toutes les catégories
hors-structure tombent à 0. Chercher un invariant n'apporte rien ici.

Sous-produit utile : à l'initialisation, P(dét en position 0) = 0,289 ≈ 6/20 et
P(nom en position 1) = 0,406 ≈ 8/20. Le réseau aléatoire est donc bien uniforme
sur les tokens — ce qui confirme indépendamment la base du calcul de
l'échantillonnage par rejet (§6).

## 7. Limites, et défauts de ma propre conception

- **La grammaire longue change deux variables à la fois** : les adverbes
  agrandissent l'espace, les adjectifs ajoutent *aussi* une règle d'accord. Pour
  le contrôle tout-ou-rien c'est sans gravité (seul le taux de réussite compte),
  mais l'interprétation du run gradué est confondue. Version propre :
  `dét nom verbe adv adv`.
- **Le premier balayage était à graine unique**, et le coefficient retenu (0,08)
  a été choisi dessus. Or à ce β la graine 0 est justement celle qui reste
  mono-branche. Tous les tests en aval ont donc tourné sur un régime non
  représentatif. Le balayage multi-graines a été refait ; les tests en aval, non.
- **L'argument de dominance ne vaut que sur le plateau** (β ≤ 0,08). Au-delà, la
  politique apprise disperse assez de masse sur l'invalide pour être *plus*
  entropique que l'uniforme sur les 48.
- **À β = 0 l'effondrement est l'optimum**, pas un échec : sans terme d'entropie
  l'objectif ne demande aucune diversité.
- **Le bonus d'entropie implémenté** est le régularisateur standard aux états
  visités, estimateur biaisé de ∇H(trajectoire). Les conclusions reposant sur
  l'optimum de Gibbs sont donc approchées ; l'argument de dominance, lui, n'en
  dépend pas.
- **Budget de 20 000 épisodes** : j'ai établi que REINFORCE est lent, pas qu'il
  est incapable.
- **Une première version du test de localisation (Q-C) était buguée** — le token
  imposé l'était après génération, donc la suite restait conditionnée sur un autre
  token. Corrigée dans `localisation_effondrement.py`.

## 8. Ce que ce test apporte à la question du projet

Le test 1 avait montré qu'un succès en RL pur pouvait venir entièrement de la
décomposabilité de la récompense. Le test 2 ajoute trois choses :

1. **Un score parfait ne prouve pas qu'une règle est apprise.** L'agent trouve
   une sous-langue où la contrainte est vide. Le diagnostic — forcer
   l'antécédent, mesurer le conséquent — vaut pour toute récompense à base de
   règles.
2. **Le RL raffine, il ne construit pas.** Capable de maintenir la bonne
   distribution, incapable de l'atteindre. Ce n'est pas une limite
   d'optimisation contournable mais une propriété de la paramétrisation
   autorégressive entraînée par gradient de politique.
3. **Le vérificateur reste écrit à la main.** On a remplacé un oracle-point
   (test 1) par un oracle-ensemble (test 2), sans supprimer l'oracle. Pour du
   langage réel, personne ne sait écrire ce parser — et c'est le seul point qui
   décide du passage à l'échelle.

## Fichiers

| fichier | contenu |
|---|---|
| [grammaire.py](grammaire.py) | lexique, parser, comptages exacts |
| [rl_grammaire.py](rl_grammaire.py) | politique, entraînement, balayage, mesures exactes |
| [balayage_graines.py](balayage_graines.py) | balayage multi-graines |
| [sonde_capacite.py](sonde_capacite.py) | sonde de représentabilité |
| [optimum_gibbs.py](optimum_gibbs.py) | optimum en forme close, taxe de mise en forme |
| [verifier_dominance.py](verifier_dominance.py) | vérification de l'argument de dominance |
| [gradient_exact.py](gradient_exact.py) | signal d'ordre 1, gradient exact |
| [parametrisation_et_recuit.py](parametrisation_et_recuit.py) | tabulaire vs GRU, recuit de β |
| [stabilite_et_trajectoire.py](stabilite_et_trajectoire.py) | stabilité, trajectoire KL |
| [trajectoire_et_structure.py](trajectoire_et_structure.py) | ACP, ANOVA, fonctionnelles conservées |
| [localisation_effondrement.py](localisation_effondrement.py) | gel de position, version corrigée |
| `resultats_test2/` | logs, CSV, JSON, courbes, portrait de phase |

Le raisonnement, les hypothèses réfutées et les questions ouvertes sont dans
[CARNET.md](CARNET.md).
