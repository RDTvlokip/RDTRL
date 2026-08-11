# Carnet de recherche — RDTRL

Ce fichier n'est ni le changelog (ce qui a été construit) ni l'analyse (ce que
les chiffres disent). C'est le **raisonnement** : les hypothèses formulées, celles
que les données ont démolies, les résultats obtenus sans expérience, les défauts
trouvés dans mes propres protocoles, et les pistes non tranchées.

Les erreurs y sont conservées avec leur date de mort. Une hypothèse réfutée et
datée vaut mieux qu'une conclusion propre sans historique — c'est elle qui dit
quel type de raisonnement m'a trompé.

---

## 1. Hypothèses que j'ai formulées et que les données ont réfutées

### 1.1 « Le bassin pluriel est plus large » — mort le 29/07/2026

**Ce que j'ai dit.** En voyant l'agent se verrouiller sur la branche pluriel à
β=0,05, j'ai produit aussitôt une cause : `les` et `des` sont neutres en genre,
donc compatibles avec les 4 noms pluriels, alors que `le` n'admet que 2 noms
masculins singuliers. Bassin deux fois plus large → l'exploration précoce y tombe.

**Ce qui l'a tuée.** Trois graines supplémentaires au même β : 3 sur 4 partent sur
le **singulier**. L'explication prédisait une préférence systématique pour le
pluriel. Faux.

**La vraie cause.** Loterie d'initialisation. L'agent verrouille *une* branche,
laquelle est décidé tôt et arbitrairement.

**La leçon, qui est la vraie valeur de l'entrée.** L'explication était compatible
avec toutes les données dont je disposais, mécaniquement plausible, et produite
en une seconde. C'est cette vitesse qui aurait dû m'alerter : elle venait d'un
récit, pas d'un test. Une cause qui « colle » colle presque toujours.

### 1.2 « L'ensemble des solutions est disconnexe dans l'espace des politiques » — mort le 29/07/2026

**Ce que j'ai dit.** Pour déplacer de la masse de `les …` vers `le …`, un bonus
d'entropie par token doit forcément en mettre sur les mélanges invalides du type
`le chats dorment`. Donc couvrir les deux branches coûte nécessairement de la
grammaticalité.

**Ce qui l'a tuée.** La sonde de capacité : le même GRU atteint 100 % de masse
valide avec 48 modes et un partage 49,9/50,1, en 500 pas d'ajustement supervisé,
sur 3 graines sur 3.

**Pourquoi c'était faux.** J'ai raisonné sur la *marginale* de la position 0 en
oubliant que la politique est **conditionnelle**. L'état caché transporte le
déterminant émis ; il n'a besoin d'encoder que 6 valeurs dans 128 dimensions.
Une politique peut donc mettre 25 % sur `les` et 12,5 % sur `le` et rester
parfaitement valide dans les deux cas. L'ensemble des solutions est connexe et
atteignable.

### 1.3 « Le test du nom jamais vu donnera du hasard (0,5) » — mort le 29/07/2026

**Ce que j'ai dit.** L'embedding de `fleurs` n'est jamais entraîné, donc l'agent
n'a aucune information sur ce token, donc P(verbe pluriel) ≈ 0,5.

**Ce qui l'a tuée.** Mesuré : 0,9966.

**Ce qui est quand même vrai.** Ma conclusion pratique (« ce test ne mesure
rien ») tenait, mais pour une raison que je n'avais pas identifiée. Le 0,9966
n'est pas une généralisation : l'agent émet un verbe pluriel **quel que soit le
nom** — les noms singuliers vus donnent 0,0003 à 0,0149. La moyenne sur les noms
vus est 0,4286, *sous* le hasard. Le test est confondu par l'effondrement sur la
branche pluriel, pas par l'embedding non entraîné.

**La leçon.** J'avais raison sur le verdict et tort sur le mécanisme. Si je
n'avais mesuré que le verdict, j'aurais gardé une explication fausse en la
croyant confirmée.

### 1.4 « Le compromis validité / diversité est une propriété de la tâche » — mort le 29/07/2026

C'est l'erreur la plus grave, parce que j'allais l'écrire dans un verdict.
Réfutée deux fois, indépendamment : par le calcul de l'optimum de Gibbs (§2.1)
et par l'argument de dominance (§2.2). Détail en §2.

### 1.5 Correction rétroactive au test 1 — 29/07/2026

J'avais écrit : « le blocage est sur l'obtention du signal, jamais sur
l'optimisation ». **Faux dès qu'il existe plusieurs solutions.** C'était vrai
pour une cible unique, où il n'y a rien à répartir. Le test 2 montre une
optimisation qui échoue alors que le signal est parfait.

### 1.6 « La branche est biaisée environ 2 contre 1 vers le singulier » — morte le 31/07/2026

Publiée dans une réponse à une critique extérieure, à partir de 15 runs
singuliers sur 24. Les 24 étaient 3 graines × 8 valeurs de β, et dans le régime
d'effondrement la branche est décidée par la graine seule. **70 graines à
condition unique : 37 / 33, Wilson [0,413 ; 0,641], p = 0,72 contre une pièce
équilibrée et p = 0,016 contre mon 2 contre 1.** Le biais d'ordre 1 au nom
(+0,0167) existe et se calcule, mais il ne survit pas à la dynamique
échantillonnée. Détail en §7.11.

### 1.7 « L'estimateur d'entropie biaisé explique l'écart exact / échantillonné » — morte le 31/07/2026

Produite en quelques secondes, séduisante, et fausse. Le bonus implémenté ne
rétropropage pas à travers la distribution de visite (§5.1), donc il ne peut pas
récompenser l'ouverture d'une branche jamais visitée — le récit tenait debout.
Mais la table de saturation dit **H(nom | `la`) = 0,997 bit contre 1,000 pour
`le`** : les branches mortes n'ont pas une entropie de continuation plus élevée,
et le terme manquant ne pousserait donc pas vers elles. Réfutée **avant** d'avoir
été utilisée, pour une fois.

### 1.8 « REINFORCE résout exactement le problème restreint aux produits » — morte le 31/07/2026

Formulée en voyant 19 des 37 runs singuliers exactement à 12,0 : un plafond
atteint si précisément devait être un optimum, pas une contrainte subie.
`optimum_produit.py` optimise le même objectif sur trois lois indépendantes et
trouve **24,00 modes**, dans le coin pluriel, à tous les β et sur trois graines.
REINFORCE se pose sur 12 une fois sur deux : il est à un optimum **local** de la
classe restreinte. Ce qui survit : conditionnellement au coin, il atteint le
produit maximal de ce coin environ une fois sur deux. Détail en §7.11ter.

### 1.9 « Le seuil de 0,35 du test 3 est dérivé, donc solide » — morte le 11/08/2026

Écrite dans TEST3.md §6.1 : le seuil venait du maximum observé sur 20 000 tirages
de la loi nulle, donc il n'était pas arbitraire. La dérivation était correcte, la
ligne dérivée ne l'était pas. Un maximum d'échantillon n'estime rien ici : les
1 296 codes compositionnels **sont** des bijections, ils appartiennent à la loi
nulle avec probabilité 1,19 × 10⁻²⁵ et valent 1, donc le supremum de la nulle vaut
exactement 1 — la valeur qu'on voulait déclarer hors d'atteinte. Douze blocs
indépendants de 10 000 000 donnent des maxima de 0,3775 à 0,4283, étendue 1,54
écart-type de la loi elle-même, quand le quantile 99,9 % varie de 0,0006. Détail
en §7.14.

### 1.10 « L'inflation du double compte croît avec la concentration » — morte le 11/08/2026

Tirée d'un balayage par tranches de concentration sur la loi nulle : l'inflation y
passait de 0,0014 sous 0,05 à 0,0228 au-dessus de 0,30, donc elle semblait suivre
le niveau. L'échelle par transpositions dit l'inverse **au même niveau** : à
concentration 0,27, un code issu de la nulle est inflaté de 0,021, un code à 14
transpositions d'un compositionnel l'est de 0,0022. Les deux mesures sont justes.
L'inflation suit la **structure**, pas le niveau, et les deux ne sont pas le même
axe. Conséquence utile : elle est exactement nulle partout où §6.1 a quelque chose
à lire. Je n'aurais pas trouvé ça en balayant une seule des deux populations.

---

## 2. Résultats obtenus par raisonnement seul, sans expérience

### 2.1 L'objectif optimisé a un optimum connu en forme close

La perte contient `− β · Σ_t H(a_t | a_<t)`. Or la somme des entropies
conditionnelles par pas **est** l'entropie de trajectoire :
`Σ_t H(A_t | A_<t) = H(S)`. L'objectif est donc `E[R] + β·H(S)`, dont l'optimum
unique est la loi de Gibbs `π*(s) ∝ exp(R(s)/β)`.

**Conséquence non triviale : les 48 phrases valides ont R = 1 exactement, donc
π\* leur assigne à toutes la même probabilité, à n'importe quel β.** L'optimum a
toujours 48 modes effectifs, 100 % d'uniformité, et un partage 50/50 entre
familles. Vérifié numériquement sur les 8 000 séquences pour 8 valeurs de β.

### 2.2 Un ensemble d'optima à égalité est un certificat gratuit de sous-optimalité

C'est l'idée que je trouve la plus réutilisable hors de ce projet.

Normalement on ne peut pas mesurer l'écart à l'optimum sans connaître l'optimum.
Mais dès que plusieurs solutions ont **exactement** la même récompense, le
maximum d'entropie impose qu'elles soient équiprobables. Donc **tout écart à
l'équiprobabilité prouve que l'optimisation a échoué**, sans jamais calculer la
valeur de l'optimum. L'effondrement de mode cesse d'être un constat qualitatif
et devient une mesure exacte avec une cible connue.

**Version indépendante de toute définition d'entropie — mais de portée limitée,
ce que j'avais d'abord affirmé à tort.** L'uniforme sur les 48 a `E[R] = 1` et
`H = ln 48 = 3,8712` nats. Comparaison de `J = E[R] + β·H` sur les politiques
sauvegardées (`verifier_dominance.py`) :

| β | E[R] | H nats | J apprise | J uniforme48 | verdict |
|---|---|---|---|---|---|
| 0,0 | 1,0000 | 0,0000 | 1,0000 | 1,0000 | **ex aequo** |
| 0,01 | 0,9993 | 2,2983 | 1,0223 | 1,0387 | sous-optimale (prouvé) |
| 0,02 | 1,0000 | 2,9255 | 1,0585 | 1,0774 | sous-optimale (prouvé) |
| 0,05 | 0,9830 | 3,4430 | 1,1552 | 1,1936 | sous-optimale (prouvé) |
| 0,08 | 0,9679 | 3,5369 | 1,2509 | 1,3097 | sous-optimale (prouvé) |
| 0,12 | 0,8527 | 5,5942 | 1,5240 | 1,4645 | **ne tient pas** |
| 0,2 | 0,5835 | 7,4569 | 2,0749 | 1,7742 | **ne tient pas** |

**Erreur corrigée n°1.** J'avais écrit « domine strictement » sans calculer le
terme d'entropie. La politique apprise place une part de sa masse sur l'invalide,
éparpillée sur 7 952 séquences, ce qui **contribue à son entropie**. À β ≥ 0,12
elle est plus entropique que l'uniforme sur 48 : l'argument ne prouve plus rien.
**La conclusion doit être restreinte au plateau β ∈ [0,01 ; 0,08].**

**Erreur corrigée n°2, plus conceptuelle.** À β = 0 la marge est exactement
nulle. Sans terme d'entropie, l'objectif est `E[R]` seul, et une politique
concentrée sur une unique phrase valide atteint `E[R] = 1`. **L'effondrement de
mode à β = 0 n'est pas un échec : c'est l'optimum.** Reprocher à l'agent de ne
pas être divers quand la diversité n'est pas dans l'objectif revient à reprocher
à l'objectif de ne pas contenir ce qu'on voulait.

**Le tableau honnête a donc trois régimes, pas deux :**
- β = 0 → l'effondrement est optimal ;
- β ∈ [0,01 ; 0,08] → l'effondrement est prouvablement sous-optimal, vrai échec
  d'optimisation ;
- β ≥ 0,12 → l'agent est proche de son optimum, et c'est **l'optimum** qui est
  mauvais (taxe de mise en forme, §2.3).

**Nuance sur la sonde de capacité.** J'ai écrit « représentable, et triviale à
trouver ». Le « triviale à trouver » est une surinterprétation : l'ajustement
supervisé voit les 48 cibles à chaque pas, en gradient plein. Il établit la
**représentabilité**, pas une difficulté d'optimisation comparable à information
égale. Ces deux choses ne doivent pas être confondues.

### 2.3 Décomposer un échec en taxe de mise en forme et écart d'optimisation

Si l'optimum est calculable, on peut séparer deux causes systématiquement
confondues :

- **taxe de mise en forme** = 100 % − validité de π\* lui-même. La récompense
  graduée paye 0,8333 les 72 quasi-ratons, donc son optimum contient des phrases
  invalides *par construction*, en proportion exp(−0,167/β). À β=0,08 elle
  plafonne à **79,12 %** contre **99,94 %** pour le tout-ou-rien.
- **écart d'optimisation** = ce qui sépare la politique apprise de π\*.

L'usage courant est de constater une mauvaise validité et de blâmer
l'optimiseur, sans vérifier que la cible visée était déjà mauvaise. Ici les deux
causes vont même en sens opposé (§2.4).

### 2.4 Le renversement : l'effondrement de mode est conservateur

À β ≥ 0,08, la politique apprise est **plus grammaticale que l'optimum de son
propre objectif** (94,87 % contre 79,12 %). Elle achète cette validité en
sacrifiant l'entropie. Conséquence perverse : **si REINFORCE réussissait à
optimiser à β=0,08, la validité tomberait de 95 % à 79 %.** L'échec
d'optimisation masquait la taxe de mise en forme.

Et la « falaise » à β=0,12 n'est pas l'apparition d'un compromis : l'écart
optimum/atteint y tombe de 24 modes à 2. **La falaise, c'est le moment où
l'optimiseur commence enfin à réussir**, et où il révèle que l'optimum visé à ce
β est mauvais.

### 2.5 Le piège de conservatisme des politiques autorégressives

Mécanisme proposé pour expliquer le verrouillage de branche : l'avantage d'un
préfixe est évalué **sous la politique de suffixe courante**. Si `le` n'est
presque jamais émis, les continuations après `le` ne sont pas entraînées, donc
émettre `le` rapporte peu, donc REINFORCE fait redescendre P(`le`). Pessimisme
auto-réalisateur.

Le biais n'est pas de moyenne nulle : il est **directionnel**, toujours contre
les préfixes peu explorés. REINFORCE sur générateur autorégressif a donc un
conservatisme intégré qui croît avec la longueur du suffixe à réapprendre.

**Prédiction falsifiable qui en découle** : l'entropie nécessaire pour rouvrir
une branche doit croître avec la longueur du suffixe. Suffixe de 2 tokens
(grammaire courte) → plateau large. Suffixe de 4 tokens (grammaire longue) →
plateau détruit. **Partiellement confirmé** : grammaire longue à β=0,08, validité
6,4 %. Confondu toutefois par §4.1, donc pas concluant.

### 2.6 Sous-langues dégénérées : haute grammaticalité sans grammaire

Sur le plateau (β ≤ 0,05), l'agent atteint 92–100 % de validité avec
P(nom accordé | dét) = 0,333 et P(verbe accordé | nom) = 0,500 — soit exactement
2 déterminants sur 6 et 4 noms sur 8. Il n'a appris **aucune règle d'accord** :
il s'est restreint à une sous-langue entièrement au pluriel, où l'accord est
automatiquement satisfait.

**Généralisation hors de ce projet.** Toute récompense de type satisfaction de
contraintes admet des **sous-langues dégénérées** : des sous-ensembles de
l'espace de sortie où la contrainte est vacuellement vraie, donc ne porte aucun
signal d'apprentissage. Un score élevé sur une récompense à base de règles ne
prouve pas que la règle a été apprise. Diagnostic : **forcer l'antécédent,
mesurer le conséquent** — ici P(verbe | nom imposé), noms singuliers inclus.
Point probablement sous-estimé dans l'évaluation des systèmes entraînés sur
récompense à base de règles.

---

## 3. Pièges de mesure trouvés en route

### 3.1 Le nombre de modes seul ne mesure rien

L'initialisation **aléatoire** a **47,5 modes effectifs sur 48**. Évidemment :
une politique quasi uniforme est quasi uniforme aussi sur les 48 valides.

Pire : REINFORCE **dégrade** la diversité sous son point de départ (47,5 → 9,9 à
β=0,01) tout en améliorant la validité. Toute métrique de diversité rapportée
sans la masse valide en regard est ininterprétable. J'ai failli publier une
frontière construite exactement là-dessus.

### 3.2 Un chiffre identique partout ne mesure pas l'agent

« 1re phrase valide à l'épisode 45 » apparaissait à l'identique dans tous les
runs de grammaire courte, tous β confondus, tout-ou-rien inclus. Lecture
tentante : « l'exploration initiale est efficace ». Faux : ces runs partagent la
graine 0 et les 45 premiers épisodes ne modifient presque pas les poids. **Le
nombre mesure l'initialisation partagée, pas l'agent.**

Note annexe : à 0,6 % de validité au hasard, la première réussite devrait tomber
vers l'épisode 167, pas 45. Un réseau à poids aléatoires n'est donc pas uniforme
sur les 8 000 séquences — il a déjà des préférences marquées. Non creusé.

### 3.3 Ma propre métrique de saturation est mal étiquetée — CORRIGÉ le 31/07/2026

Dans le tableau H(nom | déterminant), la colonne `satur.%` dépasse 100 %
(`la` : 218 %). Cause : H est calculée sur les 8 noms alors que H_max utilise le
nombre de noms *compatibles*. Une valeur > 100 % signale donc une **fuite de
masse sur des noms incompatibles**, c'est-à-dire un échec — pas une
sur-saturation.

**Signalé ici, puis laissé tel quel jusqu'à ce qu'un lecteur extérieur demande
précisément ce champ** (§7.11). Un défaut connu, écrit au carnet et non corrigé
est pire qu'un défaut inconnu : je l'aurais servi en croyant l'avoir traité,
parce qu'il était noté.

Correction : le ratio confondait deux questions, elles sont maintenant séparées
dans `analyse_exacte`.

| champ | question à laquelle il répond |
|---|---|
| `masse_accordee_pct` | l'agent reste-t-il **valide** après ce déterminant ? |
| `saturation_pct` | parmi les noms **compatibles**, combien en utilise-t-il vraiment ? |

`saturation_pct` se calcule désormais sur la conditionnelle **restreinte aux noms
compatibles puis renormalisée**, donc bornée à 100 % par construction. `H_bits`
reste l'entropie sur les 8 noms, qui est la vraie conditionnelle et n'était pas
fausse ; c'est le **rapport** qui l'était.

### 3.4 Puissance statistique du test de généralisation

P(`fleurs` | `des`) = 0,2248 après exclusion, contre 0,2560 en moyenne pour les
autres pluriels — ratio 0,878, lu comme « généralisation compositionnelle ». Mais
la dispersion entre noms **non exclus** va de 0,179 (`chiens`) à 0,309
(`tables`), soit un ratio de 0,70 à 1,21. **L'effet mesuré est plus petit que la
variabilité naturelle entre noms.** Conclusion à ne pas tenir sans plusieurs
graines.

---

## 4. Défauts de conception dans mes propres protocoles

### 4.1 La grammaire longue change deux variables à la fois

Construite pour isoler la **taille de l'espace**, elle ajoute des adverbes
(espace ↑, contraintes =) *et* des adjectifs (espace ↑, contraintes ↑, une règle
d'accord de plus). Pour le contrôle tout-ou-rien le dégât est limité — sparse
échoue si et seulement si le taux de réussite au hasard est trop bas, et ce taux
est mesuré. Mais pour interpréter la difficulté du run **gradué**, le confondant
est réel. Version propre : `dét nom verbe adv adv`.

### 4.2 Le balayage à graine unique ne permet pas de tracer une frontière

À β=0,08, la graine 0 reste sur une branche (24,4 modes, 94,9 % valide) alors que
les graines 1, 2 et 3 couvrent les deux (43–46 modes, 78–90 % valide). **Les
graines 1–3 à β=0,08 dominent, sur les deux axes, le point β=0,12 de la graine
0** (45,9 modes, 57,1 %). Une frontière tracée sur une graine est donc non
seulement bruitée mais potentiellement fausse en forme.

### 4.3 Le critère de sélection du β a hérité du défaut

Le coefficient retenu (0,08) a été choisi sur la seule graine 0, qui se trouve
être celle qui reste mono-branche à cette valeur. Tous les tests en aval
(tout-ou-rien, grammaire longue, exclusion de paire, token exclu) ont donc tourné
sur un régime non représentatif.

---

## 5. Questions ouvertes, non tranchées

### 5.1 Le bonus d'entropie implémenté est un estimateur biaisé

`entropies.sum(1).mean()` régularise l'entropie **aux états visités**. Le vrai
`∇H(trajectoire)` contient en plus un terme dû au changement de la distribution
des préfixes. La revendication « l'optimum est Gibbs » porte donc sur l'objectif
*idéalisé*. L'argument de dominance (§2.2) n'en dépend pas et suffit à la
conclusion, mais une implémentation non biaisée reste non testée.

### 5.1bis « Échoue » ou « est lent » ? Non tranché

Tous les runs font 20 000 épisodes. Je conclus que REINFORCE **n'atteint pas**
l'optimum sur le plateau. Mais je n'ai aucune donnée sur des budgets plus longs :
peut-être que les modes montent lentement de 18,6 vers 48 en 10⁶ épisodes. « Ne
peut pas » et « est lent » sont deux affirmations différentes et je n'ai établi
que la seconde.

Test décisif et bon marché : un run à β = 0,02 sur 200 000 épisodes, en traçant
les modes effectifs en fonction du temps. Si la courbe plafonne, c'est un point
fixe. Si elle monte encore, c'est une question de budget et toute la formulation
change. **Non fait.**

### 5.1ter L'explication du « premier valide à l'épisode 45 » n'est pas vérifiée

En §3.2 j'affirme que le chiffre identique partout vient de la graine partagée et
du fait que les 45 premiers épisodes ne changent presque rien. Plausible, non
testé — et il existe une alternative plus fine pour le tout-ou-rien : tant
qu'aucune récompense n'est obtenue, l'avantage vaut `0 − baseline = 0`, donc le
terme REINFORCE est **exactement nul** et seul le bonus d'entropie fait bouger
les poids. Deux mécanismes différents produisent le même chiffre. Non départagés.

### 5.2 « RL pur » n'est pas « REINFORCE »

Angle mort majeur : je teste un algorithme de 1992 et j'en tire des conclusions
sur « le RL ». L'effondrement de mode est une pathologie **spécifique** aux
méthodes on-policy qui maximisent l'espérance de récompense. Un objectif qui
échantillonne *proportionnellement* à la récompense (les GFlowNets sont conçus
exactement pour ça) n'a aucune raison de verrouiller une branche, et mes 48
solutions à égalité sont leur cas d'usage canonique. **Non testé.**

### 5.3 Test 2 contient la première récompense non décomposable du projet

Le score de structure se décompose par position, comme au test 1. Mais **l'accord
est intrinsèquement une contrainte de paire** : aucune décomposition par position
ne peut le capturer. C'est la première fois qu'on a une récompense partiellement
irréductible, et je ne l'ai pas dit explicitement.

Mesurable exactement sur les 8 000 séquences : quelle part de la variance de la
récompense est expliquée par les marginales par position, quelle part par les
interactions (décomposition de type Sobol / ANOVA fonctionnelle).

**FAIT** — voir §7.4. Graduée : 76,1 % d'ordre 1, 23,9 % d'ordre 2, **0,0 %**
d'ordre 3. Tout-ou-rien : 4,0 % d'ordre 1, 30,5 % d'ordre 2, **65,5 %** d'ordre 3.
Le gradient à politique uniforme ne voit que l'ordre 1, donc le façonnage ne
« densifie » rien : il déplace la variance des ordres élevés vers l'ordre 1.

### 5.4 « Sans oracle » fait un travail rhétorique non mérité

Le parser ne connaît pas la phrase à l'avance, mais c'est une spécification
écrite à la main de ce qui compte comme correct. On n'a pas supprimé l'oracle :
on a remplacé un **oracle-point** (test 1) par un **oracle-ensemble** (test 2).
Le problème de passage à l'échelle est inchangé — pour du langage réel, personne
ne peut écrire ce parser.

### 5.5 La question que je n'ai jamais posée en retour

**Quel résultat ferait abandonner l'hypothèse ?**

Depuis le début il y a un motif : on construit un environnement où un humain
encode la réponse, puis on observe que le RL la trouve. Chaque test « réussit »
pour une raison qu'on a nous-mêmes fournie. Un test qui pourrait réellement
soutenir l'hypothèse devrait avoir une récompense qui ne provient d'aucun
ensemble-cible spécifié par un humain — sinon on mesure la spécification, pas
l'agent.

---

## 5bis. Questions que je crois sous-explorées, au-delà de ce projet

Avertissement épistémique : je ne peux pas vérifier qu'une question n'a jamais
été posée. J'indique pour chacune ce qui est établi, ce qui me paraît inhabituel,
et ce qui est testable ici et maintenant.

### 5bis.1 Acheter de la diagnosticabilité avec de la résolution de récompense

Le certificat de §2.2 exige des récompenses **exactement** à égalité. Or dans un
RLHF réel, le modèle de récompense est continu : les égalités sont de mesure
nulle, donc le certificat est inutilisable.

**Question : et si on quantifiait délibérément le modèle de récompense pour
fabriquer des égalités ?** Arrondir à *k* niveaux crée des classes
d'équivalence dont les membres doivent être équiprobables à l'optimum. On perd
un peu de résolution et on gagne un certificat d'optimalité exact, gratuit, à
chaque pas d'entraînement.

Ce qui me paraît inhabituel : personne ne traite la **diagnosticabilité comme
une quantité achetable** avec de la précision de récompense. C'est un arbitrage
de conception qui n'est jamais posé parce que les égalités sont vues comme un
défaut à éviter, pas comme une ressource à fabriquer.

### 5bis.2 Un détecteur d'effondrement de mode qui prouve au lieu d'estimer

Mon certificat utilise la distribution exacte, donc l'énumération — impossible
sur un vrai modèle de langue. **Mais l'énumération n'est pas nécessaire.** Il
suffit de *k* passes avant en teacher forcing : prendre *k* sorties que le modèle
de récompense note identiquement, calculer leur log-probabilité sous la
politique, et tester l'uniformité. Coût O(k), aucune énumération, applicable à
n'importe quel modèle.

On passe d'une heuristique (« l'entropie a baissé, c'est peut-être un
effondrement ») à une **réfutation** : les probabilités relatives d'un ensemble à
récompense égale *doivent* être uniformes à l'optimum, tout écart est une preuve.
Combiné à 5bis.1, ça donne un protocole complet. Testable immédiatement.

### 5bis.3 Une loi d'échelle pour l'effondrement de mode, dérivée et non ajustée

De §2.5 : un préfixe dont le suffixe est peu entraîné est systématiquement
sous-évalué, et le biais est directionnel. Donc il existe un coefficient
d'entropie critique β_c(L) en dessous duquel une branche ne peut pas rester
vivante, croissant avec la longueur L du suffixe à réapprendre.

**Question : β_c(L) est-il polynomial ou exponentiel en L ?** Si exponentiel, il
existe une longueur critique au-delà de laquelle *aucun* β praticable ne
fonctionne, et le max-ent RL sur séquences longues est structurellement
condamné sans correction hors-politique. Ce serait une loi d'échelle de
l'effondrement de mode **dérivée d'un mécanisme**, pas ajustée sur des courbes.

Mes deux grammaires (suffixe 2 tokens contre 4) sont une mesure à deux points de
cette courbe. Une famille de grammaires à longueur variable la donnerait
proprement, pour un coût dérisoire.

### 5bis.4 L'estimateur de gradient est décomposé même quand la récompense ne l'est pas

Point que je n'avais pas formulé. Ma politique est autorégressive, donc elle
*peut* représenter n'importe quelle loi jointe. Mais REINFORCE donne à chaque
`log π(a_t)` **le même avantage global**. Le gradient est donc décomposé par
position, y compris pour une contrainte d'accord qui est intrinsèquement binaire.

**Question : la variance de l'estimateur croît-elle avec l'arité de la
contrainte ?** Intuition : pour qu'une contrainte à k positions soit apprise, il
faut que la corrélation entre k tirages simultanément corrects et la récompense
émerge du bruit — ce qui suggère un besoin d'échantillons croissant avec k.

Ça prédirait exactement ce que j'observe : l'accord (arité 2) est appris tard et
mal, et la grammaire longue, qui ajoute une seconde contrainte binaire, échoue.
Le tableau 2×2 récompense décomposable / non-décomposable × politique factorisée
/ non factorisée n'est, à ma connaissance, jamais posé explicitement — et la
case intéressante est celle où le test 2 vit.

### 5bis.5 Une entropie masquée plutôt qu'une entropie aveugle

Le bonus d'entropie par token pousse chaque conditionnelle vers l'uniforme **sur
tout le vocabulaire**, pas sur les continuations valides. C'est exactement ce qui
produit la falaise : pour garder `le` vivant il faut aussi mettre de la masse sur
des tokens franchement invalides.

**Question : et si l'entropie n'était maximisée que sur le support des actions
ayant déjà reçu un avantage positif ?** Diversité à l'intérieur du bon ensemble,
sans fuite vers l'invalide. Quelques lignes à écrire, et ça cible précisément le
mode de défaillance mesuré ici. Je n'ai pas souvenir d'avoir vu cette variante
posée sous cet angle.

### 5bis.6 Existe-t-il un vérificateur pour le langage naturel ?

La question la plus profonde, et elle recadre tout le projet.

Chaque récompense utilisée ici est un **vérificateur** écrit à la main : égalité
de chaînes au test 1, parser au test 2. Le code a un vérificateur naturel (les
tests passent). Les mathématiques aussi (la preuve se vérifie). Le langage
naturel, non.

**Conjecture : pour le langage naturel, tout signal de récompense dense qui n'est
pas un modèle se réduit à une forme de prédiction.** Un signal du type « ce texte
me permet de mieux prédire un texte tenu à l'écart » est bien un vérificateur, ne
nécessite aucune cible spécifiée — mais c'est de la prédiction du token suivant
déguisée.

Si la conjecture tient, alors le pré-entraînement **n'est pas un raccourci qu'on
pourrait éviter : c'est le seul vérificateur disponible pour le langage.** Ce
serait la vraie réponse à la question du projet, et elle expliquerait pourquoi
chaque test « réussit » dès qu'on fournit un vérificateur, et échoue à passer à
l'échelle dès qu'on ne peut plus l'écrire.

Contre-exemples à chercher avant d'y croire : le langage a-t-il des propriétés
vérifiables sans modèle et non triviales ? Cohérence interne, absence de
contradiction, satisfaction de contraintes formelles, invariance par
paraphrase... Chacune est soit vérifiable mais vacuelle, soit non vacuelle mais
nécessitant un modèle. C'est le point à attaquer.

---

## 5ter. Idées reçues passées au crible de l'énumération

Règle que je m'impose ici : je rapporte aussi celles qui **tiennent**. Une liste
qui ne contient que des réfutations est le signe qu'on a cherché des
réfutations, pas qu'on a mesuré. Et chaque entrée porte sa portée : réfuté *dans
ce cadre*, ce qui n'est pas réfuté en général.

### A. RÉFUTÉ — « une récompense plus dense vaut mieux »

Le test 1 semblait l'établir. Le test 2 le contredit sur les deux plans.

- Grammaire courte, résultat final : tout-ou-rien **99,58 %** de masse valide et
  24,0 modes, contre récompense graduée **94,87 %** et 24,4 modes. Le signal
  sparse fait *mieux*.
- Et ce n'est pas un accident de trajectoire : l'optimum lui-même est pire. La
  récompense graduée paye 0,8333 les 72 quasi-ratons, donc son optimum contient
  de l'invalide par construction, en proportion exp(−0,167/β). À β=0,08 elle
  plafonne à **79,12 %** contre **99,94 %** pour le tout-ou-rien, et elle est
  battue à **tous** les β testés.

Honnêteté sur la nouveauté : que le façonnage non potentiel déplace la politique
optimale est un résultat classique (Ng, Harada & Russell, 1999). Ce qui est
absent de la pratique, ce n'est pas le théorème, c'est le **calcul** : personne
ne mesure la taxe avant de blâmer l'optimiseur, alors qu'elle est ici calculable
en forme close.

### B. RÉFUTÉ — « le bonus d'entropie empêche l'effondrement de mode »

À β=0,01 l'agent finit à **9,9 modes effectifs**. L'initialisation aléatoire en a
**47,5**. Le bonus d'entropie ne prévient pas l'effondrement, il le ralentit. Et
pour obtenir une vraie couverture il faut monter à un β où la validité s'écroule
(β=0,12 → 45,9 modes mais 57,1 % de validité).

### C. RÉFUTÉ, et c'est le plus frappant — « l'entraînement améliore ce qu'on mesure »

Sur la métrique de diversité elle-même, **le réseau non entraîné bat tous les
réseaux entraînés du plateau** : 47,5 modes contre 4,0 à 24,4. L'entraînement
*détruit* la diversité tout en améliorant la validité. Toute métrique de
diversité rapportée sans la masse valide en regard est donc ininterprétable — et
c'est exactement la frontière que j'ai failli publier.

### D. MAL CADRÉ plutôt que faux — « l'effondrement de mode est une pathologie »

À β = 0, `J = E[R]` seul : une politique concentrée sur une unique phrase valide
atteint `E[R] = 1`, soit **l'optimum exact**. L'effondrement n'y est pas un
défaut de l'optimiseur, c'est la satisfaction correcte d'un objectif qui ne
demande pas de diversité. On appelle « pathologie » le fait que l'objectif ne
contienne pas ce qu'on voulait.

### E. CONCEPT À JETER — « récompense sparse »

Le mot confond deux choses indépendantes : la **forme** de la récompense
(graduée / tout-ou-rien) et la **probabilité de succès au hasard**. Trois points
du projet le montrent :

| cadre | validité au hasard | tout-ou-rien |
|---|---|---|
| test 1, copie de 12 caractères | 1,1 × 10⁻¹¹ % | échec total (0,000 sur 30 000 ép.) |
| test 2, grammaire courte | 0,6 % | **réussite** (99,58 %) |
| test 2, grammaire longue | 0,001 % | échec total (0,0 %) |

La variable qui décide est le **taux de réussite au hasard**, pas la forme du
signal. « Sparse » n'est pas une propriété de la récompense mais du couple
récompense × taille d'espace × politique initiale. L'objection initiale du
projet était donc juste sur le fond et mal nommée.

### F. RÉFUTÉ, et le plus transférable — « un score élevé prouve que la règle est apprise »

Sur le plateau, l'agent atteint 92–100 % de grammaticalité avec
P(nom accordé | dét) = **0,333** et P(verbe accordé | nom) = **0,500** — soit
exactement 2 déterminants sur 6 et 4 noms sur 8. Aucune règle d'accord n'est
apprise. Il s'est restreint à une sous-langue entièrement au pluriel, où l'accord
est vacuellement satisfait.

Autrement dit : **on peut satisfaire un vérificateur de règles à 100 % sans avoir
appris la moindre règle**, en se réfugiant dans un sous-espace où la contrainte
est sans objet. Aucun bug dans la récompense, aucune triche — juste un score qui
ne mesure pas ce qu'on croit.

### G. CE QUI TIENT — à ne pas passer sous silence

- **REINFORCE avec baseline réduit la variance** : fonctionne, sans surprise.
- **Un GRU autorégressif peut représenter la loi cible** : confirmé exactement
  par la sonde (100 % de masse valide, 48,0 modes, P(dét) à 3 décimales de la
  valeur théorique, 3 graines sur 3).
- **La taille de l'espace gouverne l'échec du signal tout-ou-rien** : confirmé
  sur trois ordres de grandeur (tableau E).
- **Les optima d'un objectif max-ent sont de Gibbs** : confirmé numériquement sur
  les 8 000 séquences, 8 valeurs de β.

Le fait que ces quatre-là tiennent est ce qui rend les six premières crédibles.
Une liste uniquement à charge signalerait une recherche de réfutations.

---

## 5quater. Dix questions de plus, dont quatre résolues sans expérience

### Q1 — L'« anomalie » de l'épisode 45 n'existe pas : c'était mon erreur

J'ai écrit qu'à 0,6 % de validité au hasard « la première réussite devrait tomber
vers l'épisode 167, pas 45 », et j'en ai tiré que le réseau initial n'était pas
uniforme. **Faux.** J'ai comparé un tirage unique à la *moyenne* d'une loi
géométrique. La médiane est ln2/0,006 ≈ 116, et P(première réussite ≤ 45) =
1 − 0,994⁴⁵ = **24 %**. Un épisode 45 est un tirage parfaitement ordinaire.

La sonde confirme d'ailleurs que le réseau initial est à 0,60 % de masse valide,
soit exactement le hasard. Il n'y a jamais eu d'anomalie à expliquer.

**Leçon** : j'ai fabriqué un phénomène en comparant une observation à la mauvaise
statistique, puis j'ai commencé à lui chercher une cause. C'est le même mode
d'erreur que §1.1, à un étage plus bas.

### Q2 — RÉSOLUE : l'échantillonnage par rejet depuis le réseau NON entraîné bat REINFORCE

Le réseau à l'initialisation a 0,60 % de masse valide et **47,5 modes effectifs**
sur les 48. Donc échantillonner puis rejeter les phrases invalides donne, par
construction, **100 % de validité et ~47,5 modes**, au prix de ~167 tirages par
sortie acceptée.

Comparaison avec 20 000 épisodes de REINFORCE :

| méthode | validité | modes effectifs |
|---|---|---|
| rejet depuis le réseau **non entraîné** | 100 % | ~47,5 |
| REINFORCE β=0,02 | 99,99 % | 18,6 |
| REINFORCE β=0,08 | 94,87 % | 24,4 |

**L'entraînement n'a rien acheté qu'un filtre trivial ne donnait déjà, sauf du
coût d'inférence en moins.** Objection recevable : le rejet exige le vérificateur
au moment de l'inférence. Mais c'est exactement la question du projet — si on a
le vérificateur, entraîner apporte quoi ? Ici : rien, et même une perte de
diversité. Cette comparaison n'est presque jamais faite.

### Q3 — RÉSOLUE : une fois majoritairement valide, REINFORCE n'a plus AUCUN signal entre solutions

Structurel, et je ne l'avais jamais énoncé. À 99 % de validité, la baseline vaut
≈ 0,99. Une phrase valide a donc un avantage de ≈ +0,01, **identique pour les 48**.
Aucun gradient ne distingue une solution valide d'une autre : elles ont la même
récompense, donc le même avantage.

Conséquence : **la diversité n'est pas apprenable par RL sur cette tâche.** Elle
ne peut venir que de la régularisation. Ce n'est pas une faiblesse de REINFORCE,
c'est une propriété de tout objectif fondé sur l'espérance de récompense dès que
plusieurs solutions sont à égalité.

### Q4 — RÉSOLUE : le bonus d'entropie n'agit QUE là où la politique va déjà

Mécanisme précis de la falaise. Le bonus régularise l'entropie **aux états
visités**. Si `le` n'est presque jamais émis, l'état qui suit `le` n'est jamais
visité, donc ne reçoit **aucune pression entropique**. Le bonus ne peut agir que
sur la marginale de la position 0.

D'où l'impasse : pour ouvrir la branche il faut monter P(`le`) via l'entropie de
position 0 ; mais le suffixe après `le` reste non entraîné donc rapporte peu,
donc REINFORCE fait redescendre P(`le`). Le seul moyen de gagner est de pousser
l'entropie de position 0 assez fort pour que `le` soit visité *longtemps* — et ce
même niveau détruit la discrimination entre déterminants et non-déterminants à
cette position. **C'est exactement la falaise, et elle est expliquée sans
paramètre libre.**

### Q5 — L'effondrement est-il un artefact d'Adam plutôt que de REINFORCE ?

Adam normalise par paramètre, ce qui **amplifie les petits gradients cohérents**.
Dans un softmax où un token prend l'avantage, ça accélère la dynamique du riche
qui s'enrichit bien plus qu'un SGD nu. Question presque jamais posée :
l'effondrement de mode est-il une propriété de l'**objectif** ou de
l'**optimiseur** ? Test : rejouer le balayage avec SGD. Très bon marché.

### Q6 — L'ordre de génération interagit-il avec la direction de l'accord ?

La grammaire est `dét nom verbe`, et l'accord part du **nom** (position 1). Le
déterminant est donc généré **avant** de connaître le nom : l'agent doit
s'engager sur un genre et un nombre sans savoir ce qu'il dira ensuite. Le verbe,
lui, est généré après le nom : accord purement causal.

C'est précisément l'asymétrie qu'on observe — P(verbe accordé | nom) est bien
plus facile à obtenir que P(nom accordé | dét). Prédiction : réordonner en
`nom dét verbe` rendrait les deux accords causaux et devrait **casser le
verrouillage de branche**. Une ligne à changer. Je n'ai vu nulle part la question
« la direction de l'accord grammaticale interagit-elle avec l'ordre de génération
autorégressif en RL ».

### Q7 — À quel épisode le verrouillage de branche se produit-il ?

Je ne mesure que les extrémités. Si le verrouillage se joue dans les 500 premiers
épisodes, alors la totalité du budget de 20 000 est décidée par une fenêtre
minuscule, et toute intervention doit être précoce. Jamais regardé.

### Q8 — La sur-paramétrisation facilite-t-elle le verrouillage ?

Il faut distinguer 6 préfixes ; l'état caché en a 128 dimensions. Une capacité
énorme permet de mémoriser un chemin unique très précisément. Un GRU à 4 unités
s'effondrerait-il **moins** ? L'intuition courante (« plus de capacité = mieux »)
pourrait s'inverser ici.

### Q9 — La baseline scalaire est-elle une partie du problème ?

Une baseline globale mélange les familles de solutions. Une baseline
**conditionnelle au préfixe** donnerait un avantage calculé à l'intérieur de
chaque branche, ce qui pourrait empêcher qu'une branche entière soit évaluée sous
la moyenne de l'autre. Non testé.

### Q10 — Les égalités de récompense sont à la fois le cadeau et la cause

Elles offrent le certificat gratuit de §2.2 — et elles sont exactement ce qui
supprime tout gradient entre solutions (Q3). Le même fait rend le diagnostic
possible et la maladie inévitable.

Reformulation générale : **dans toute tâche admettant plusieurs réponses
également bonnes, le RL ne fournit aucun signal pour choisir entre elles.** La
diversité n'est donc pas quelque chose que le RL apprend, c'est quelque chose
qu'on lui impose. Ce qui déplace la question, en RLHF, de « comment entraîner un
modèle divers » vers « quelle régularisation encode la diversité voulue ».

---

## 7. Ce qui s'est fermé après coup, et ce que ça a cassé

Le carnet s'arrêtait à la mi-journée. Voici la suite, y compris la conclusion
publiée qu'un run tardif a démentie.

### 7.1 Bruit ou géométrie : j'ai publié la moitié de la réponse

Sur les seuls runs à β=0,01, gradient exact, le GRU s'effondre à 12,0 modes sur
3 graines/3 alors que le tabulaire atteint 48,0. J'en ai conclu, et écrit dans
l'article : *« ce n'est pas le bruit, c'est la géométrie ; aucune réduction de
variance ne sauvera la méthode »*.

Les runs à β ≥ 0,05, terminés **après** la publication, l'ont démentie :

| β | graines | gradient exact | optimum de Gibbs calculé |
|---|---|---|---|
| 0,05 | 0, 1, 2 | 48,0 modes, 50/50, 94,60 / 94,60 / 94,59 % | **94,59 %** |
| 0,08 | 0, 1, 2 | 48,0 modes, 50/50, 79,12 / 79,13 / 79,10 % | **79,12 %** |

Le GRU à gradient exact reproduit l'optimum analytique **à deux décimales sur six
runs indépendants**. Il ne s'effondre pas du tout.

**Deux régimes, transition nette entre β=0,02 et β=0,05.** En dessous, la
factorisation à paramètres partagés bloque même avec un gradient parfait. Au
dessus, le blocage disparaît et tout ce qui échoue relève de la procédure
échantillonnée. L'échantillonnage décale d'un facteur 3 à 5 la pression
entropique nécessaire : gradient exact β≈0,05, échantillonné β≈0,12 — où
l'optimum est déjà tombé à 52 % de validité.

**Confondant que je ne peux pas lever** : le run exact optimise `E[R] + β·H(p)`,
l'échantillonné utilise le bonus d'entropie standard, estimateur biaisé. À β ≥ 0,05
« bruit » recouvre peut-être « biais d'estimateur ». Seule la comparaison à faible
β est propre, tabulaire et GRU y partageant objectif et gradient.

**La leçon** : j'ai tiré une conclusion générale d'un seul point du balayage,
alors que le balayage tournait encore. Attendre la fin d'un sweep avant d'écrire
sa conclusion n'est pas de la prudence, c'est la condition minimale.

### 7.2 L'attracteur à 45,3 modes, expliqué au chiffre près

Les deux recuits de β et la trajectoire partie de la politique idéale convergent
tous vers un partage sg/pl de ~66,7 / 33,3. Ce n'est pas du bruit : 2/3–1/3 = 4/6–2/6,
soit exactement ce qu'on obtient quand **P(déterminant) est uniforme sur les 6
déterminants**, le lexique en comptant 4 singuliers et 2 pluriels.

```
24 phrases sg à masse (2/3)/24, 24 pl à (1/3)/24
H = ⅔·log₂(36) + ⅓·log₂(72) = 5,5032 bits
2^H = 45,35 modes effectifs
```

**45,3 mesuré.** Le plafond résiduel est donc un décalage de cible : **le bonus
d'entropie par token vise l'uniformité sur les tokens, pas sur les séquences.**
Les deux ne coïncident que si tous les préfixes ont le même nombre de complétions
valides — faux ici, `les` en admet 12 et `le` seulement 6.

### 7.3 Le recuit de β, seul correctif validé

| méthode | validité | modes / 48 | familles |
|---|---|---|---|
| β constant 0,02 | 99,99 % | 18,6 | 1 |
| β constant 0,12 | 57,13 % | 45,9 | 2 |
| **recuit 0,2 → 0,01** | **99,97 %** | **45,3** | **2** |
| **recuit 0,12 → 0,02** | **99,96 %** | **45,3** | **2** |

Domine les deux régimes constants, reproduit sur deux calendriers. Mécanisme : à
β élevé les six conditionnelles reçoivent du gradient, donc la représentation
partagée se forme pour toutes ; quand β redescend, l'interférence n'a plus lieu
d'être. **Le recuit ne combat pas l'effondrement, il l'empêche de se former.**

### 7.4 Le spectre ANOVA de la récompense

| récompense | ordre 1 | ordre 2 | ordre 3 |
|---|---|---|---|
| graduée | **76,1 %** | 23,9 % | **0,0 %** |
| tout-ou-rien | **4,0 %** | 30,5 % | **65,5 %** |

L'ordre 3 exactement nul de la graduée est structurel : c'est une somme de termes
par position (ordre 1) et d'accords par paire (ordre 2). L'indicateur est un
produit, d'où ses 65,5 % à l'ordre 3.

Détail d'ordre 2 pour la graduée : `pos0-1` 10,0 %, `pos1-2` 14,0 %, **`pos0-2`
0,0 %** — déterminant et verbe n'ont aucune contrainte directe, ils n'interagissent
que par le nom. **La décomposition retrouve la structure de dépendance de la
grammaire à partir de la seule récompense.**

Piège de curriculum : la séquence gloutonne d'ordre 1 est `des chat chantent`,
**invalide**, R = 0,50. Le premier signal que l'agent suit ne pointe pas vers une
solution.

### 7.5 Le signal d'ordre 1 décide de la branche, et c'est mon lexique qui le décide

Marginale `E[R | nom]` : **0,2944** pour les noms singuliers, **0,2778** pour les
pluriels, écart **+0,0167** en faveur du singulier.

Cause : j'ai mis **4 déterminants singuliers et seulement 2 pluriels**. Un
déterminant tiré au hasard s'accorde donc en nombre avec un nom singulier 4 fois
sur 6, contre 2 fois sur 6. Vérification : crédit partiel moyen 0,667 (sg) contre
0,500 (pl), écart 0,167 sur le sous-score, /3 pour la moyenne, × 6/20 pour la
dilution = **0,0167**. Exactement la valeur mesurée.

**Un déséquilibre involontaire du vocabulaire, calculable avant tout
entraînement, décide dans quelle sous-langue l'agent s'effondre.** Confirmé par
le gradient exact, qui part au singulier de façon déterministe sur toutes les
graines : le bruit était la seule chose qui permettait parfois de surmonter ce
biais.

**Complément du 30/07/2026, après une critique extérieure (§7.10).** Il y a un
second signal d'ordre 1, à la position 0, et il pointe **en sens inverse** :
`E[R | dét]` vaut 0,3089 pour `les` et `des` contre 0,2756 pour les quatre
singuliers, soit **+0,0333 en faveur du pluriel**, deux fois l'écart du nom. Les
deux causes sont orthogonales, une par trait :

| | genre | nombre | moyenne |
|---|---|---|---|
| crédit reçu par un nom **singulier** | 2/3 | **2/3** | 2/3 |
| crédit reçu par un nom **pluriel** | 2/3 | **1/3** | 1/2 |
| crédit reçu par `le` (dét sg) | **1/2** | 1/2 | 1/2 |
| crédit reçu par `les` (dét pl) | **1** | 1/2 | 3/4 |

L'avantage du nom singulier est **entièrement dans le nombre** — le genre est à
2/3 des deux côtés — donc il vient du déséquilibre 4 contre 2. L'avantage du
déterminant pluriel est **entièrement dans le genre** — le nombre est à 1/2 des
deux côtés — donc il vient du `None`. Deux accidents de lexique indépendants,
deux traits différents, signes opposés.

C'est pour ça que la séquence gloutonne d'ordre 1 est **invalide** : aucune
phrase valide ne peut satisfaire les deux positions à la fois. Le fait que
`des chat chante` soit invalide n'est pas une curiosité, c'est la **signature de
la contradiction entre marginales**, et je ne l'avais écrit nulle part.

### 7.6 L'effondrement est localisé dans la position 0

| figée | modes | sg % | pl % | P(nom\|dét) | P(verbe\|nom) |
|---|---|---|---|---|---|
| aucune | 11,5 | 0,0 | 100,0 | 0,333 | 0,500 |
| **pos0 (dét)** | **30,3** | **61,9** | **38,1** | **0,999** | **0,924** |
| pos1 (nom) | 17,7 | 0,2 | 99,8 | 0,005 | 0,875 |
| pos2 (verbe) | 8,0 | 100,0 | 0,0 | 0,500 | 0,009 |

Figer la seule marginale du déterminant fait passer P(nom accordé | dét) de 0,333
à **0,999**, pour les six déterminants, avec les deux familles vivantes. Les
lignes pos1 et pos2 sont non informatives : figer à un tirage indépendant détruit
la dépendance par construction.

Défaut de mon protocole à signaler : la validité affichée pour ces lignes est un
artefact — la position figée est exclue du gradient mais laissée libre à
l'évaluation. Seules les conditionnelles survivent à ce défaut.

### 7.7 Deux résultats nuls, et une figure qui dit plus que ce que j'y cherchais

**ACP sur la trajectoire — hypothèse réfutée.** Je pariais 2 ou 3 dimensions et un
portrait de phase dessinable. Il en faut **8 pour 90 %** du mouvement, 21 pour
99 %, 33 pour 99,9 %.

**Fonctionnelles conservées — aucune.** Zéro fonctionnelle varie de moins de 0,02.
Les neuf masses catégorie × position bougent toutes massivement.

**Mais le portrait de phase, dessiné quand même, montre autre chose.** Les trois
initialisations aléatoires démarrent **empilées sur l'optimum** :

```
distance à l'optimum dans ce plan — départ : 0,001    arrivée : 0,212
```

L'entraînement éloigne la politique **200× plus loin** de la distribution idéale
que son point de départ. Version géométrique du résultat de l'échantillonnage par
rejet. Nuance : ce plan ne porte que 56,5 % du mouvement, et l'axe validité — sur
lequel le réseau non entraîné est évidemment mauvais — est dans les 43,5 %
restants. Ce que la projection isole, c'est l'axe diversité.

### 7.8 Q-A revisité : l'optimum n'est pas un point fixe

J'avais écrit « parti de l'idéal, il s'y maintient ». Trop simple. Il **quitte
l'optimum dès les 250 premiers épisodes** (48,0 → 44,0 modes, 49,9/50,1 →
66,7/33,3), puis oscille autour de l'attracteur à 45,3 avec des excursions
jusqu'à 26,7, et finit à 43,0 après 18 250 épisodes.

Énoncé correct : *l'optimum est instable, mais le bassin dans lequel il retombe
(45,3 modes, deux familles) est incomparablement meilleur que ce qui est
atteignable depuis l'aléatoire (11,5–18,6 modes, une famille).*

### 7.9 Q-B : la diversité culmine à mi-parcours

Depuis l'aléatoire à β=0,02 : maximum de **24,0 modes à l'épisode 4 750**, KL
minimale à l'épisode 11 500, état final **11,5 modes**. Un arrêt précoce battrait
la convergence de **+12,5 modes**.

Non répliqué : une seule graine, et j'ai observé un écart run-à-run à réglages
nominalement identiques (11,5 ici contre 18,6 au balayage), probablement du
non-déterminisme multithread de torch sur CPU. **Je ne l'inscris pas comme acquis.**

> **Corrigé le 31/07/2026 (§7.11quinquies).** L'écart 11,5 / 18,6 n'était pas du
> non-déterminisme mais **deux chemins d'arrondi déterministes** sur la ligne
> d'avantage. Et le chiffre du titre change : le pic vaut 24,00 au pas 5 750 sur
> **les deux** chemins, mais l'écart d'arrêt précoce vaut **+5,38** sur le chemin
> du balayage et +12,50 sur l'autre.
>
> **RETIRÉ le 31/07/2026 après 20 graines (§7.11octies).** Écart médian
> **+0,00**, moyenne +1,03, et **3 runs sur 20** seulement dépassent 1 mode. Ce
> n'était pas un résultat, c'était une graine. Ne pas citer ce paragraphe sans
> §7.11octies.

### 7.10 Première critique extérieure : juste sur la méthode, fausse sur la conclusion

Le 30/07/2026, **dipankarsarkar** commente l'article après avoir fait tourner ma
classe `Grammaire` sans entraînement. Il calcule `E[R | premier token]` à
politique uniforme, trouve `des` et `les` à 0,3089 contre 0,2756 pour les
singuliers, identifie correctement le `None` comme mécanisme, et conclut :
*« l'effondrement était décidé avant l'épisode 1, la sous-langue dégénérée n'a pas
été trouvée par 20 000 épisodes de recherche, c'était la direction la plus raide
au pas 0 »*.

**Ses chiffres sont exacts, je les reproduis à la virgule près.** Deux précisions
seulement : `des` et `les` sont la même entrée de lexique à l'orthographe près, ils
sont donc **exactement** égaux (0,153277835… sur la grammaire longue), et son
0,1536 contre 0,1535 est du bruit d'échantillonnage ; et son `E[accord|dét]` de
0,75 contre 0,50 est bien la bonne forme close.

**Ce que la sonde rate : elle ne regarde que la position 0.** La position 1 pointe
en sens inverse (§7.5 complété). Et c'est la position 1 qui nomme la branche : le
déterminant, l'adjectif et le verbe s'accordent **avec le nom**, le nom est le seul
porteur de traits, donc « sous-langue au pluriel » est un énoncé sur le nombre du
nom, pas sur celui du déterminant.

**Qui gagne ? Les données disent le nom.** Gradient exact, même GRU, zéro
échantillonnage : **6 graines sur 6** à β = 0,01 et 0,02 finissent à **100 %
singulier**. β = 0 échantillonné : 3 graines sur 3 à 100 % singulier, 1,0 mode.
Balayage complet, 8 β × 3 graines : **15 runs singulier, 9 pluriel**. Et à son
β = 0,02 précisément :

| graine | branche | modes |
|---|---|---|
| 0 | **pluriel** | 18,6 ← le run qu'il cite |
| 1 | singulier | 11,7 |
| 2 | singulier | 12,0 |

Énoncé correct : **l'effondrement est décidé avant l'épisode 1, le coin ne l'est
pas.** Lequel des deux coins dégénérés est atteint reste une loterie de graine,
biaisée environ 2 contre 1 vers le singulier.

Son argument par la diversité ne tranche pas non plus : la sous-langue singulière
contient elle aussi exactement 24 des 48 phrases (4 déterminants × 6). 18,6 vaut
77,5 % de 24, mais les graines 1 et 2 sont à 11,7 et 12,0, soit ~49 % de **leur**
24, à β et architecture identiques.

**La question posée par dipankarsarkar — la grammaire longue s'effondre-t-elle au
pluriel elle aussi ?** Elle est de lui, pas de moi : je n'avais jamais mesuré la
branche sur la grammaire longue, et je n'y aurais pas pensé, parce que sa faible
validité (6 à 9 %) me la faisait ranger comme « échec de passage à l'échelle »
plutôt que comme un effondrement à analyser. Mesurée sur sa demande, 5 graines,
β = 0,08, 40 000 échantillons par graine :

| graine | validité | phrases valides distinctes | nombre du nom, masse valide |
|---|---|---|---|
| 0 | 6,98 % | 144 | **100,0 % sg** |
| 1 | 9,41 % | 144 | **100,0 % sg** |
| 2 | 6,79 % | 151 | 99,7 % sg |
| 3 | 7,54 % | 144 | **100,0 % pl** |
| 4 | 7,27 % | 148 | 99,9 % sg |

Réponse : **elle s'effondre sur une seule famille, et c'est le singulier 4 fois
sur 5.** Les 144 phrases valides distinctes sont exactement la taille d'une
famille (144 des 288), la répartition étant parfaitement symétrique — 8 noms × 2
déterminants × 2 adjectifs × 3 verbes × 3 adverbes = 36 chacun.

Structurellement le coin vacuellement satisfait **n'existe pas** sur la grammaire
longue : les adjectifs sont écrits pour les quatre combinaisons genre × nombre,
aucun `None` nulle part, donc `E[accord_adj_nom | nom]` vaut exactement 0,5 pour
tous les noms et la position de l'adjectif est parfaitement plate (0,1329 partout).
Passer au pluriel n'achète plus que le genre gratuit du déterminant, un
sous-score sur quatre au lieu d'un sur trois. Les deux écarts sont divisés par
exactement (3/4)(20/31) = 0,4839 : dét +0,0333 → +0,0161, nom +0,0167 → +0,0081.
**Ajouter une règle d'accord n'a ajouté que du dénominateur, aucun contre-signal.**

**Ce qu'il a raison de dire malgré tout**, et que j'aurais dû mettre en avant : la
sonde se calcule avant tout entraînement, elle coûte une énumération, et elle
appartient au protocole d'avance de phase, pas à l'analyse post-hoc. Elle existait
déjà dans le dépôt (`gradient_exact.py`, partie 1, qui imprime les deux tableaux)
mais enterrée dans un script d'analyse. Sortie en script autonome :
`src/test2_grammar/sonde_ordre1.py`, toutes positions, deux grammaires.

**La leçon à en tirer n'est pas la sienne.** Ce n'est pas « trouver le coin
vacuellement satisfait », c'est **calculer toutes les positions, parce que le coin
est là où elles se contredisent, et que le signe de la contradiction n'est pas
lisible depuis le premier token**.

**Ce que ça révèle de mon article** : je donne le tableau des marginales, je donne
`des chat chantent` invalide, et je n'écris **jamais pourquoi** cette séquence est
invalide. La phrase manquante est celle qui empêche la lecture « la position 0
décide ». Un lecteur attentif a fait exactement l'inférence que mon texte
autorisait. C'est un défaut d'écriture, pas de mesure.

### 7.11 Deuxième critique, et trois erreurs à moi dans la réponse à la première

Même interlocuteur, 31/07/2026, en réponse à §7.10. Il apporte un résultat que je
n'avais pas, et il trouve trois fautes dans ce que je venais d'écrire.

**Son résultat : les deux coins ne se valent pas.** Une politique **sans
couplage** dét → nom a un support **produit**. À validité 1 ce support doit donc
tenir dans le plus grand produit entièrement valide du coin. C'est un plafond, et
il se calcule sans entraînement. Vérifié par énumération exhaustive de tous les
sous-ensembles de noms :

| | phrases valides | plus grand produit | plafond |
|---|---|---|---|
| courte, coin pluriel | 24 | **24** = {des,les} × 4 noms × 3 verbes | 4,585 bits |
| courte, coin singulier | 24 | **12**, genre verrouillé | 3,585 bits |
| longue, coin pluriel | 144 | **72** | 6,170 bits |
| longue, coin singulier | 144 | **72** | 6,170 bits |

Le 24 pluriel est **un seul produit**, parce que le `None` supprime la contrainte
de genre entre déterminant et nom. Le 24 singulier est une **union de deux**
produits (masculin et féminin), donc il exige que la récurrence porte le genre.
Écart de plafond sur la grammaire courte : **exactement 1 bit**. Sur la longue :
**exactement 0**, parce que l'adjectif s'accorde en genre sans aucun `None` et
force donc le couplage des deux côtés.

Conséquence directe, et c'est ce que je n'avais pas vu : les 12,0 modes du
gradient exact à β = 0,01, trois graines sur trois, ne sont **pas** un tirage.
C'est le plafond exact d'une politique non couplée dans le coin singulier. Et le
coin pluriel offre 24 modes à `E[R] = 1` identique, donc à entropie strictement
supérieure : sous mon propre objectif il gagne de **β·ln2 = 0,0069** à coût nul.
Ces trois runs se sont **arrêtés avant l'optimum**, ils n'y ont pas convergé.

**Synthèse des deux lectures, et elles sont compatibles.** Mon signal d'ordre 1
au nom envoie la dynamique vers le singulier ; son plafond de produit rend le coin
pluriel meilleur à l'arrivée. L'agent va où l'ordre 1 l'envoie, puis reste bloqué
au plafond non couplé du coin où il a atterri. Prédiction de son modèle que mes
données confirment déjà : son avantage vaut β·ln2, donc il **disparaît à β = 0**,
et à β = 0 mes trois graines vont toutes au singulier avec 1,0 mode.

**Mes trois erreurs.**

**E1 — j'ai compté 24 tirages là où il y en a 3.** Mon « 15 singulier / 9 pluriel
sur 24 runs, biaisé environ 2 contre 1 » vient de 3 graines × 8 valeurs de β. Il
l'a diagnostiqué depuis le tableau seul. Les données brutes disent pire : dans le
régime d'effondrement la branche est décidée par la **graine**, β ne fait que la
recopier.

```
  graine 0 : sg pl pl pl pl
  graine 1 : sg sg sg sg
  graine 2 : sg sg sg sg
```

Ce ne sont pas 24 tirages corrélés, ce sont **3 tirages recopiés**. Wilson à
n = 24 donne [0,427 ; 0,788] ; à n = 3 il donne **[0,208 ; 0,939]**. Je n'avais
aucune information sur le biais de branche.

**E2 — mon dénominateur était faux avant même la question de l'indépendance.**
Onze runs sur 24 ne sont pas des effondrements : à β ≥ 0,08 les deux familles sont
vivantes. J'ai étiqueté chaque run par la famille majoritaire, donc un partage
50,1 / 49,9 a été compté comme « singulier ». Du bruit compté comme une branche.
Seuls **13 runs sur 24** sont de vrais effondrements.

**E3 — j'ai commis E1 le jour même où j'ai écrit la mémoire qui l'interdit.**
`un-run-nest-pas-une-propriete` dit « vérifier sur combien de runs X est vrai ».
Je l'ai fait : j'ai compté 24 lignes. La règle était insuffisante, il fallait
**compter les tirages indépendants, pas les lignes du tableau**. Corrigé dans la
mémoire.

**Symétrie à ne pas manquer.** Je lui reprochais de s'arrêter à la position 0 ;
je me suis arrêté au nombre de lignes. Même faute de forme, chacun sur son axe.

**Les 70 graines, mesurées. Il avait raison et j'avais tort.** Une seule
condition, β = 0,02, 20 000 épisodes, `balayage_70_graines.py`.

| | |
|---|---|
| singulier / pluriel | **37 / 33** |
| proportion singulier | 0,5286 |
| Wilson 95 % | **[0,413 ; 0,641]**, contient 1/2 |
| binomial contre 1/2 | p = **0,72** |
| binomial contre 2/3 | p = **0,016** |

**Le choix de branche est indiscernable d'une pièce équilibrée, et mon « biaisé
2 contre 1 vers le singulier » est rejeté à p = 0,016.** Le biais d'ordre 1 au nom
(+0,0167) existe et se calcule, mais il ne survit pas à la dynamique
échantillonnée. Ce que j'avais lu comme un biais était 3 graines recopiées 8 fois.

**Et le plafond de produit, lui, tient exactement.**

| branche | n | plafond | max observé | dépassements | pile au plafond | moyenne |
|---|---|---|---|---|---|---|
| singulier | 37 | 12 | **12,0** | **0** | 19 | 9,41 |
| pluriel | 33 | 24 | **24,0** | **0** | 6 | 15,12 |

**Zéro dépassement sur 70 runs**, et le résultat modal est le plafond lui-même
(19 runs singuliers exactement à 12,0). Mieux : les modes effectifs sont des
**produits d'entiers**, pas des valeurs quelconques — singulier {2, 4, 6, 8, 12},
pluriel {6, 8, 12, 16, 18, 24}, c'est-à-dire |A_dét| × |A_nom| × |A_verbe|. La
structure produit se lit directement dans l'histogramme.

`P(nom accordé | dét)` reste à **0,333 ± 0,003** sur les 37 runs singuliers et
0,330 ± 0,018 sur les 33 pluriels. **Aucun des 70 runs n'acquiert la
conditionnelle.**

**Ce qui sépare enfin le bruit du gradient, et c'est neuf.** Le gradient exact à
β = 0,02 atteint 24,0 modes **dans le coin singulier** (graines 0 et 2), donc il
franchit le plafond de 12 : il a acquis le couplage. REINFORCE échantillonné ne
le franchit jamais, 0 fois sur 70. Le plafond n'est donc pas une propriété de la
tâche ni de l'architecture, c'est **le plafond de la procédure échantillonnée**,
et il a une forme close.

**Sa question, mesurée : la saturation de `le` et `la` sur les runs à 12 modes.**
Six runs à 12 modes, quatre à gradient exact et deux échantillonnés, plus deux
témoins à 24 modes.

| run | modes | déterminants porteurs | masse | saturation | genre |
|---|---|---|---|---|---|
| exact β=0,01 g0 | 12,0 | `le`, `un` | 0,500 / 0,500 | **100,0 / 100,0** | m |
| exact β=0,01 g1 | 12,0 | `le`, `un` | 0,500 / 0,500 | **100,0 / 100,0** | m |
| exact β=0,01 g2 | 12,0 | `la`, `une` | 0,500 / 0,500 | **100,0 / 100,0** | **f** |
| exact β=0,02 g1 | 12,0 | `le`, `un` | 0,499 / 0,501 | **100,0 / 100,0** | m |
| échant. β=0,02 g1 | 11,7 | `le`, `un` | 0,448 / 0,551 | 95,9 / 99,9 | m |
| échant. β=0,02 g2 | 12,0 | `la`, `une` | 0,500 / 0,500 | **100,0 / 100,0** | **f** |
| **témoin** exact β=0,02 g0 | **24,0** | `le`, `la`, `un`, `une` | **0,25 × 4** | **100,0 × 4** | les deux |
| **témoin** exact β=0,02 g2 | **24,0** | `le`, `la`, `un`, `une` | **0,25 × 4** | **100,0 × 4** | les deux |

**Six sur six : deux déterminants de même genre à la moitié de la masse chacun et
100 % de saturation, les quatre autres à masse nulle.** Le produit verrouillé sur
un genre, exactement sa prédiction. Le genre lui-même est une seconde loterie :
4 runs masculins, 2 féminins.

**Nuance qui corrige sa question.** La saturation vaut 100 % dans **les deux**
structures : le run à 24 modes a aussi 100 % sur ses quatre déterminants. Ce n'est
donc pas la saturation qui discrimine, c'est le **profil de masse** — deux
déterminants à 0,5 contre quatre à 0,25. Un produit et une union de deux produits
saturent également, la différence est dans le nombre de branches ouvertes. La
saturation seule aurait donné la même valeur pour les deux et n'aurait rien
tranché.

**Et la correction de §3.3 paie tout de suite.** Dans le run exact β=0,01 g0,
`la` affiche 86,7 % de saturation sur une masse de 0,00000, et `des` 99,5 % sur
0,00000 : ce sont des conditionnelles jamais entraînées, lues sur du vide. La
colonne `accord%` à 0,00 les élimine en un coup d'œil. Sans la séparation des deux
champs, j'aurais servi « `la` à 86,7 % » comme si c'était un signal.

**Le champ qui tranchait était déjà calculé et jeté.** `analyse_exacte` renvoie
`entropie_nom_sachant_det` avec `H_max = log2(noms_compatibles)`, soit 1 bit pour
un déterminant singulier et 2 pour un pluriel. `balayage_graines.py` sauvegarde
`moyenne_cond_det` à la place. Troisième fois dans ce projet qu'une mesure
décisive existe déjà et n'est pas regardée, après la sonde d'ordre 1 (§7.10) et le
balayage multi-graines (§7.5).

### 7.11bis Troisième critique : ma statistique de couplage mesurait la couverture

31/07/2026, même interlocuteur. Il attaque la phrase « aucun des 70 runs
n'acquiert la conditionnelle » et il a raison sur les quatre points.

**`moyenne_cond_det` est une moyenne NON pondérée sur les six déterminants.** Un
softmax n'atteint jamais zéro, donc les quatre déterminants morts passent le
garde-fou `total > 0` et entrent dans la moyenne avec le même poids que les deux
vivants. **La quantité obtenue est (déterminants émis)/6, pas un taux d'accord.**
Vérifié sur mes vraies politiques, pas sur des jointes reconstruites : les 8 runs
dont j'ai l'analyse complète donnent 0,3333 pour 2 déterminants vivants et 0,6667
pour 4, à quatre décimales.

Conséquence directe et fatale à ma phrase : **un effondrement singulier à 12 modes
et un effondrement pluriel à 24 modes lisent tous deux 0,3333.** C'est exactement
la distinction que je faisais porter à ce chiffre.

**Mais son remède est pire que le mal.** Il propose de pondérer par la masse. J'ai
reconstruit ses quatre structures et passé les trois statistiques dessus :

| structure | modes | non pondérée | pondérée | **I(dét;nom)** |
|---|---|---|---|---|
| singulier verrouillé genre | 12,0 | 0,433 | **1,0000** | **0,0000** |
| pluriel `les`/`des` | 24,0 | 0,400 | **1,0000** | **0,0000** |
| singulier 24, genre acquis | 24,0 | 0,733 | **1,0000** | **1,0000** |
| les six, tout couplé | 48,0 | 1,000 | **1,0000** | **1,5000** |

La pondérée vaut 1,0000 pour les quatre : elle ne distingue plus rien. Parce
qu'un accord parfait s'obtient **par restriction** aussi bien que par
conditionnement — un produit verrouillé sur un genre est parfaitement accordé
sans le moindre couplage.

**La quantité qui répond à la question est l'information mutuelle I(dét ; nom).**
C'est une **dépendance**, pas un taux d'accord : 0 pour un produit, quelle que
soit sa validité. Ajoutée à `analyse_exacte`, avec `cond_det_pondere` et
`determinants_emis`.

**Les 70 graines relancées avec la bonne statistique, et ma conclusion survit :**

| | |
|---|---|
| I(dét;nom) médiane | **0,0000 bit** |
| I(dét;nom) maximum sur 70 | **0,0377 bit** |
| runs au-dessus de 0,05 bit | **0 / 70** |
| conditionnelle pondérée | 0,9941 ± 0,0393 — inutilisable, comme prévu |
| déterminants émis | 1 pour 13 runs, 2 pour 57 |

Il faut 1,0 bit pour l'union singulière et 1,5 pour la politique complète. Le
maximum atteint par 70 runs est **0,038**. « Aucun run n'acquiert la
conditionnelle » est donc vrai, mais je l'avais affirmé sur une statistique qui
ne pouvait pas le dire.

**Ses deux autres points, tous deux justes.** Le coin pluriel a un écart nul par
construction (24 valides, plus grand produit 24), donc mes « 0 violations sur
70 » sont en réalité **0 sur 37** : seul le coin singulier peut falsifier le
plafond, ma grammaire est asymétrique comme appareil de mesure. Et sa borne à 36
est exacte, mesurée à 36,0 modes avec I = 0,918 bit : le recuit à 45,3 franchit
donc une barre plus haute que les 24 que je m'étais donnés.

**Et l'audit qu'impose la cinquième occurrence.** `masse_par_determinant` existait
dans `analyse_exacte` et n'arrivait pas au tableau de résultats — même faute que
`saturation_pct`, un commit plus tôt. J'ai donc audité les huit moyennes non
pondérées sur une dimension de tokens dans tout le dépôt. **Six sont correctes,
deux sont fausses, et la ligne de partage est nette :**

| moyenne sur | verdict |
|---|---|
| conditionnelles **observées** (`moyenne_cond_det`, `moyenne_cond_nom`) | **faux** |
| conditionnelles **interventionnelles** (`test_conditionnel`, token forcé) | correct |
| marginales à politique uniforme (`sonde_ordre1`, `gradient_exact`) | correct |

**Observationnel contre interventionnel.** Quand je force le token, chaque ligne
existe vraiment et les poids égaux sont justes. Quand je l'observe, les lignes
mortes sont des artefacts du softmax. Les deux versions sont dans le même fichier
et j'ai mis la mauvaise dans le tableau. C'est le critère qui manquait, et il vaut
mieux qu'une sixième correction ponctuelle.

**Sa dernière question : « avez-vous encore les 70 politiques, ou seulement les
lignes ? »** Seulement les lignes. Relancé en sauvegardant la masse par
déterminant, la conditionnelle détaillée, l'information mutuelle **et les 70
poids**.

### 7.11ter Hypothèse réfutée le 31/07/2026 : REINFORCE ne résout pas non plus le problème restreint

Approfondissement, question 1. J'avais avancé, en voyant 19 des 37 runs
singuliers exactement à 12,0 : *un plafond atteint aussi précisément n'est pas
une contrainte subie, c'est un optimum ; REINFORCE résout donc exactement le
problème restreint aux politiques sans couplage et échoue uniquement à quitter la
classe.*

**Faux.** `optimum_produit.py` optimise le même objectif `E[R] + β·H` par gradient
exact sur trois lois indépendantes p(d), p(n), p(v), donc `I(dét;nom) = 0` par
construction.

| β | classe produit | classe libre |
|---|---|---|
| 0,01 | **24,00** modes, I = 0 | 48,00 modes, I = 1,500 |
| 0,02 | **24,00** | 48,00, I = 1,497 |
| 0,05 | **24,00** | 48,00, I = 1,181 |
| 0,08 | **24,00** | 48,00, I = 0,744 |

Trois graines sur trois, à tous les β : l'optimum de la classe produit vaut
**24,00 modes** et se place dans le coin **pluriel**, qui est le plus grand
produit global. REINFORCE se pose sur 12 une fois sur deux. **Il n'est donc pas à
l'optimum de la classe restreinte, il est à un optimum LOCAL de cette classe.**

Il y a donc **deux échecs emboîtés**, pas un :

| niveau | ce qui est raté | fréquence |
|---|---|---|
| 1 | trouver le meilleur produit **du coin où il est** | 19/37 sg, **6/33** pl |
| 2 | trouver le meilleur produit **tout court** (coin pluriel, 24) | 37 runs sur 70 le ratent |
| 3 | **quitter la classe produit** | **0/70** |

Ce qui survit de mon énoncé : *conditionnellement au coin, REINFORCE atteint le
produit maximal de ce coin environ une fois sur deux.* Et le coin pluriel, dont le
produit est plus grand, est **moins bien rempli** (6/33) que le singulier
(19/37) — plus il y a à couvrir, moins c'est couvert.

**Ce que ça confirme de son argument.** Le coin singulier coûte exactement
`log2(24/12) = 1 bit`, soit `β·ln2 = 0,0139` d'objectif à β = 0,02, à récompense
strictement égale. Sa phrase « ces runs se sont arrêtés avant l'optimum plutôt
que d'y converger » est donc vraie **deux fois** : en dessous de l'optimum libre
(48) et en dessous de l'optimum produit (24). Vérifié par optimisation directe de
la classe restreinte, pas par argument.

### 7.11quater Le couplage se décide tard, et l'échantillonnage écrase la politique en un point

Approfondissement, question 2. `trajectoire_couplage.py` suit I(dét ; nom) pas à
pas, sonde exacte, β = 0,02, trois graines par procédure.

**Rien ne prédit à l'initialisation.** I au départ vaut 0,0045 / 0,0045 / 0,0035
bit, et les six masses de déterminants sont toutes entre 0,042 et 0,057, sans
structure qui distingue la graine qui va coupler de celle qui ne le fera pas. Le
prédicteur que je cherchais dans les marginales de position 0 **n'existe pas**.

**Le gradient exact tient le plafond mille pas, puis en sort.**

```
  pas     0 : I = 0.0045 |  47.54 modes | valide   0.60 %
  pas   100 : I = 0.0000 |  12.00 modes | valide  99.99 %
  pas  1000 : I = 0.0000 |  12.00 modes | valide  99.97 %
  pas  1250 : I = 0.8518 |  17.87 modes | valide  99.92 %
  pas  1500 : I = 0.9980 |  24.00 modes | valide  99.98 %
```

**12,00 modes exactement, I strictement nul, pendant mille pas, puis échappée.**
Donc le plafond du coin singulier n'est pas un bassin, c'est un **plateau**, et on
peut en sortir sans le moindre bruit. Les instants d'échappée varient beaucoup —
pas 1250 et pas 2875 sur deux graines, jamais sur la troisième en 4 000 pas.

**Et voilà ce qui sépare vraiment les deux procédures.**

| procédure | modes au **minimum** de la trajectoire | à quel pas | validité alors |
|---|---|---|---|
| exact, 3 graines | **10,7 / 11,1 / 11,2** | 25 | 99,5 à 99,9 % |
| échantillonné, 3 graines | **1,09 / 1,88 / 1,18** | 400 à 800 | 87,7 à 99,4 % |

Toutes les trajectoires démarrent à **47,5 modes** — le réseau non entraîné —
puis l'entraînement détruit la diversité. Mais l'échantillonné l'écrase jusqu'à
**une seule phrase** avant de la reconstruire, alors que l'exact ne descend jamais
sous 10,7.

**Hypothèse mécaniste, explicitement non démontrée.** Reconstruire une politique
depuis un point quasi déterministe se fait **position par position** — c'est ce
que le bonus d'entropie sait faire, il agit sur des conditionnelles par position.
Or une reconstruction position par position engendre **un produit par
construction**. Pour obtenir du couplage il faudrait ouvrir une direction
*jointe*, ce que le terme d'entropie par position ne fait jamais. Le gradient
exact, qui ne passe pas par le point, garde assez de structure jointe pour
trouver la direction couplée plus tard.

Ce que ça prédit, et qui se teste : **la profondeur de l'effondrement transitoire
doit prédire l'acquisition du couplage.** Mesurable sur les 70 politiques
sauvegardées si on refait les trajectoires, ou sur un balayage dédié.

Et ça donne au recuit une explication qu'il n'avait pas. §7.3 disait « garder
toutes les conditionnelles entraînées pendant que la représentation partagée se
forme ». La vraie raison serait plus simple : **β élevé au début empêche
l'écrasement en un point**, donc la politique n'a jamais à se reconstruire depuis
un produit.

**Anomalie à ne pas enterrer.** La graine 0 échantillonnée donne 11,50 modes ici
et **18,6** dans `balayage_70_graines.py`, à configuration nominalement
identique, et les deux valeurs sont reproductibles. §7.9 attribuait cet écart au
non-déterminisme multithread de torch ; les deux scripts sont désormais en
mono-thread, donc **cette explication ne tient plus**. Il y a une différence de
chemin de code que je n'ai pas trouvée, et tant qu'elle n'est pas trouvée l'un
des deux chiffres vient d'un code que je n'ai pas audité.

> **RÉSOLU le 31/07/2026 en §7.11quinquies** — c'est la ligne d'avantage, et un
> arrondi de scalaire. Les deux chiffres sont sains.

### 7.11quinquies Deux chemins numériques dans mon dépôt, et un arrondi qui déplace un titre

Quatrième critique, 31/07/2026. Il trouve que la ligne d'avantage n'existe pas en
une seule version dans le dépôt, et que les deux versions ne calculent pas la
même chose.

```
rl_grammaire.py:141              (recompenses_t - baseline).detach()
stabilite_et_trajectoire.py:79   torch.tensor(r - baseline, dtype=torch.float32)
parametrisation_et_recuit.py:90  idem
localisation_effondrement.py:55  idem
trajectoire_couplage.py:84       torch.tensor(r - base).detach()
```

La première soustrait **en float32** : `recompenses_t` est déjà float32 et
`baseline` est un flottant Python, donc la promotion tenseur-scalaire arrondit la
baseline **avant** de soustraire, soit deux arrondis. Les autres soustraient deux
float64 puis arrondissent une fois. Vérifié :

```
  float32 d'abord : 0.08333331346511841
  float64 d'abord : 0.0833333358168602
```

**Correction à sa lecture : il y a deux chemins, pas trois.** Il annonce que
`trajectoire_couplage.py:84`, sans `dtype`, laisse l'avantage en float64 et
promeut la perte. Faux en torch : `torch.get_default_dtype()` vaut **float32**,
donc `torch.tensor(x)` sur un flottant Python rend un tenseur float32. Même
dtype, même valeur au bit près, même dtype de perte. Son raisonnement serait
correct en numpy. Ça ne change pas sa conclusion — le chemin float32 est bien
isolé — mais le décompte est de deux.

**Divergence mesurée sur le flux de récompenses réel** (`chemin_avantage.py`,
partie A). `recompense_graduee` rend des tiers et des neuvièmes, dont aucun n'est
exact en binaire, donc les deux lignes cessent d'être le même calcul tout de
suite :

| graine | premier désaccord | % des 2 000 premiers pas | % ensuite | écart relatif max |
|---|---|---|---|---|
| 0 | pas **5** | 57,7 | 24,2 | 5,45e-06 |
| 1 | pas **5** | 78,3 | 29,6 | 7,96e-06 |
| 2 | pas **4** | 79,0 | 26,1 | 5,45e-06 |

Son écart relatif maximal de 5,4e-06 est retrouvé exactement. Ses pourcentages
diffèrent des miens (49 % et 1,5 % contre 58-79 % et 24-30 %) parce que sa
seconde fenêtre est « après saturation de la récompense » et la mienne « après le
pas 2 000 » ; je ne prétends pas qu'il a tort, les définitions ne coïncident pas.

**L'anomalie 18,6 contre 11,50 est intégralement expliquée.** Même graine, même
boucle, seule la ligne d'avantage change :

| chemin | modes finaux, graine 0 |
|---|---|
| float32 (`rl_grammaire:141`) | **18,62** — le chiffre du balayage |
| float64 puis arrondi | **11,50** — le chiffre de la trajectoire |
| float64 sans `dtype` | **11,50**, identique au précédent |

Un bit suffit parce que `distribution.sample()` est un seuil sur un tirage
uniforme : il finit par faire basculer un token, après quoi les deux runs ne
partagent plus que la graine. Et les deux restent reproductibles parce que **les
deux arrondis sont déterministes** — ce que « non-déterminisme multithread »
n'expliquait pas, et c'est ce qui aurait dû me mettre la puce à l'oreille.

**Sa deuxième question, et sa prédiction est juste au centième.** Le maximum de
modes, **restreint aux pas où la validité dépasse 90 %** :

| chemin | graine | pic | au pas | fin | écart d'arrêt précoce |
|---|---|---|---|---|---|
| float32 | 0 | **24,00** | 5 750 | 18,62 | **+5,38** |
| float32 | 1 | 12,00 | 9 000 | 11,72 | +0,28 |
| float32 | 2 | 12,00 | 16 500 | 12,00 | 0,00 |
| float64 | 0 | **24,00** | 5 750 | 11,50 | **+12,50** |
| float64 | 1 | 12,00 | 11 750 | 12,00 | 0,00 |
| float64 | 2 | 8,00 | 5 750 | 8,00 | 0,00 |

Il annonçait « +5,4 et non +12,5 » à partir de l'arithmétique seule, sans rien
lancer. Mesuré : **+5,38**.

Deux choses de plus que sa question ne demandait pas. **Le pic est le même sur
les deux chemins** — 24,00 au pas 5 750 exactement — donc c'est le *point
d'arrivée* qui dépend de l'arrondi, pas le sommet de la trajectoire. Et **une
seule graine sur trois montre un écart** : +0,28 et 0,00 pour les deux autres.
Le titre du §7.9 repose donc sur une graine, deux fois de suite.

**Défaut de ma propre mesure, corrigé en route.** Ma première colonne « modes
max » donnait 47,54 au pas 0 pour tous les runs : c'est le réseau non entraîné,
qui domine l'argmax et n'a rien à voir avec l'arrêt précoce. Restreindre aux pas
à validité ≥ 90 % était nécessaire pour que la question ait un sens.

**Ligne canonique.** Le chemin float64 est le bon : un seul arrondi au lieu de
deux, et c'est déjà ce que font quatre scripts sur cinq. Mais le basculer en
silence réécrirait tous les chiffres archivés sous DOI. `entrainer` prend donc un
paramètre `chemin_avantage` explicite, **défaut `"float32"` pour ne rien changer
sans le dire**, et le balayage 70 graines est relancé sur `"float64"` pour
répondre à la seule question qui compte : *les conclusions agrégées survivent-elles
au changement de chemin ?* Si oui elles sont robustes, si non elles étaient des
artefacts d'arrondi.

**Ménage, son dernier point.** Seuls deux fichiers épinglaient les threads
eux-mêmes ; les autres dépendaient du shell, donc se dés-épinglaient
silencieusement pour qui les relance. `torch.set_num_threads` est maintenant dans
`rl_grammaire.py`, que **14 scripts importent**, avec `RDTRL_THREADS` pour revenir
en arrière sur les calculs à gros lot.

### 7.11sexies L'arrondi est une meilleure expérience que la graine

Le bug de §7.11quinquies donne, sans le vouloir, le contrôle que je cherchais
depuis le début du test 2.

**Changer de graine change deux choses à la fois** : l'initialisation *et* toute
la trajectoire d'échantillonnage. C'est un confondant que je traîne depuis le
premier balayage. **Changer la ligne d'avantage n'en change qu'une** :
`fixer_graine` puis `PolitiqueGRU` donnent des poids initiaux identiques au bit
près, et les deux runs ne divergent qu'au pas 4 ou 5, pendant l'entraînement.
Même point de départ, trajectoire différente. C'est exactement la dissociation
qu'une graine ne permet pas.

**Résultat, 70 graines sur chaque chemin.**

| | float32 | float64 |
|---|---|---|
| singulier / pluriel | 37 / 33 | **37 / 33** |
| Wilson 95 % | [0,413 ; 0,641] | **[0,413 ; 0,641]** |
| p contre 1/2 | 0,7202 | **0,7202** |
| dépassements du plafond | 0 | **0** |
| I(dét;nom) max | 0,0377 bit | 0,0158 bit |
| runs à I > 0,05 | 0/70 | **0/70** |
| modes, branche sg | 9,41 ± 3,13 | 8,84 ± 3,13 |
| modes, branche pl | 15,12 ± 5,90 | 16,08 ± 6,23 |

**Les 70 graines sur 70 gardent le même coin.** Zéro bascule, alors que les
trajectoires diffèrent sur 58 à 79 % des 2 000 premiers pas.

**Mais le remplissage du coin, lui, ne résiste pas.** Seuls 21 runs sur 70 ont les
mêmes modes effectifs au centième, la corrélation vaut 0,68, l'écart absolu moyen
est de 2,87 modes et monte à 12,7.

**Énoncé : l'initialisation décide du coin, la trajectoire décide du
remplissage.** Deux niveaux, deux causes, séparés par une manipulation qui ne
touche qu'à l'une des deux. Ça recoupe §7.11quater : le coin est choisi tôt, par
la marginale d'ordre 1 et les poids initiaux ; le remplissage est le produit de
la reconstruction depuis un point quasi déterministe, où le bruit entre.

**Et les trois conclusions du dépôt sont robustes au dernier bit** : pièce
équilibrée à l'identique, plafond jamais franchi sur les deux chemins,
`I(dét;nom)` nulle sur les deux. Ce n'étaient pas des artefacts d'arrondi.

> **Phrase retirée le 31/07/2026.** J'avais écrit ici « je ne connais aucun
> résultat de RL publié pour lequel ce contrôle ait été fait ». C'est une
> revendication de nouveauté déduite de ma propre ignorance de la littérature,
> que je n'ai pas explorée. Le contrôle est bon ; savoir s'il est inédit
> demanderait une recherche bibliographique que je n'ai pas faite. Ne pas la
> remettre dans un article.

**Le plafond n'est qu'un attracteur faible.** Parmi les 25 runs float32 posés
exactement sur le plafond de leur coin, 11 y sont encore en float64 (44 %) ; parmi
les 45 en dessous, 10 gardent leur valeur (22 %). Être au plafond double la
probabilité de reproduire, sans la garantir.

**Deux bugs à moi, trouvés en fixant ça.** Les poids étaient nommés
`politique_b{β}_g{graine}.pt` sans le chemin numérique : relancer sur l'autre
chemin a **écrasé les 70 politiques float32**, une heure après que j'aie écrit
qu'elles étaient sur disque pour ne plus avoir à réentraîner. Et le motif de
fusion `..._b0.02_*.json` ramassait les tranches de l'autre chemin **et sa propre
sortie**, soit 13 fichiers pour 6 tranches, donc des graines comptées deux ou
trois fois. Les deux sont corrigés, le second avec un garde-fou qui compte les
doublons et le dit.

> **Ce compte est faux, il y en a eu cinq. Voir §7.11nonies**, écrit après coup :
> les trois autres sont arrivées dans l'heure qui a suivi ce paragraphe.

### 7.11septies Décision : float64 devient le défaut, et pourquoi c'était facile

Sa question de fin était *« quelle ligne veux-tu canonique ? »*, c'est-à-dire :
le dépôt contient deux comportements sans le dire, lequel devient **le** bon.

Je m'attendais à un arbitrage entre justesse et vitesse. Il n'y en a pas.
Mesuré :

```
  chemin float32 :  19.46 us par appel
  chemin float64 :   4.57 us par appel   (+77 %)
```

**Le chemin plus juste est aussi 4× plus rapide sur cette ligne.** Parce que le
nom est trompeur : rien n'est stocké en double, le tenseur produit est float32
dans les deux cas. Un `float` Python **est déjà** un double, donc `r - baseline`
en Python est natif et gratuit, et il ne reste qu'une création de tenseur.
L'autre chemin crée un tenseur, appelle un noyau torch pour la soustraction
tenseur-scalaire, puis détache : plus d'opérations, et un arrondi de plus.

Honnêteté sur l'ordre de grandeur : 15 µs gagnées sur un pas qui en coûte
~6 800, soit **0,2 %** au total. Ce n'est pas un argument de performance, c'est
que la performance ne s'oppose pas à la justesse ici.

**Le float32 n'était pas un choix, c'était un accident d'écriture** dans une
seule fonction, minoritaire dans son propre dépôt : six fichiers sur onze
faisaient déjà l'autre. Défaut basculé.

**Ce que la bascule coûte, inventorié plutôt qu'estimé.**

| catégorie | scripts | à refaire |
|---|---|---|
| passent par `entrainer()` | `rl_grammaire`, `balayage_graines`, `balayage_70_graines`, `sonde_ordre1`, `produit_et_saturation` | 5, dont un déjà fait |
| ligne float64 propre | 6 fichiers | 0 |
| **aucun entraînement échantillonné** | `gradient_exact`, `optimum_produit`, `optimum_gibbs`, `verifier_dominance`, `sonde_capacite`, `grammaire` | **0** |

Le fait qui rassure et qu'il faut retenir : **les résultats qui portent le plus
n'ont pas de ligne d'avantage du tout.** Plafond de produit, optimum de Gibbs,
marginales d'ordre 1, sonde de capacité, optimum de la classe produit — gradient
exact ou forme close. La bascule ne peut pas les toucher.

**Et le vrai coût n'est pas le calcul.** ~1 h 30 de runs, contre la reprise de
dizaines de chiffres cités dans l'article publié, `ANALYSE_TEST2.md` et ce
carnet. C'est la réécriture qui décide, pas le CPU.

**Quatrième collision de noms de la journée, désamorcée avant.** Relancer aurait
écrasé `rapport.json`, `balayage_graines.json` et les CSV float32.
`relancer_float64.py --archiver` copie `results_test2/` d'abord. J'ai préféré une
copie de dossier à un suffixe sur chaque sortie : moins invasif, et ça garde de
quoi comparer les deux chemins ligne à ligne.

### 7.11octies L'arrêt précoce ne gagne rien, sauf dans un coin sur deux

Il écrivait, à propos de l'écart d'arrêt précoce : *« same sign, under half the
size, and it is a headline »*. Il avait raison de le signaler, et la mesure va
plus loin que sa correction : **le titre ne tient pas du tout.**

Vingt graines tracées sur le chemin canonique, pic restreint aux pas où la
validité dépasse 90 % :

| | |
|---|---|
| écart moyen | +1,03 mode |
| écart **médian** | **+0,00** |
| runs avec un écart > 1 mode | **3 / 20** |
| écart maximum | +8,13 (graine 6) |

**Dix-sept runs sur vingt ne gagnent rien.** Le §7.9 annonçait « un arrêt précoce
battrait la convergence de +12,5 modes » à partir d'**une** graine ; corrigé une
première fois à +5,38 par le changement de chemin, il tombe à une médiane de zéro
dès qu'on regarde vingt graines. Ce n'était pas un effet, c'était un run.

**J'ai cru qu'il restait quelque chose de conditionnel au coin. C'est faux
aussi.** Sur le chemin float32, les trois runs qui gagnent sont tous pluriels, et
j'en avais tiré « l'arrêt précoce n'est utile que là où le plafond est haut ».
Les mêmes 20 graines sur le chemin **canonique** disent le contraire :

| | float32 | float64 (canonique) |
|---|---|---|
| médiane | +0,00 | +0,03 |
| runs > 1 mode | 3 / 20 | **5 / 20** |
| dont coin pluriel | **3 / 8** | 2 / 8 |
| dont coin singulier | **0 / 12** | **3 / 12** |

**Trois des cinq sont singuliers sur le chemin canonique.** Le « 0 sur 12 » qui
fondait toute l'interprétation était un artefact d'un seul chemin numérique.

Ce qui survit des deux côtés, et seulement ça : **la médiane est nulle, la grande
majorité des runs ne gagne rien à s'arrêter tôt.** Toute lecture plus fine que
celle-là n'a pas résisté à un changement d'arrondi.

**Quatrième fois dans la journée**, et cette fois ce n'était même pas une graine
unique : vingt graines, mais un seul chemin numérique. Le contrôle qui manquait
n'était pas « plus de graines », c'était **la même mesure sur l'autre chemin** —
celui-là même que je venais de rendre canonique.

**Ce que ça dit de ma méthode plus que du résultat.** Trois fois aujourd'hui, un
chiffre publié s'est révélé être une graine : le biais de branche 2 contre 1, la
sous-langue « au pluriel », et maintenant l'arrêt précoce. Les trois ont survécu
parce que je n'avais pas de raison de relancer un résultat qui ne me gênait pas.
La règle à en tirer n'est pas « répliquer », que je savais déjà, c'est
**répliquer d'abord ce qui arrange**.

### 7.11nonies Cinq fois le même défaut en une session : l'artefact ne porte pas sa provenance

Le §7.11sexies annonçait « deux bugs à moi ». Il y en a eu **cinq**, tous du même
défaut, et les trois derniers sont arrivés **après** que j'aie écrit ce
paragraphe. Ce n'est donc pas une série de distractions, c'est un schéma :

> **un artefact qui n'encode pas la dimension que le run fait varier finit en
> collision silencieuse ou en fausse étiquette.**

| artefact | dimension omise | conséquence |
|---|---|---|
| `politique_b{β}_g{n}.pt` | le chemin numérique | **70 politiques écrasées** |
| motif `..._b0.02_*.json` | idem, **plus sa propre sortie** | 13 fichiers pour 6 tranches |
| `chemin_avantage_{chemins}.json` | la plage de graines | deux tranches parallèles dans un fichier |
| `rapport.json`, `balayage_graines.json` | le chemin numérique | attrapé avant, dossier archivé |
| étiquette « float64 path » d'une figure | **écrite en dur** à côté d'un chargement avec repli | la figure allait annoncer un chemin en traçant les chiffres de l'autre |

**La cinquième est la pire, et elle mérite d'être racontée.** J'avais ajouté aux
quatre panneaux de `figure_comparaison.py` une mention du chemin numérique,
précisément parce que trois panneaux sur quatre venaient de chemins différents et
que la figure ne le disait pas. Une heure plus tard, j'ai basculé le panneau D
sur le float64 en écrivant l'étiquette **en dur**, alors que les données
n'existaient pas encore et qu'un repli chargeait le float32. La figure aurait
annoncé *float64* en traçant du *float32* — le défaut exact que l'étiquette
venait d'être ajoutée pour empêcher.

Règle générale qui en sort : **une étiquette écrite en dur à côté d'un chargement
conditionnel est un mensonge en attente.** Elle se calcule depuis la donnée
réellement lue, et le repli doit dire qu'il s'est déclenché :

```python
lignes, chemin_utilise = [], "float64 path"
...
if not lignes:
    chemin_utilise = "float32 path — float64 not measured yet"
```

**Correctifs appliqués :** la dimension variée est dans le nom ; le glob de
fusion se termine par `_[0-9]*` pour ne pas ramasser sa propre sortie ; un
garde-fou compte les doublons après fusion et le signale ; `relancer_float64.py`
archive le dossier de résultats plutôt que de renommer dix sorties ; et les
étiquettes de figure se calculent.

**Ce que j'en retiens sur la forme des erreurs de ce projet.** Aucune de ces cinq
n'est une erreur de raisonnement, et aucune n'aurait été trouvée en relisant le
code pour l'algèbre — exactement ce que dipankarsarkar écrivait à propos de
l'arrondi : *« it is the dtype of one scalar, which is exactly the kind of thing
reading for algebra does not catch »*. Les erreurs qui survivent ici ne sont pas
dans les idées, elles sont dans la plomberie.

### 7.11decies La figure de synthèse

`src/tools/figure_comparaison.py` → `figures/comparaison_test2.png`. Quatre
panneaux sur les mêmes runs :

| | question |
|---|---|
| **A** | les deux chemins graine par graine — 70/70 gardent le coin, 21/70 les modes |
| **B** | le plafond n'est jamais franchi, et les modes sont des produits d'entiers |
| **C** | exact contre échantillonné : la profondeur de l'effondrement, 10,7 contre 1,09 |
| **D** | l'arrêt précoce, 20 graines, médiane +0,00 |

Chaque panneau **affiche le chemin numérique qui l'a produit**, en haut à droite.
Ce n'est pas une coquetterie : trois panneaux sur quatre viennent de chemins
différents, et on venait de passer la journée à montrer que le chemin déplace les
chiffres.

Palette réduite à deux teintes catégorielles, validées en mode « toutes paires »
avant d'écrire la première ligne de tracé : CVD ΔE 24,7 et vision normale 33,6,
tous deux très au-dessus des seuils.

### 7.11undecies Ce que la bascule a réellement déplacé

Les cinq scripts concernés ont été relancés sur le chemin float64 en 37,8 min.
Comparaison au dossier archivé, tableau du balayage d'entropie, **graine 0 comme
avant** :

| β | validité f32 | validité f64 | modes f32 | modes f64 |
|---|---|---|---|---|
| 0,0 | 100,00 | 100,00 | 1,0 | 1,0 |
| 0,01 | 99,84 | 99,99 | **9,9** | **18,0** |
| 0,02 | 99,99 | 99,94 | **18,6** | **11,5** |
| 0,05 | 92,65 | **99,76** | 23,8 | 19,9 |
| 0,08 | **94,87** | **84,11** | 24,4 | 26,5 |
| 0,12 | 57,13 | 55,31 | 45,9 | 45,4 |
| 0,2 | 20,59 | 19,96 | 41,2 | 45,0 |
| 0,35 | 5,27 | 5,27 | 43,5 | 43,5 |
| 0,5 | 3,01 | 3,01 | 43,6 | 43,6 |

Et deux chiffres qui sortent souvent :

| | float32 | float64 |
|---|---|---|
| tout-ou-rien, grammaire courte | 99,58 % | 99,91 % |
| **grammaire longue, graduée** | **6,4 %** | **15,8 %** |

**Le 6,4 % de la grammaire longue devient 15,8 %, soit 2,5 fois plus.** C'est un
chiffre cité dans l'article publié.

**Ce n'est pas une surprise, c'est la confirmation d'un défaut déjà écrit.** Le
§4.2 dit depuis le début que le balayage à graine unique ne permet pas de tracer
une frontière ; le §7.11sexies a montré sur 70 graines que le détail par graine
n'est pas robuste au dernier bit. Ce tableau **est** un tableau à graine unique :
il devait donc bouger, et il bouge.

Ce qui ne bouge pas, et c'est ce qui compte : les valeurs extrêmes (β = 0 → un
seul mode ; β ≥ 0,35 → validité effondrée), l'allure de la frontière, et toutes
les conclusions établies sur 70 graines ou par énumération.

**Conséquence pratique pour la v0.4.0** : les tableaux à graine unique doivent
être remplacés par des moyennes multi-graines avec écart-type, pas simplement
remis à jour avec les nouveaux chiffres. Sinon on republie la même fragilité avec
d'autres décimales.

### 7.13 Le test de renversement : ma première version ne testait rien

Le §7.12 désignait le test de renversement comme la seule expérience qui décide
si le plafond de produit est une loi ou une coïncidence de mon lexique. Je l'ai
conçu, et **Théo a vu qu'il était vide avant que je le lance.**

**La version fausse.** J'ai construit un lexique où la neutralité de genre passe
des déterminants pluriels aux singuliers, en gardant les mêmes 20 tokens, le même
espace de 8 000, les mêmes 48 phrases valides et les mêmes deux coins de 24.
Résultat annoncé : plafonds échangés, 24 au singulier et 12 au pluriel, et les
deux marginales d'ordre 1 échangées aussi. Tout basculait proprement.

**Trop proprement.** Les noms et les verbes du lexique standard sont **déjà
symétriques en nombre** — 2 par (genre, nombre), 3 verbes de chaque. Échanger le
nombre des déterminants **est donc le renommage `sg` ↔ `pl`**, et rien d'autre.
Vérifié sur les multiensembles de traits :

```
  det     standard avec sg<->pl : {('f','pl'): 2, ('m','pl'): 2, (None,'sg'): 2}
  det     renverse              : {('f','pl'): 2, ('m','pl'): 2, (None,'sg'): 2}
  ISOMORPHES sous le renommage sg<->pl : True
```

Les 70 graines auraient produit l'image miroir **par construction**, en une heure
de calcul, et j'en aurais tiré une confirmation qui ne confirme rien : elle aurait
seulement prouvé que mon code ne teste pas les chaînes `"sg"` et `"pl"`.

**Le principe qui manquait, et qui vaut au-delà de ce projet :**

> **Un renommage peut permuter, il ne peut pas changer un rapport.** Un contrôle
> parfaitement symétrique est souvent un contrôle parfaitement vide. Pour qu'un
> renversement teste quelque chose, il faut faire varier la **valeur** de la
> quantité prédite, pas ses étiquettes.

Je garde la variante `renverse` dans le code, documentée comme contre-exemple.
La supprimer effacerait la leçon.

**La version qui teste : trois genres.**

| | standard | trois_genres |
|---|---|---|
| tokens | 20 | 26 |
| espace | 8 000 | 17 576 |
| phrases valides | 48 | 72 (force brute confirmée) |
| taille des deux coins | 24 / 24 | 36 / 36 |
| **plafonds** | 12 et 24 | **36 et 12** |
| **rapport** | **2** | **3** |

Le coin singulier a des déterminants neutres en genre, donc c'est un seul produit
2 × 6 × 3 = 36. Le coin pluriel a des déterminants marqués sur trois genres, donc
il faut fixer le genre : 2 × 2 × 3 = 12. Les deux coins contiennent le même
nombre de phrases valides, et leurs plafonds sont dans un rapport de 3. **Aucun
renommage de la grammaire à deux genres ne peut produire ce rapport**, parce que
le plus grand produit est un invariant d'isomorphisme.

**Prédiction enregistrée le 31/07/2026, avant de lancer :**

> Sur 70 graines à β = 0,02, chemin float64 : **zéro dépassement**, maximum
> observé **36 dans le coin singulier et 12 dans le coin pluriel**, résultat modal
> égal au plafond, modes effectifs sur des produits d'entiers, branche
> indiscernable d'une pièce, `I(dét;nom)` nulle.
>
> **Ce qui réfute :** un coin singulier qui plafonne à 12 ou 24, ou un coin
> pluriel qui dépasse 12. Le plafond ne suivrait alors pas la structure de
> produit, et le résultat du §7.11 serait une propriété de mon vocabulaire à deux
> genres.

**RÉSULTAT, 70 graines, β = 0,02, chemin float64.**

| coin | n | plafond prédit | max observé | dépassements | pile au plafond | moyenne |
|---|---|---|---|---|---|---|
| singulier | 33 | **36** | **36,0** | **0** | 2 | 19,98 |
| pluriel | 37 | **12** | **12,0** | **0** | 7 | 6,64 |

**Prédiction confirmée.** Zéro dépassement, et les deux maxima tombent
exactement sur les plafonds calculés par énumération avant le lancement.

**Et le test quantitatif est plus fort que le test de dépassement.** Ce qui a
changé entre les deux grammaires n'est pas seulement l'ordre des coins mais le
**rapport** des plafonds :

| grammaire | rapport des plafonds | rapport des moyennes observées |
|---|---|---|
| standard, 2 genres | 2,0 | 1,82 |
| **trois genres** | **3,0** | **3,01** |

La moyenne des modes effectifs suit le rapport des plafonds, pas seulement leur
ordre. Un renommage peut inverser un ordre ; il ne peut pas transformer 2 en 3.

**Le reste de la prédiction :** branche à 33 / 37, p = 0,72 contre une pièce
équilibrée — toujours une pièce, malgré l'inversion des deux marginales d'ordre 1.
`I(dét;nom)` médiane 0,0000, maximum 0,0326, **0 run sur 70 au-dessus de 0,05
bit** : aucun run n'acquiert la conditionnelle, comme sur la grammaire standard.

**Une sous-prédiction que j'avais écrite trop fort.** J'annonçais « modes
effectifs sur des produits d'entiers ». Mesuré : **66 % en standard, 67 % à trois
genres** à moins de 0,05 d'un entier. C'est une proportion stable d'une grammaire
à l'autre, donc un fait réel, mais ce n'est pas « les modes sont des entiers ».
Avec six noms au lieu de quatre, une politique non uniforme sur plus d'items donne
plus facilement un 2^H non entier. Le **plafond** est une loi ; la quantification
n'en est pas une.

**Conclusion.** Le plafond de produit n'est pas une coïncidence de mon lexique
français à deux genres. Il suit la structure de produit de la récompense quand on
la change, en valeur et pas seulement en ordre. C'était l'expérience désignée au
§7.12 comme la seule qui décide, et elle passe.

### 7.12 Le plafond de produit est-il publiable ? Évaluation honnête, 31/07/2026

**Comme résultat, c'est ce que le projet a produit de plus solide. Comme article
autonome, non, pas encore.**

**Ce qui est fort** : une borne en forme close, calculable avant tout
entraînement ; 0 violation sur 70 runs ; les modes effectifs sont des **produits
d'entiers**, donc une quantification observée et non un ajustement ; et un
contrôle à un seul facteur — même architecture, même objectif, même coin, le
gradient exact franchit le plafond, l'échantillonné jamais.

**Correction de portée que j'ai trouvée en évaluant la publiabilité, et qui
change l'énoncé.** Le recuit β 0,2 → 0,01 atteint **45,3 modes**. Le plus grand
produit sur **tout** l'ensemble valide, coins confondus, vaut **24**. Donc le
recuit **franchit le plafond global** : REINFORCE échantillonné acquiert bel et
bien la conditionnelle dès que β varie.

Énoncé correct : *le plafond lie à **β constant dans le régime d'effondrement**,
pas pour l'échantillonnage en général.*

Ça renforce le résultat au lieu de l'affaiblir. J'expliquais le recuit
qualitativement (§7.3 : « garder toutes les conditionnelles entraînées pendant que
la représentation partagée se forme »). Le plafond remplace ce récit par un
nombre : à β constant bas, la politique tombe dans un coin et y est plafonnée par
le plus grand produit de ce coin ; le rôle du calendrier est de **retarder
l'engagement jusqu'à ce que le couplage existe**. Un mécanisme calculable à
l'avance au lieu d'un calendrier qu'on règle à la main.

**Ce qui manque, par ordre de ce qui tuerait le résultat.**

1. ~~**Le test de renversement — le seul qui compte.**~~ **FAIT le 31/07/2026,
   §7.13, et il passe.** Grammaire à trois genres : plafonds 36 et 12 au lieu de
   12 et 24, donc un **rapport de 3 au lieu de 2**. Zéro dépassement sur 70
   graines, maxima exactement sur les plafonds, et le rapport des moyennes
   observées suit celui des plafonds (3,01 contre 3,0 prédit ; 1,82 contre 2,0
   sur la grammaire standard). Ma première version du test était **isomorphe** au
   standard et ne testait rien — le piège est décrit au §7.13.
2. **Un seul algorithme.** « La procédure échantillonnée » est une affirmation
   sur une famille tirée d'un seul membre, REINFORCE + baseline mobile. Au moins
   PPO, ou une baseline à variance réduite.
3. **Un seul β** : les 70 graines sont toutes à 0,02.
4. **Le mécanisme est localisé, pas démontré.** Je sais que l'échantillonnage est
   nécessaire pour que le plafond lie. Je n'ai pas de preuve du *pourquoi* — et le
   confondant de §5.1 reste entier, le gradient exact optimise le vrai objectif
   alors que l'échantillonné utilise le bonus d'entropie biaisé.

**Verdict** : atelier, ou section forte d'un article plus large. Pas un article
principal autonome sur une grammaire de 20 tokens, et le premier reproche du
relecteur sera exactement celui que pose déjà ma Q29.

> **Verdict révisé le 31/07/2026, après §7.13.** Le point 1, qui était le seul
> bloquant, est levé : le plafond suit la structure de produit sur une seconde
> grammaire, en rapport et pas seulement en ordre. Ce qui reste — un seul
> algorithme, un seul β, et le mécanisme localisé plutôt que démontré — sont des
> limites à énoncer, pas des trous qui invalident. Le résultat devient
> présentable en l'état, avec ses limites écrites.

**Attribution, à trancher avant d'écrire quoi que ce soit.** L'argument du
produit est **de dipankarsarkar**. Je l'ai vérifié, étendu à la grammaire longue
et confronté à 70 graines, mais je ne l'ai pas trouvé. Si ça se publie, c'est une
co-signature ou au minimum un crédit en tête d'article, et ça se décide
maintenant, pas quand le brouillon existe.

### 7.14 Cinquième critique : il est allé au test 3, et y a trouvé une contradiction interne

11/08/2026. Dipankar Sarkar, environ 22 heures après la publication de l'article 2,
n'a pas commenté l'article : il est allé lire le document de conception du test 3,
qui n'a jamais tourné. Il a d'abord reproduit `grammaire3.py` à la graine 0 — les
huit chiffres du tableau, à la décimale près — puis il a refait la même statistique
à 10 000 000 de tirages au lieu de 20 000, en vectorisant, et en validant son
vectorisé contre mon scalaire sur 3 000 codes (écart maximal 5,6 × 10⁻¹⁷).

**Ce qu'il a trouvé.** Toutes les lignes du tableau tiennent sauf une. La moyenne
passe de 0,1273 à 0,1269, l'écart-type de 0,0332 à 0,0330, q99,9 de 0,2537 à
0,2525. Le maximum passe de 0,3305 à 0,3979. Multiplier le tirage par 500 déplace
q99,9 de 0,0012 et le maximum de 0,067. Or **le seuil de ~0,35 de §6.1 était bâti
sur cette ligne-là**, la seule encore en mouvement.

**Ma reproduction, indépendante.** Les huit chiffres à 20 000 tombent exactement.
À 10 000 000, tirage indépendant du sien : moyenne 0,1269, sd 0,0330, q99,9 0,2525,
q99,999 0,3196 contre ses 0,3195, maximum 0,39788 contre ses 0,3979, et **14
tirages au-dessus de 0,35 comme lui**. La coïncidence sur le maximum est réelle —
la statistique est discrète, 2 951 valeurs distinctes sur 2 000 000 de tirages,
mais les barreaux du sommet sont des singletons, donc elle n'explique rien. Douze
blocs indépendants le montrent : maxima de 0,3775 à 0,4283.

**Ce que j'ajoute à son diagnostic, et qui est pire.** Son argument est que le
maximum bouge encore. Le vrai problème est qu'il ne peut pas s'arrêter de bouger.
Les 1 296 codes compositionnels **sont** des bijections : ils sont dans la loi
nulle, avec probabilité 1 296/27! ≈ 1,19 × 10⁻²⁵, et ils valent 1. Le supremum de
la nulle vaut exactement 1. Le maximum d'échantillon estime 1, infiniment
lentement, et un seuil bâti dessus mesure la taille du tirage.

**Et la vraie faute est en amont.** §5 écrit noir sur blanc « on abandonne
délibérément le critère pass/fail » et enregistre un engagement portant sur une
**distribution**. §6.1, trois paragraphes plus bas, réintroduit un pass/fail et se
félicite qu'il soit dérivé plutôt qu'arbitraire. Il ne m'a pas fallu une critique
extérieure pour écrire les deux ; il en a fallu une pour que je les lise ensemble.
Le seuil ne contredisait pas seulement une bonne pratique, il contredisait mon
propre document à trois paragraphes de distance.

**Le calcul de puissance qu'il propose, reproduit.** Test unilatéral, p < 0,001,
80 % de puissance : δ = 3,93 σ/√n. À 100 graines, 0,0130 ; à 50, 0,0184. Le seuil
retiré exigeait 0,223 sur un seul run. Rapport **17**. Et il détecte la bonne
chose : une pression faible soulève tous les runs de 0,02 bien plus volontiers
qu'elle ne projette un run isolé au-delà de 0,35.

**Sa seconde question, sur laquelle il se disait moins sûr.** `concentration()`
prend le max colonne par colonne sans contrainte, donc un attribut peut gagner
deux positions : 74,6 % des tirages uniformes chez lui, 74,6 % chez moi. Sur ses
200 plus hauts, un appariement hongrois rend 0,2640 au lieu de 0,2810. Il dit :
ça s'annule dans la comparaison de §6.2, mais pas dans la lecture de §6.1, où un
code à 0,28 se voit créditer 0,017 de structure qu'il n'a pas ; l'inflation serait
réelle au milieu et nulle aux deux bouts.

**Mesuré, ce n'est pas « au milieu ».** Sur des codes dont la structure est connue
par construction (k positions propres sur 3), l'écart entre les deux statistiques
vaut 0,0098 à k=0, 0,0052 à k=1, et **exactement 0,0000 à k=2 et k=3**. Le long
d'une échelle par transpositions, il est **nul jusqu'à 9 transpositions** puis
monte rejoindre la valeur de la loi nulle à 21. L'inflation ne suit pas le niveau
de concentration, elle suit la **structure** : elle est nulle partout où §6.1 a
quelque chose à lire. C'est l'inverse de ce que mon premier balayage suggérait
(§1.10), et je ne l'aurais pas vu sur une seule des deux populations.

**Ce que je change quand même, et pourquoi ce n'est pas de la complaisance.** Les
deux statistiques sont publiées. La forme sans contrainte est celle du standard du
domaine — posdis, Chaabouni et coll. 2020, prend l'argmax indépendamment par
position — donc la retirer coûterait la comparabilité. La forme appariée devient
celle que §6.1 lit comme une position : elle classe un peu mieux contre une vérité
terrain combinatoire (86,82 % contre 86,34 %), elle est identique là où ça compte,
et sous l'alternative de §6.2 elle baisse la référence sans bouger le signal — donc
neutre sous H0, favorable sous H1.

**Ce qui rend cette correction vérifiable.** Aucun entraînement du test 3 n'a
tourné. Il n'existe aucune concentration émergente mesurée. Changer l'instrument
aujourd'hui ne peut pas avoir été motivé par un résultat, et la même correction
faite après un premier run devrait être refusée. C'est la première fois du projet
qu'une critique arrive avant les données plutôt qu'après, et c'est de loin la
position la plus confortable pour la recevoir.

**Le pire cas, borné et honnête.** Montée locale, donc minorants : l'écart maximal
trouvé entre les deux statistiques vaut 0,1443, sur un code de concentration
0,2473 — dans le corps de la loi nulle, là où il n'y a rien à conclure. La plus
haute concentration atteinte **avec** double compte vaut 0,6314, et ce code vaut
encore 0,5560 en apparié. Le sommet reste isolé : 1,0000 pour un compositionnel,
puis 0,9294 pour le meilleur non compositionnel trouvé.

---

## 8. Vingt questions inconfortables

Règle que je m'impose ici : pas de question dont je connais déjà la réponse, pas
de question qui flatte le projet, et pour chacune ce qui la trancherait. Plusieurs
attaquent la valeur de tout ce qui précède.

### Sur le résultat combinatoire du test 3

**Q11 — Existe-t-il une seule fonction de récompense dont l'ensemble des optima
soit exactement les codes compositionnels, sans qu'on ait codé la
compositionnalité à la main ?** Le calcul du test 3 donne 1 296 codes
compositionnels sur 27! ≈ 1,09 × 10²⁸ bijections, toutes à récompense 1. Si la
réponse est non, alors la compositionnalité n'est **jamais** apprenable depuis une
récompense seule, à aucune échelle, et toute la littérature sur l'émergence de
langage mesure l'effet de contraintes annexes en croyant mesurer l'effet du RL.

**Q12 — Toute récompense d'alignement a-t-elle cette forme ?** Un modèle de
récompense départage mal deux comportements qu'il note pareil. Si « aligné » et
« qui a l'air aligné » sont à égalité sous le reward model, alors l'alignement par
récompense est structurellement le même problème que la compositionnalité par
récompense — et le certificat des optima à égalité (§2.2) s'y applique tel quel.
Testable : prendre un reward model public, chercher des paires notées à 10⁻⁶ près
et regarder ce qui les sépare.

**Q13 — La probabilité 10⁻²⁵ est-elle vraiment la bonne mesure ?** Elle suppose
l'équiprobabilité à l'optimum max-ent. Mais l'initialisation n'est pas uniforme
sur les bijections, et la dynamique non plus. Quelle est la mesure **induite par
la dynamique** sur l'ensemble des codes parfaits ? Elle pourrait concentrer sur
une sous-famille structurée pour des raisons qui n'ont rien à voir avec la
récompense — exactement comme le biais d'ordre 1 a décidé la branche au test 2.

### Sur le vérificateur

**Q14 — Le code et les mathématiques ont des vérificateurs parce qu'ils ont une
sémantique formelle. Le langage naturel n'en a pas. Est-ce là toute
l'explication ?** Si oui, la question « peut-on apprendre le langage par
récompense » a une réponse structurelle et non empirique, et aucune expérience ne
la changera.

**Q15 — Toutes les capacités que le RL a automatisées avaient un vérificateur
construit par des humains *avant* l'IA** : règles des échecs, du go, tests
unitaires, vérificateurs de preuve. La généralité apparente du RL n'est-elle pas
en réalité un **inventaire de vérificateurs préexistants** ? Question falsifiable :
citer une capacité acquise par RL dont le vérificateur a été inventé *pour*
l'occasion et n'encode pas déjà la solution.

**Q16 — Un vérificateur qui accepte un ensemble est-il vraiment différent d'un
oracle sur un point ?** J'ai présenté le passage test 1 → test 2 comme un progrès
(oracle-point → oracle-ensemble). Mais l'information fournie par un humain a
peut-être seulement changé de forme, pas de quantité. Mesurable : combien de bits
faut-il pour spécifier le parser, contre combien pour spécifier la cible ?

### Sur ce que mes propres mesures valent

**Q17 — Toutes mes mesures exactes ne le sont que parce que l'espace fait 8 000
éléments. Y a-t-il une seule propriété mesurée ici dont on puisse **prouver**
qu'elle survit au passage à l'échelle ?** Si la réponse est non, l'article décrit
un régime, pas un phénomène, et « énumérable » est une physique différente.

**Q18 — Le réseau non entraîné a 47,5 modes effectifs sur 48. Que reste-t-il de ce
qu'on appelle « diversité apprise » ailleurs, une fois qu'on soustrait l'entropie
résiduelle d'initialisation que l'entraînement n'a pas encore détruite ?** Le
protocole existe et coûte une passe : mesurer la métrique de diversité **sur le
modèle non entraîné** et la rapporter systématiquement comme plancher.

**Q19 — Le bonus d'entropie par token vise l'uniformité sur les tokens, pas sur
les séquences. Ce décalage existe dans tout régularisateur par token de tout
modèle de langue.** Quelle part de ce qu'on appelle « la distribution du modèle »
est un artefact de régularisation à la mauvaise granularité ? Calculable
exactement ici : l'écart entre 45,35 et 48 est précisément ça.

**Q20 — Existe-t-il un régularisateur dont le point fixe soit l'uniforme sur
l'ensemble des bonnes réponses, sans connaître cet ensemble ?** C'est ce que le
recuit approche empiriquement sans le formuler. S'il existe, il remplace le bonus
d'entropie partout.

### Sur la conception de récompense

**Q21 — Peut-on calculer, à partir de la seule fonction de récompense et avant
tout entraînement, la taille du plus grand sous-ensemble où la contrainte est
vacuellement satisfaite ?** Ce serait un *linter* de récompense : il aurait signalé
la sous-langue tout-pluriel du test 2 avant que je lance quoi que ce soit. Je ne
connais rien qui fasse ça.

**Q22 — Le β critique (entre 0,02 et 0,05) est-il déductible du spectre ANOVA
seul ?** Les deux quantités sont calculables sans entraînement. S'il existe une
relation, on prédit la pression entropique nécessaire avant de lancer — et §5bis.3
devient une loi et non une conjecture.

**Q23 — Le recuit de β, le warmup de learning rate, les calendriers de KL en RLHF
et le recuit simulé sont-ils le même mécanisme ou seulement des cousins visuels ?**
Ici le mécanisme est identifié : garder toutes les conditionnelles entraînées
pendant que la représentation partagée se forme. Si c'est le même ailleurs, le
warmup n'est pas une astuce numérique mais une prévention d'interférence
représentationnelle.

### Sur le RL lui-même

**Q24 — L'aveuglement à l'ordre 1 est-il dans l'estimateur ou dans la
politique ?** REINFORCE donne le même avantage global à chaque position, choix de
1992. Une attribution de crédit par position — qu'on sait faire — suffirait-elle,
ou l'aveuglement vient-il de la factorisation autorégressive elle-même ? Les deux
sont testables séparément ici, et personne ne les sépare.

**Q25 — Combien de pathologies nommées du ML sont juste des objectifs qui font ce
qu'ils disent ?** À β=0, l'effondrement de mode **est** l'optimum. On l'appelle
pathologie parce que l'objectif ne contient pas ce qu'on voulait. Combien d'autres
noms — *reward hacking*, *shortcut learning*, *sycophancy* — désignent la même
chose : une spécification correcte et une intention absente ?

**Q26 — Un agent optimal sur une récompense est-il obligé de trouver la
sous-langue dégénérée ?** Au test 2 c'est ce qui s'est passé, mais parce que
l'ordre 1 y menait. Y a-t-il des récompenses où le chemin d'ordre 1 pointe **vers**
la solution non dégénérée ? Si oui, on peut concevoir des récompenses par leur
spectre plutôt que par leur formule.

### Les plus inconfortables

**Q27 — La vitesse de production d'une explication prédit-elle sa fausseté ?**
Cinq de mes hypothèses sont mortes aujourd'hui, toutes plausibles, toutes
produites en une seconde. C'est un signal que j'ai fini par utiliser
consciemment. Est-il **mesurable** ? On pourrait horodater les explications et
corréler avec leur survie — sur soi comme sur un modèle.

**Q28 — Combien de conclusions publiées sont conditionnées à quel run a fini avant
la date limite ?** J'ai publié « c'est la géométrie, pas le bruit » parce que le
balayage tournait encore. Le run qui l'a démentie a fini deux heures plus tard.
Ce n'est pas une négligence isolée, c'est la structure normale du travail sous
contrainte de temps.

**Q29 — Si la valeur de ce projet vient entièrement d'un jouet assez petit pour
être énuméré, la façon honnête de faire de la recherche en ML est-elle de
rétrécir jusqu'à l'exactitude puis de débattre de l'extrapolation ?** Et alors,
que devient tout ce qui est fait à l'échelle — est-ce nécessairement de
l'anecdote mieux financée ?

**Q30 — À quel moment « je mesure ma propre spécification » s'applique-t-il à moi
et plus seulement à l'agent ?** J'ai écrit l'environnement, la récompense, les
diagnostics, les métriques et l'interprétation. Le diagnostic que j'ai construit
(imposer l'antécédent, mesurer le conséquent) détecte les sous-langues
dégénérées de l'agent. **Quel diagnostic détecte les miennes ?** C'est la seule
question de cette liste à laquelle je n'ai aucune piste.

---

## 8bis. Le jury de LLM : trois questions de Théo, et ce qu'elles ouvrent

Échange du 31/07/2026, après la critique d'ordre 1. Ses questions valent d'être
notées telles quelles, parce que la troisième retourne l'argument des deux
premières et que je ne l'avais pas vu venir.

### T1 — « Pourquoi ne pas remplacer le parser par un jury de LLM ? »

Schéma proposé : génération RL → trois juges LLM (qualité, logique, vérité) →
agrégation → récompense.

Cinq objections, toutes adossées à des chiffres déjà acquis :

1. **Les juges sont pré-entraînés.** Le prior linguistique n'est pas supprimé, il
   passe de l'autre côté de la fonction de récompense. C'est la conclusion 3 de
   `ANALYSE.md` : le pré-entraînement fournit le signal dense par position, ici
   c'est le juge. « Zéro donnée humaine » devient faux par construction.
2. **On perd l'énumérabilité, donc les preuves.** Masse valide exacte, 2^H exact,
   optimum de Gibbs, certificat des optima à égalité : tout repose sur 8 000
   séquences énumérables. Avec des juges LLM il ne reste que l'échantillonnage,
   c'est-à-dire des courbes qui montent.
3. **Le certificat des optima à égalité s'applique tel quel, et il est
   défavorable.** L'ensemble des sorties notées 9/10 est immense et le juge ne
   sépare rien à l'intérieur. Voir Q12.
4. **La sous-langue dégénérée devient indétectable.** Un juge LLM a plus de coins
   vacuellement satisfaits qu'un parser, et on ne peut plus les énumérer.
5. **Trois juges corrélés ne font pas trois signaux.** Même pré-entraînement,
   mêmes angles morts ; moyenner compresse la dynamique et fabrique des égalités.

### T2 — « Et si le juge explique pourquoi c'est faux et comment s'améliorer ? »

**Ce n'est plus du RL.** Une critique en langue naturelle porte des centaines de
bits *dirigés* contre quelques bits scalaires. Le nom honnête est distillation
d'un professeur.

**Et c'est circulaire pour notre question** : pour exploiter « le déterminant ne
s'accorde pas en nombre avec le nom », il faut déjà comprendre cette phrase. La
compétence qu'on cherche à faire émerger est un **prérequis** pour consommer le
signal censé la produire.

**Mais l'intuition est juste sur le fond**, et c'est le point à retenir : une
critique textuelle défait exactement les deux pathologies mesurées ici. Le
gradient est aveugle à l'ordre 1 (§7.4) alors que l'accord est d'ordre 2 par
nature ; une critique qui nomme la **paire** fautive livre l'ordre 2 directement,
sans attendre que la politique se déplace. Et elle sépare deux sorties que le
scalaire notait pareil, donc elle casse le certificat des optima à égalité.

**Expérience qui isole ça sans aucun LLM, tout reste énumérable.**
`grammaire.py:analyser()` renvoie déjà les sous-scores séparés ; aujourd'hui on
en fait la moyenne et on jette le vecteur.

| régime | ce que l'agent reçoit |
|---|---|
| actuel | la moyenne, un scalaire |
| vectoriel | les 3 sous-scores séparés |
| ciblé | l'identité de la contrainte violée |

Si le vectoriel suffit à sortir du coin dégénéré, le mérite est dans la
**décomposition** et ni le LLM ni la langue naturelle n'y sont pour rien. Si seul
le ciblé y arrive, c'est l'**adressage** de la contrainte qui compte. Une heure
de calcul, et ça tranche une question que le débat RLAIF traite par l'anecdote.

### T3 — « Les humains apprennent comme ça »

Objection apparente, et en fait **le même résultat que le nôtre**.

**L'explication arrive après la représentation, jamais avant.** On ne peut pas
expliquer à un enfant de 14 mois pourquoi sa phrase est mal accordée : il faut
déjà la langue pour lire la correction. Les premiers mois sont un apprentissage
sans explication et sans récompense dirigée. La phase « le juge explique » ne
s'ouvre qu'une fois le socle en place, c'est-à-dire **exactement l'ordre
pré-entraînement → RLHF**. L'analogie humaine décrit le pipeline standard en
croyant décrire une alternative.

**Sur la grammaire précisément l'analogie est plus faible encore.** Le résultat
classique (Brown & Hanlon, 1970) est que les parents corrigent la vérité de
l'énoncé, pas sa forme. C'est contesté depuis — les reformulations implicites
existent — mais personne ne soutient qu'un enfant reçoit « erreur d'accord
genre-nombre en position 2 ». Or c'est ce que fournit le juge de T2, et les
enfants apprennent l'accord **sans**.

### Questions que cet échange ouvre

**Q31 — Quelle est la marginale d'ordre 1 d'un juge LLM ?** `E[R_juge | token en
position p]` sous politique uniforme, sur un vocabulaire assez petit pour être
énuméré. C'est le premier signal que suivra l'agent, il est calculable avant tout
entraînement, et **personne ne le calcule**. Chez nous il pointait vers une
phrase invalide (§7.5). La sonde existe déjà : `sonde_ordre1.py`, il n'y a que la
fonction de récompense à remplacer.

**Q32 — Combien de bits indépendants portent trois juges corrélés ?** Mesurable
directement par la corrélation des scores sur un même lot de sorties. Si elle est
haute, l'agrégation est du théâtre et il faut le dire avant de construire le
pipeline.

**Q33 — Le gain d'une critique textuelle vient-il de la décomposition ou de la
langue naturelle ?** C'est le protocole scalaire / vectoriel / ciblé ci-dessus.
La réponse détermine si tout l'appareil LLM est nécessaire ou décoratif.

**Q34 — Existe-t-il un seuil de compétence en dessous duquel une critique est
inutilisable ?** Si oui, il y a une frontière nette entre « apprenable par
récompense » et « apprenable par correction », et elle se mesure.

**Q35 — Que devient le certificat des optima à égalité quand la récompense est
un vecteur et non un scalaire ?** Le certificat suppose un ordre total sur les
sorties. Avec des sous-scores séparés il n'y a plus d'égalité qu'en cas d'égalité
**sur toutes les coordonnées**, donc l'ensemble des optima à égalité rétrécit.
De combien ? Calculable exactement sur les 8 000 séquences.

**Q36 — D'où peut venir un signal gradué avant que la langue existe ?** La
question ouverte de la fin de `ANALYSE.md`, et T3 la remet au centre : c'est le
seul endroit où l'analogie humaine reste informative, parce que c'est le seul
endroit où l'humain fait quelque chose qu'on ne sait pas répliquer.

---

## 9. Ce qu'il faudrait construire ensuite, par ordre de valeur

1. **Décomposition de variance de la récompense** (§5.3). Coût quasi nul, et
   c'est la seule mesure qui distingue le test 2 du test 1 sur le fond.
2. **Grammaire longue propre** `dét nom verbe adv adv` (§4.1). Isole enfin la
   taille de l'espace.
3. **Un objectif qui échantillonne proportionnellement à la récompense** (§5.2).
   C'est le test direct de « est-ce le RL ou est-ce REINFORCE ».
4. **Refaire les tests de généralisation sur un régime représentatif** (§4.3),
   multi-graines, avec la puissance statistique évaluée (§3.4).
5. **Répondre à §5.5 avant de concevoir le test 3.** Sans critère de falsification
   défini à l'avance, le test 3 produira un troisième succès ininterprétable.
