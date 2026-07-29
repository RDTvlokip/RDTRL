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

### 1.5 Correction rétroactive au test 1

J'avais écrit : « le blocage est sur l'obtention du signal, jamais sur
l'optimisation ». **Faux dès qu'il existe plusieurs solutions.** C'était vrai
pour une cible unique, où il n'y a rien à répartir. Le test 2 montre une
optimisation qui échoue alors que le signal est parfait.

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

### 3.3 Ma propre métrique de saturation est mal étiquetée

Dans le tableau H(nom | déterminant), la colonne `satur.%` dépasse 100 %
(`la` : 218 %). Cause : H est calculée sur les 8 noms alors que H_max utilise le
nombre de noms *compatibles*. Une valeur > 100 % signale donc une **fuite de
masse sur des noms incompatibles**, c'est-à-dire un échec — pas une
sur-saturation. À renommer ou à normaliser.

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
interactions (décomposition de type Sobol / ANOVA fonctionnelle). Si l'agent
apprend la part d'interaction, ce serait le premier résultat du projet qui ne se
réduit pas à « la récompense a fait le travail ». **Non fait.**

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

## 6. Ce qu'il faudrait construire ensuite, par ordre de valeur

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
