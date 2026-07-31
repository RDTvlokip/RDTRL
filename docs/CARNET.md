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

**Le champ qui tranchait était déjà calculé et jeté.** `analyse_exacte` renvoie
`entropie_nom_sachant_det` avec `H_max = log2(noms_compatibles)`, soit 1 bit pour
un déterminant singulier et 2 pour un pluriel. `balayage_graines.py` sauvegarde
`moyenne_cond_det` à la place. Troisième fois dans ce projet qu'une mesure
décisive existe déjà et n'est pas regardée, après la sonde d'ordre 1 (§7.10) et le
balayage multi-graines (§7.5).

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
