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

### 1.11 « Le certificat des optima à égalité s'applique tel quel à deux agents » — morte le 11/08/2026

Écrite en §3 de TEST3.md, et elle porte tout le calcul des 1,19 × 10⁻²⁵. Le
certificat exige que **les objets à égalité soient le support de la loi dont
l'entropie figure dans l'objectif**. Au test 2 c'était le cas — objets à égalité :
des séquences ; entropie : celle de la loi des séquences. Au test 3 les objets à
égalité sont des **codes**, et aucune loi sur les codes n'apparaît dans l'objectif.
La récompense étant de coordination, étaler l'émetteur sur K codes fait chuter
`E[R]` comme 1/K : mesuré 1,0000 / 0,5000 / 0,3416 / 0,2237 / 0,1511 / 0,0713 pour
K = 1, 2, 3, 5, 10, 27. Le chiffre survit par un argument de symétrie, qui est plus
fort mais plus étroit. Détail en §7.15.

### 1.12 « L'écart entre le seuil mesuré et 1/27 vient de la taille de la perturbation » — morte le 11/08/2026

La bissection sur la montée donnait β = 0,0381 contre 1/27 = 0,0370 prédit. Mon
explication : la perturbation vaut 10⁻³, pas un infinitésimal, donc elle franchit
une barrière peu profonde. Testée en réduisant le bruit de 10⁻² à 10⁻⁵ : 0,0383 ·
0,0381 · 0,0382 · 0,0375. L'écart ne se referme pas, et la suite n'est même pas
monotone. La vraie cause est **Adam** : ses pas sont normalisés, donc il ne
ralentit pas là où le gradient s'annule et quitte un maximum local que l'objectif
tient pour stable. Tranché sans aucune dynamique, par le hessien au point de babil,
dont la plus grande valeur propre croise zéro en **0,037037037** — à 3,4 × 10⁻¹² de
1/27. La prédiction était exacte ; c'est l'instrument de mesure qui mesurait autre
chose.

### 1.13 « Le sommet de l'échelle de concentration est sûr » — morte le 11/08/2026

Argument de Dipankar Sarkar, que j'avais vérifié et repris à mon compte le matin
même : une concentration de 1 force chaque colonne à déterminer entièrement un
attribut, et deux positions déterminant le **même** attribut effondreraient neuf
référents sur trois messages, ce qu'une bijection ne peut pas faire. L'argument est
juste. **Sa prémisse tombe** : §6.5 mesure que les codes atteints ont 1 à 4
collisions, donc ne sont pas des bijections.

Contre-exemple explicite, `m₁ = a₁`, `m₂ = a₁`, `m₃ = a₂` — le premier attribut
dupliqué sur deux positions, le troisième jeté, 9 messages utilisés sur 27 :
**concentration max = 1,000000**, concentration appariée = 0,666667. La statistique
publiée décerne le sommet réservé aux codes compositionnels à un code qui perd un
attribut sur trois.

Conséquence : le double compte n'est pas un défaut du milieu de l'échelle réservé
aux codes sans structure, comme la mesure sur les bijections nous l'avait fait
croire à tous les deux. Hors des bijections, **il atteint le sommet**. La version
appariée cesse d'être une amélioration marginale de 0,48 point de concordance pour
devenir la seule interprétable. Et les trois bornes de §7.14 — 0,1443, 0,6314,
0,9294 — restent vraies mais **conditionnellement à la bijectivité**, la montée
locale ayant été faite sur des permutations. Corrigées depuis, et dans l'autre
sens que je croyais : voir §7.24 et §7.25.

### 1.14 « La paramétrisation structurée préfère le compositionnel dès le premier pas » — morte le 11/08/2026

Écrite avant mesure dans `gradient_premier_pas.py`, et elle semblait sûre : cette
paramétrisation finit à z = +9,9 en §6.1, donc son biais devait être visible dans
son gradient initial. Mesuré, cosinus entre ∇J et ∇L(compositionnel) contre 300
bijections témoins : **z = −0,08 ± 0,24**. Rigoureusement rien.

Le mécanisme, une fois cherché : près de l'uniforme, la contrainte de la
paramétrisation **ne mord pas**, puisque toute loi est représentable à faible
confiance. Elle n'apparaît qu'à mesure que la loi se concentre. La courbe le
montre — z passe de −1,18 au pas 0 à **+4,36 au pas 30**. La préférence est donc
**amorcée par la trajectoire en quelques dizaines de pas**, pas présente au départ.
Détail en §7.21.

*(Corrigé le 18/08/2026, §7.35. Cette phrase disait « et n'en bouge plus ». Faux :
la courbe complète, dans le même dictionnaire de `6_4_gradient_premier_pas`, vaut
+4,36 au pas 30, +4,25 à 100, **+3,91 à 300**, **+5,81 à 1000**, **+5,85 à 3000** —
une hausse de 34 % au-delà du point où j'annonçais l'arrêt, et elle monte encore au
dernier pas mesuré. L'amorce est juste, l'achèvement est faux : la préférence
continue d'être construite pendant trois mille pas. J'avais cité une clé du fichier
et décrit ses voisines sans les lire.)*

### 1.15 « Le bruit de canal favorise les codes compositionnels » — morte le 11/08/2026

Écrite dans la table de TEST3.md §6.6 comme justification du bouton le plus
prometteur : « un code compositionnel ne perd qu'un attribut quand un token est
corrompu ; un code holistique perd tout ». Vrai en information, **sans aucun effet
sur cette récompense**, et ça se démontre en une ligne sans entraîner quoi que ce
soit. Pour un émetteur déterministe sur un code `c` et le décodeur optimal,
`E[R]* = (1/27) Σ_m' max_r C[c(r), m']`, et `c` étant une bijection sur les 27
messages, `max_r C[c(r), m'] = max_m C[m, m']` — **indépendant de `c`**. Mesuré :
écart compositionnel/aléatoire ≤ 1,1 × 10⁻¹⁶ à tout ε de 0 à 0,8.

Perdre un seul attribut ne rapporte rien quand le crédit est tout-ou-rien sur le
référent exact. Sous une récompense à **crédit partiel par attribut**, l'égalité se
brise (+0,108 à ε = 0,2) — mais ça met la compositionnalité dans la spécification.
Détail en §7.22.

### 1.16 « Briser la symétrie suffit à produire de la compositionnalité » — morte le 11/08/2026

Hypothèse unificatrice que j'avais tirée de §6.7 le matin même, et qui remplaçait
la liste de recettes de §6.6 par une question unique. Elle est fausse sur deux
points, mesurés le soir. Le renouvellement de population **ne brise pas** `S₂₇` du
tout, étant une opération échangeable — donc le théorème s'applique encore, et z
reste nul à toutes les périodes, comme prédit. Et le bruit de canal, lui, **brise
bien** la symétrie (écart 0,00e+00 sur le groupe structurel, ≥ 0,050 sur 200
permutations quelconques) sans rien produire : z de −0,44 à +0,38 sur six valeurs
d'ε, aucun run sur 90 au-delà du quantile 99,9 %.

**Briser la symétrie est nécessaire, pas suffisant.** Détail en §7.22.

### 1.17 « L'écart max/appariée dépend de R » — morte le 12/08/2026

Implicite dans ma façon de lire le tableau du pire cas par plancher R, qui est
fortement monotone : 0,0526 à R = 27 contre 0,2152 à R = 23. J'ai laissé croire que
l'écart **observé** suivait le même index. Mesuré sur 210 runs : corrélation
+0,09 linéaire, η² = 1,2 %, F(6,203) = 0,419 à **p = 0,87**. Le pire cas
atteignable par une recherche et l'écart produit par la dynamique sont **deux
fonctions différentes de R**, la première monotone et la seconde plate, dans un
rapport de 15. Détail en §7.25 et §7.25bis.

### 1.18 « La ligne beta ne porte aucune tendance au-delà du bruit » — morte le 14/08/2026

Écrite au septième tour, sans test, pour justifier de ne pas m'y arrêter. La ligne
portait **trois contrastes au-delà de deux sigma**, dont le plus grand de tout le
tableau : beta = 0,005 contre beta = 0,03, t = −2,968, plus grand que le contraste
en R (t = 2,430) sur lequel deux tours de relecture ont porté. Son omnibus valait
F(4,145) = 2,595 à p = 0,039.

Ce qui rend cette mort particulière : la conclusion était **juste**. Les soixante
runs indépendants font passer le contraste de −0,00981 à −0,00135 (t = −0,34) et
l'omnibus à F(4,55) = 1,790, p = 0,144. La ligne est bien plate. Mais je l'avais
affirmé sans mesure, et **avoir raison en ne regardant pas n'est pas avoir raison**.
La faute n'est pas dans la conclusion, elle est dans le fait que la même phrase
aurait été écrite si la ligne avait porté un effet.

Conséquence de méthode : le jeu de sélection d'un tableau inclut les lignes qu'on a
regardées et pas rapportées. Vingt contrastes et non dix, ce qui fait passer le p
corrigé du contraste en R de 0,101 à 0,200. Règle en §7.26.

### 1.19 « R est un facteur de ce plan » — morte le 15/08/2026

Présupposé de tout §7.25, §7.25bis et §7.26 : j'ai lu, corrigé, répliqué et
re-corrigé des contrastes sur une ligne indexée par R, en discutant pendant trois
tours de leur multiplicité et de leur réplication. Aucun de ces trois tours n'a
demandé si la colonne était un facteur.

Elle ne l'est pas. Le fichier de résultats porte la récompense finale de chaque run,
que personne n'avait ouverte : **|récompense − k/27| < 10⁻³ pour un entier k dans 150
cas sur 150**, avec k = R dans 141 et k = R − 1 dans 9. corr(R, récompense) = +0,9725.
R est l'objectif du run, arrondi à une grille de 1/27. Stratifier par R revient à
trier les runs par le score qu'ils ont atteint, puis à demander si un biais de mesure
diffère entre ceux qui ont fait 25/27 et ceux qui ont fait 24/27.

**Ce qui rend cette mort plus coûteuse que les précédentes :** la ligne était
disqualifiée avant l'arrivée des données, par lecture du générateur, et trois tours
de correction statistique de plus en plus fine n'ont pas pu s'en apercevoir parce
qu'ils tarifaient tous un contraste au lieu de demander si la colonne méritait un
tarif. Le bump d'une cellule relevé au dixième tour n'est pas la raison : la ligne
n'aurait pas dû être publiée même si toutes ses cellules avaient été plates.

Règles 5 et 6 en §7.27. La ligne beta, elle, survit : beta est réglé avant le run,
et χ²(16) = 15,67 à p = 0,476 montre qu'il ne déplace pas R.

### 1.28 « La dynamique choisit parmi les optima liés » — morte le 18/08/2026

Prémisse de tout le test 3, écrite dans le document de conception et reprise comme
arithmétique de tête de l'article : les 27! bijections valent toutes récompense 1,
1296 sont compositionnelles, donc un résultat compositionnel ne peut pas s'expliquer
par la récompense (1,19 × 10⁻²⁵ sous tirage uniforme).

**La dynamique n'entre jamais dans cet ensemble.** Sur 1200 runs, le bras tabulaire
finit bijectif **60 fois (5,0 %)** et le factorisé **1 fois (0,1 %)**. Le reste
converge à récompense 0,93 avec ~1,8 collisions — et les 1296 codes compositionnels
étant tous bijectifs, un run à collisions **ne peut pas** être compositionnel.

Et ce n'est pas une troncature. La montée converge : 0,92896 à 3000 pas, 0,92901 à
12 000, 0,92901 à 30 000. Elle converge vers un point **strictement pire de son propre
objectif** : J = 0,96395 depuis l'aléatoire contre **J = 1,00000** à l'état
compositionnel ajusté puis remonté. Écart +0,03605. Le paysage a des optima locaux et
Adam tombe dedans 95 % du temps.

**Le point critique est réel.** Le gradient de J tombe à 3,65 × 10⁻⁷ (relatif
1,1 × 10⁻⁹) au plateau, et 20 000 pas de SGD n'en sortent pas.

*(Mais j'en avais conclu « aucune méthode locale n'y échappe », **faux, réfuté le
19/08 en §7.36** : REINFORCE en sort 11 fois sur 12. Le 5 % est une propriété de la
montée exacte, pas du banc.)*

**Conséquence sur l'énoncé publié**, dans sa forme corrigée : *sous montée exacte,
l'objectif a des points critiques en k/27 qui piègent le flot 95 % du temps, donc
tout ce que §6.1 à §6.7 mesure porte sur des codes hors de l'ensemble lié.* La
question du plan reste posable — elle l'est même dans 92 % des runs sous REINFORCE —
et elle n'a jamais été posée. La question du plan — que trancherait la récompense parmi des
optima liés — n'a été posée que sur 60 runs, avec 0 compositionnel et une borne
supérieure de 6,0 % contre un nul de 1,19 × 10⁻²⁵ : vingt-quatre ordres de grandeur de
jeu, donc aucune puissance.

Survivent intacts : le no-go d'équivariance de §6.7, propriété de l'objectif ; et
l'uniformité intra-classe de fibres, bien mesurée sur sa population. Détail en
§7.35ter.

*(Correction du 19/08, §7.36 : le « 95 % » est une propriété de la **montée exacte**.
Sous REINFORCE lot 64 à 20 000 pas, 92 % des runs atteignent une bijection —
11/12 contre 0/12 pour la montée exacte à budget, graines et lr identiques,
p = 9,6 × 10⁻⁶. La prémisse reste non testée, mais elle est **testable**, et le
premier item de la suite est de relancer §6.2 sous REINFORCE.)*

### 1.29 « Les points critiques sous-optimaux ne sont pas franchissables » — morte le 19/08/2026, née la veille

Écrite en §7.35ter le 18/08, sous la forme « les attracteurs sous-optimaux sont une
propriété du paysage, **aucune méthode locale n'en sort**, ce n'est pas réparable en
changeant d'optimiseur ». C'était la phrase la plus forte de la journée, et je l'avais
gagnée en corrigeant une erreur — j'accusais Adam, j'ai regardé le gradient, il tombe
à 7 × 10⁻¹¹, donc le point critique est réel. La déduction était fausse.

Un point critique est réel **et** franchissable par une méthode bruitée. Cellule
appariée, 20 000 pas, mêmes graines, même lr, une seule différence — gradient calculé
ou échantillonné :

| méthode | bijections |
|---|---|
| montée exacte, lr 0,05 | **0/12** |
| montée exacte, lr 0,01 | **0/12** |
| REINFORCE lot 64, lr 0,01 | **11/12** |
| REINFORCE lot 64, lr 0,05 | **9/12** |

Fisher exact p = 9,6 × 10⁻⁶.

**Ce qui rend cette mort instructive :** la mesure qui l'a tuée avait d'abord donné le
résultat inverse (0/25 sous REINFORCE), et je l'avais annoncée comme confirmant mon
interlocuteur. Ce n'était qu'un budget de pas — 4000 au lieu de 20 000. Le seul
réflexe qui a évité la publication est le caveat posé avant : *les bras ne sont pas
appariés*. Détail en §7.36.

*(Correction du 15/08/2026 au soir, §7.28 : la flèche est à l'envers. `R` est la
taille d'alphabet du code argmax, et R symboles n'admettent au plus que R référents
décodables, donc **récompense ≤ R/27** par dénombrement — vérifié 150/150 et 60/60,
déficit dans {0, 1} sur les 210 runs. R n'est pas l'objectif sur une grille : c'est
une structure qui **borne** l'objectif. Ma formulation décrivait les 141 runs où la
borne est serrée et laissait tomber les neuf où l'optimiseur rate le plafond, qui
sont les informatifs. La ligne reste disqualifiée, et pour une raison plus large.)*

### 1.20 « Le nombre de rupture est une diagnostique » — morte le 15/08/2026, née la veille

Proposée en règle 6 le 15/08/2026 au matin, réfutée le soir même : la plus courte
espérance de vie de toutes les hypothèses de ce carnet. L'idée était qu'un entier —
plus petit nombre de runs dont le retrait fait passer le contraste sous la barre —
mesure la fragilité sans supposer de loi.

Elle meurt, mais **pas pour la raison que j'ai publiée le 15/08** — corrigé le
17/08/2026, voir §7.33. J'avais écrit que son nul n'était pas identifié, la médiane du
nombre de rupture valant 2, 3 ou 4 selon la provenance des résidus. **C'était une
erreur de calibration à moi** : je fixais l'effet planté une fois, depuis le sigma par
niveau de R, puis tirais des résidus d'écart-type 10 % plus petit dans le bras
« cellules », qui portait donc un effet effectif plus grand et résistait mécaniquement
mieux. Recalibré par bras, les trois provenances donnent 0,496, 0,514 et 0,496. **La
provenance ne déplace rien.**

La vraie raison est celle du relecteur : sur un effet **vrai** planté à la taille
observée, **la moitié des réplicats cassent à deux runs sur 150** et 57 % à trois.
Le 2 observé est donc parfaitement ordinaire pour un effet réel de cette taille, et
l'entier ne disait rien de la fragilité — il disait que t valait 2,43 à n = 150. Un
seuil sur le nombre de rupture est une exigence de puissance déguisée en robustesse.

**Leçon transposable :** une statistique vendue comme sans hypothèse ne l'est
généralement pas ; l'hypothèse est déplacée vers l'étape de calibration, là où
personne ne la cherche — y compris quand c'est moi qui calibre. Ce qui reste après ce
tour est le **plancher de détection**, 2,80 × SE, fonction du plan seul et calculable
avant la première graine. Détail en §7.28 et §7.33.

### 1.26 « Je n'audite mes chiffres que dans une direction » — établie le 17/08/2026

Pas une hypothèse morte : une hypothèse **vérifiée sur moi-même**, et la seule entrée
de cette section qui ne soit pas une erreur ponctuelle mais une habitude.

Le relecteur concède que ses deux nombres non reproduits venaient d'un même choix —
conditionner sur un événement de sélection, ce qui rétrécit toujours vers le nul, donc
toujours en ma faveur ici. Une habitude appliquée deux fois, pas deux accidents.
Appliqué à moi : sur les vingt-cinq entrées ci-dessus, **vingt-quatre affaiblissent
une de mes affirmations**, une seule va dans l'autre sens (§1.3, où j'annonçais 0,5 et
mesurais 0,9966).

Ça ne prouve rien en soi — c'est aussi la trace d'une convergence depuis un départ
trop confiant. **Le test qui sépare : ai-je déjà dépensé du calcul pour rendre un
résultat négatif plus fort ?** Non, jamais. §6.2 publie « toute sélection résiduelle
est sous 0,0087 », borne qui n'est qu'une fonction du nombre de graines (100) ; six
cents graines la divisent par deux et renforcent l'affirmation négative centrale du
projet. Personne ne l'a proposé en six jours, moi compris.

**Un audit qui ne peut trouver qu'une espèce d'erreur la rapportera à 100 %.** Ce
carnet est la sortie d'un tel audit. Correctif engagé le jour même : §6.2 relancé à
600 graines contre une nulle de 200 000, graine indépendante.

**Le correctif a été lancé, et il a rendu l'inverse de ce que j'annonçais.** À 600
graines la borne passe de 0,00874 à 0,00356, et le bras factorisé sortait de zéro à
la graine 11 : z = +0,0935, IC [+0,009 ; +0,177]. J'ai écrit ici, le 17/08, que « le
correctif a mordu dès le premier essai » et que c'était ce qu'un biais d'audit
directionnel prédit. **Phrase retirée le 18/08 :** la graine 907 rend z = −0,0280,
signe inversé, et les 1200 graines réunies donnent z = +0,0327 ± 0,0299, |z/SE| = 1,09.
Le 2,18 σ était le seizième contraste de cet échange à ne pas survivre au second
tirage — et je l'avais avancé comme preuve d'un biais **dans le paragraphe qui décrit
ce biais**.

L'affirmation centrale survit : je n'avais jamais payé pour resserrer un négatif, et
un audit qui ne peut trouver que des surestimations en rapportera à 100 %. Ce qui est
mort est la preuve que j'en donnais.

Et ce que le run a acheté vaut mieux que ce que j'espérais : **borne 2,45 fois plus
serrée, conclusion de §6.2 tenue sur 1200 graines, rien à retirer.** C'est la première
entrée de cette section où payer une mesure a rendu un résultat négatif **plus fort**,
et donc le premier point de donnée contre la lecture selon laquelle je serais
incapable de trouver mes bornes lâches. Un seul point, sur l'axe où je n'en ai qu'un.
Voir §7.34.

### 1.27 « La paramétrisation factorisée porte une sélection résiduelle » — morte le 18/08/2026, née la veille

Née de mon propre correctif d'audit le 17/08 à 600 graines (z = +0,0935, IC
[+0,009 ; +0,177], p = 0,029), morte le lendemain sur une graine indépendante :
z = **−0,0280**, IC [−0,110 ; +0,054]. Réunis sur 1200 graines, z = +0,0327 ± 0,0299,
**|z/SE| = 1,09**.

Contraste à 2,18 σ dans un tableau qui en offrait deux, mort au second tirage. C'est
le seizième de cet échange et le premier que j'ai répliqué **avant** d'en écrire
l'interprétation — ce qui est le seul progrès de procédure que je peux revendiquer sur
seize tours.

**Ce qui reste, et qui n'est pas rien :** les deux runs à 600 graines diffèrent sur ce
bras de +0,1214 avec SE 0,0598, soit t = +2,03. Un run unique sous-estime donc la
variabilité inter-run, et une borne citée depuis un seul run de 600 graines est
optimiste même honnête. C'est l'intervalle groupé qui se publie désormais. Détail en
§7.34.

### 1.21 « La colonne observé/plancher est une quantité du plan » — morte le 15/08/2026, née le jour même

Proposée en §7.28 le soir, réfutée dans la nuit : elle bat le record de §1.20 de
quelques heures. `plancher = 2,80 × se` et `t = d / se` dans le même fichier, donc la
colonne imprimée **est** |t| / 2,80 identiquement — écart 0,00e+00 sur les six lignes.
Le verdict « tout effet est à son plancher ou dessous » est donc |t| < 2,80, soit
p > 0,0058 : un alpha 8,6 fois plus strict que le 0,05 de la définition du plancher.

J'avais publié le même fait deux fois dans le même message, une fois comme p
(t = 2,97 → p = 0,0035) et une fois comme rapport (1,06), en n'en signalant qu'un — et
**un message après avoir expliqué que la puissance observée est un p redimensionné**.
Le plancher **absolu** survit et sert ; c'est le rapport qui est retiré.

**Leçon :** « est-ce une fonction du plan » se vérifie sur l'expression imprimée, pas
sur l'intention. Voir §7.29.

### 1.22 « L'écart max/appariée est une grandeur continue » — morte le 15/08/2026

Présupposé de tout §7.24 à §7.29 et de la comparaison à la borne qui les précède
tous. **63 des 210 runs ont un écart exactement nul** — 30 % — parce que l'argmax non
contraint y est déjà une bijection. La grandeur est une masse ponctuelle plus une
partie positive asymétrique (asymétrie +1,34 ; ni elle ni son log ne passent Shapiro).

Trois dégâts. Le rapport publié à la borne compare **un mélange à une conditionnelle**,
la borne étant un pire cas sachant collision : 13,9 devient 9,8 apparié, et 2,4 si on
compare deux pires cas. La grandeur est en réalité **deux** — P(collision) = 0,700 et
E[inflation | collision] = 0,01479 — dont le produit vaut la moyenne publiée, et qui
ont des consommateurs différents. Et le contraste en R, défendu sur quatre tours,
passe de p = 0,0156 à **p = 0,062** en Mann-Whitney et 0,146 sur le log.

**Leçon :** `min`, `max` et un compte de zéros exacts auraient tout attrapé au premier
tour. Douze tours d'inférence de plus en plus correcte sur une variable que personne
n'avait tracée. Détail en §7.29bis.

### 1.23 « L'écart max/appariée est une propriété des codes émergents » — morte le 16/08/2026

Présupposé de §7.24 à §7.29bis en entier, et de la phrase publiée « les codes
émergents n'approchent nulle part ce qu'une recherche adverse atteint ».

Ce n'est pas une propriété des codes émergents. C'est **l'inflation propre de la loi
nulle**, reproduite par eux parce qu'ils en sont indistinguables :

| | nulle 10⁷ (11/08) | 210 runs | |
|---|---|---|---|
| P(collision d'argmax) | 0,7465 | 0,7000 | binomial p = 0,13 |
| E[inflation] | 0,01005 | 0,01035 | z = 0,36 |
| E[inflation \| collision] | 0,01346 | 0,01479 | z = 1,32 |

Et KS des 210 concentrations max contre la nulle : D = 0,0508, **p = 0,638**.

**Ce qui rend cette mort la plus coûteuse du carnet :** les trois nombres de la
colonne de gauche étaient dans `loi_nulle_longue_n10000000_g0.json` depuis le
11/08/2026, sous les noms `taux_global`, `inflation_moyenne_globale` et
`inflation_maximale`. Et §6.2, publié le même jour, avait déjà établi que les codes
émergents sont tirés de la nulle (z = −0,0098 ± 0,1025, KS p = 0,386). La question
était répondue **avant d'être posée**, par deux fichiers du même répertoire aux
horodatages identiques.

Conséquence sur le rapport publié : mon 15 comparait une **moyenne** à un **maximum**
sur des tailles d'échantillon séparées de cinq ordres de grandeur. De même nature,
c'est 0,1443 contre 0,1081 — la recherche adverse fait **34 % de mieux que le hasard,
pas quinze fois**.

Règle 8 en §7.30 : quand un résultat est établi, lister ce à quoi il répond.

### 1.24 « Le maximum d'inflation de la nulle vaut 0,1081 » — morte le 17/08/2026

Publiée dans `loi_nulle_longue_n10000000_g0.json` sous `inflation_maximale`, puis
reprise en §7.30 pas plus tard qu'hier pour corriger un rapport. Elle sort du
réservoir, qui cesse de se remplir à 2 000 000 tirages, et s'imprime sous « toute la
loi ». **Le vrai maximum sur 10⁷ vaut 0,122365, et 13 tirages dépassent le nombre
publié.** Le rapport à la borne de recherche passe de 1,34 à 1,18.

Deuxième défaut du même bloc, celui-là trouvé en vérifiant le premier :
`taux_global` = 0,7465 compte les collisions d'argmax, pas celles qui coûtent quelque
chose, lesquelles font 0,6762. Mon observé étant calculé sur le coût, la comparaison
publiée hier mélangeait deux définitions — corrigée, **E[inflation | > 0] passe de
z = 1,32 à z = −0,07**.

**Ce qui rend celle-ci différente des vingt-trois précédentes :** ce n'est pas une
faute de raisonnement, c'est un plafond de tampon jamais relu, dans un fichier public
depuis le 12/08 et utilisé par moi aux tours 7, 8 et 13. Elle a survécu à treize tours
de critique statistique parce que personne ne relisait le code qui produisait les
nombres qu'on corrigeait. Corrigé dans la source, avec les comptes de dépassement que
le script réservait à la colonne voisine. Détail en §7.31.

### 1.25 « 0,1443 est le pire cas atteignable » — morte le 17/08/2026

Publiée dans `appariement_4000par_famille_1500par_niveau_g7.json` sous
`pire_cas.inflation_maximale`, citée comme borne pendant neuf tours, et corrigée deux
fois dans les vingt-quatre dernières heures **sans que ni moi ni le relecteur ne
relise sa provenance**. Elle vient de `recherche_pire_cas(..., n_restarts=24)` : c'est
le meilleur de **vingt-quatre montées** à la graine 7. Ma graine 0 aux mêmes
vingt-quatre donne 0,146685, et le budget la fait monter jusqu'à 0,154322 à 384
départs.

**Donc les deux membres du rapport que nous corrigions étaient des statistiques
d'ordre**, à budgets non déclarés et de sens opposés. Pire : la nulle tire des
bijections uniformes, le grimpeur part d'une permutation et bouge par transpositions
qui préservent la bijectivité, et l'objectif est la même fonction. **Ce sont deux
estimateurs du même supremum, et le rapport converge vers 1.**

Le supremum, lui, se calcule : 1500 départs, deux voisinages indépendants, plateau à
**0,154322** en 43 secondes. Tout ce que l'un ou l'autre a publié en est une fraction
— 93,5 % pour ma borne, 90,1 % pour son max à 3·10⁹, 79,3 % pour le mien à 10⁷,
6,7 % pour la moyenne émergente.

**Ce qui rend celle-ci la plus gênante du carnet :** ce projet existe pour que 27
référents rendent l'optimum, la loi nulle et le gradient **calculables plutôt
qu'estimés** — c'est le titre de l'article 3. Dans le seul endroit où il fallait un
maximum, nous avons tous les deux tiré au sort, lui à 3·10⁹, moi à 10⁷, et la borne
publiée à 24 départs. Détail en §7.32.

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

### 4.7 Mon critère de falsification du test 3 omettait la variable qui décide — 11/08/2026

C'est le défaut le plus sérieux trouvé aujourd'hui, et le plus gênant, parce que
l'engagement de TEST3.md §5 avait été enregistré **avant toute donnée**, daté, et
présenté comme ce qui rendait le test « falsifiable de façon bien plus tranchante
qu'un seuil arbitraire ». Il l'était. Il était aussi sous-spécifié sur trois points.

**1. Il ne nomme pas la paramétrisation.** Mesuré : `z = −0,12` en tabulaire,
`−0,25` en factorisé, `+9,92` en structuré. Le même engagement est confirmé sur
deux paramétrisations et réfuté sur une troisième, et son énoncé ne permet pas de
dire laquelle il visait. Un critère de falsification qui omet la variable dont
dépend la réponse ne tranche rien — il enregistre une intuition.

**2. Sa clause d'interprétation est fausse.** Il annonce qu'un dépassement
signifierait que « le raisonnement des optima à égalité comporte une faille ». Le
dépassement a lieu, et cette conclusion ne suit pas : §7.15 montre que le
raisonnement était depuis toujours conditionnel à la symétrie de la
paramétrisation, ce que ni §3 ni §5 de TEST3.md n'énonçaient. La récompense reste
indifférente. J'avais donc écrit d'avance la mauvaise interprétation de mon propre
test.

**3. Sa première moitié est fausse.** « Les codes émergents seront des bijections
quasi parfaites » : une sur vingt en tabulaire, zéro sur vingt ailleurs, 2 à 5
collisions. Le succès de tâche est élevé (E[R] ≈ 0,92), mais **récompense élevée et
bijection quasi parfaite ne sont pas la même chose**, et je les avais confondues.

Ce qui survit intact : *non compositionnels*, sur toutes les paramétrisations, à au
moins 13 référents sur 27 du plus proche code compositionnel.

**Leçon, et elle vaut au-delà de ce test.** Enregistrer une prédiction à l'avance
protège de l'ajustement après coup. Ça ne protège **pas** d'avoir omis une
variable, ni d'avoir écrit d'avance la mauvaise interprétation. Un engagement
daté doit donc nommer explicitement : sur quelle population, sous quelle
paramétrisation, et ce que chaque issue prouverait — le troisième point étant celui
que j'ai raté.

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

### 7.15 §6.7 traité : le certificat tombe, et son remplaçant désigne le coupable

11/08/2026, `certificat_deux_agents.py`. C'était « la question la plus
inconfortable, et je ne connais pas la réponse », et « le premier endroit du projet
où un résultat que j'ai publié pourrait s'effondrer sur un point technique que je
n'ai pas vérifié ». Il s'est effondré.

**Ce qui casse, et pas où je le cherchais.** Je soupçonnais le terme d'entropie,
qui porte sur deux politiques séparément. Le défaut est en amont : le certificat
exige que les objets à égalité soient le support de la loi dont l'entropie est dans
l'objectif. Détail en §1.11.

**Le remplaçant, plus fort et plus étroit.** `c → π ∘ c` est transitive sur les 27!
bijections, donc une paramétrisation tabulaire à initialisation échangeable rend les
27! codes exactement équiprobables — sans Gibbs. Vérifié : 8 essais sur 8 rendent
exactement `π ∘ c`. C'est un **théorème sur la paramétrisation**, valable pour tout
algorithme équivariant, et non un résultat d'optimisation.

**Le compte qui tombe juste, et qui est le vrai résultat.** Les renommages
respectant la décomposition en (m₁, m₂, m₃) forment un groupe d'ordre **exactement
1 296** — compté par retour arrière et pas seulement construit, les lignes du monde
étant exactement les triangles du graphe de Hamming H(3,3). Et les 1 296 codes
compositionnels sont **exactement l'orbite du code canonique sous ce groupe**, les
deux ensembles étant construits par deux chemins de code indépendants. Autrement
dit : la seule paramétrisation dont le groupe est plus petit que `S₂₇` est
précisément celle dont le groupe distingue les codes compositionnels.

**Ce que ça change au programme, et c'est beaucoup.** §6.1 et §6.2 sur un émetteur
tabulaire ne peuvent **rien découvrir** : leur issue est un théorème. Ils gardent
une valeur comme détecteurs de bogue — un écart au hasard prouverait que
l'implémentation a cassé la symétrie. L'expérience réelle est le contraste
tabulaire / structuré. Et §6.6 reçoit une hypothèse unificatrice : bruit de canal,
goulot de vocabulaire, pression de longueur, renouvellement de population brisent
tous `S₂₇` vers un groupe respectant la structure de produit. Une liste de recettes
devient une question unique.

**⚠️ Cette dernière phrase est fausse, et §6.6 l'a montrée le soir même.** Le
renouvellement ne brise pas `S₂₇` du tout, et le bruit de canal le brise sans rien
produire. Briser la symétrie est **nécessaire, pas suffisant**. Voir §1.16 et
§7.22. Je laisse la phrase ici plutôt que de la réécrire : elle a été formulée
avant la mesure, elle a orienté l'expérience qui l'a réfutée, et c'est exactement
ce qu'une hypothèse doit faire.

**Trois erreurs à moi dans ce fichier, trouvées avant de le committer.**

1. J'avais écrit qu'une ligne d'émetteur autorégressif est une loi produit. Faux :
   `P(m₁)P(m₂|m₁)P(m₃|m₁,m₂)` représente n'importe quelle loi sur 27 messages.
   L'argument porte sur la **symétrie de la paramétrisation**, pas sur
   l'expressivité — un π quelconque ne correspond à aucune permutation des poids,
   donc l'équivariance tombe même à expressivité pleine. Même forme qu'au test 2,
   où l'effondrement venait de la factorisation et non de l'objectif. Le 1 296 est
   donc un **majorant** : une architecture concrète peut être bien moins symétrique.
2. Mon tableau de phase lisait **un seul départ par β**, et la variation entre β
   voisins (0,926 puis 0,889 puis 0,963) était du bruit d'initialisation. C'est la
   faute de §1.6 recommise. En 8 départs le tableau devient unanime : 100 %
   d'échappement jusqu'à 0,037, 0 % à partir de 0,040, min = max.
3. Le 0,0381 mesurait Adam et non l'objectif. Voir §1.12.

**Un résultat de §6.5 arrivé avec deux étapes d'avance.** Les valeurs atteintes
depuis le babil sont 0,8518 · 0,8888 · 0,9259 · 0,9629 · 1,0000, soit exactement
23/27 à 27/27. **La montée de gradient exacte, sans le moindre échantillonnage, ne
rejoint presque jamais un code parfait depuis le babil** — un départ sur 40 — et se
pose sur des codes où 1 à 4 référents entrent en collision. Partie *sur* un code
parfait, elle y reste à 1,0000. C'est *atteignable ≠ stable*, sans pouvoir
l'imputer au bruit d'échantillonnage puisqu'il n'y en a aucun.

**Et le piège que ça ouvre, fermé le jour même.** La loi nulle de §6.1 est tirée sur
des **bijections**, et le chemin vectorisé de `loi_nulle_longue.py` s'appuie sur
l'identité « les deux marges d'un code bijectif sont uniformes ». Sur un code
non bijectif il rend des nombres faux **sans lever d'erreur** : mesuré 0,110573 au
lieu de 0,108071, soit 0,0025, un cinquième de ce que §6.2 doit résoudre. Garde
ajoutée. Mais le fond demeure et doit être traité **avant** §6.1 : comparer un code
émergent non bijectif à une loi nulle bijective compare deux supports différents.
La sortie retenue est de tirer la nulle sur la classe réellement atteinte, à nombre
de collisions apparié run par run.

**Ce qui rend tout ça vérifiable.** Aucun entraînement n'a tourné. Deuxième fois
dans la journée que ce point sauve une correction : elle ne peut pas avoir été
choisie au vu d'un résultat, et l'historique git le montre.

### 7.16 §6.5 traité : les trois réponses sont différentes, et §6.7 était incomplet

11/08/2026, `representable_atteignable_stable.py`. Trois paramétrisations
d'émetteur, même objectif, même optimiseur, même récepteur tabulaire. Seule la
carte des paramètres change.

**L'équivariance rendue visible.** En tabulaire, l'ajustement supervisé vers le
code compositionnel et vers une bijection quelconque prend **2 198 pas dans les
deux cas**, à l'unité près, et l'écart d'E[R] vaut −1,1 × 10⁻⁷. En stabilité,
l'écart vaut −2,3 × 10⁻¹⁰. La prédiction de §6.7 tient à la précision machine.

**Et §6.7 était incomplet, ce que seule cette mesure a révélé.** J'y avais raisonné
sur le renommage des **messages**. Or `c → c ∘ ρ⁻¹`, le renommage des
**référents**, est lui aussi transitif sur les 27! bijections : l'équivariance
d'**un seul des deux côtés** suffit. C'est pourquoi la paramétrisation `factorise`
— autorégressive en tokens, mais à paramètres libres **par référent** — ne préfère
rien du tout : écart 3,3 × 10⁻¹⁶, puis −0,0013 sur 20 graines. J'avais construit ce
contraste en croyant qu'il contrasterait ; il ne contrastait rien, et c'est la
mesure qui me l'a appris, pas le raisonnement.

> **Une table d'embedding libre par référent annule d'avance tout ce que la
> structure du message pourrait apporter.** La plupart des implémentations feraient
> ça sans le savoir.

**Le contraste qui contraste.** Un émetteur où le référent entre par ses
**attributs** avec des **poids partagés** (81 + 9 poids contre 729 libres) :

| | E[R] | bijections | collisions | concentration appariée |
|---|---|---|---|---|
| tabulaire | 0,9240 | 1/20 | 2,00 | 0,1283 ± 0,0405 |
| factorisé | 0,8092 | 0/20 | 5,15 | 0,1270 ± 0,0379 |
| structuré | 0,8573 | 0/20 | 3,80 | **0,4233 ± 0,1233** |

`structure − tabulaire = +0,2950`, soit 7,3 écarts-types, à récompense et objectif
identiques.

**Ce que je refuse d'en conclure.** `structure` ne peut pas écrire la plupart des
bijections : contrôle à 20 000 pas et lr 0,2, il atteint 1,00000 sur le
compositionnel en 704 pas et plafonne à 0,09–0,24 sur les bijections quelconques.
C'est donc une limite de **capacité**, pas un échec d'optimisation — et trouver
qu'une paramétrisation qui ne peut écrire que des codes structurés produit un code
structuré n'est pas une émergence. C'est ce que §6.6 prévoyait déjà, mesuré ici
contre une ligne de base calculée exactement.

Trois nuances qui l'empêchent d'être un résultat plus gros qu'il n'est :

1. `structure` **n'atteint pas** le code compositionnel — il rendrait 1,0000, il
   rend 0,4233. La contrainte produit de la structure partielle ;
2. elle la paie : E[R] de 0,9240 à 0,8573 et collisions de 2,00 à 3,80. C'est la
   taxe de mise en forme de §2.3 dans un autre décor ;
3. la loi nulle appariée (0,1168 ± 0,0315) n'est pas une référence valide ici, les
   codes atteints n'étant pas bijectifs. Seul le contraste entre paramétrisations
   est valide, les trois populations étant comparées entre elles.

**Ce qui reste dû avant §6.1** : la loi nulle sur la classe réellement atteinte,
à nombre de collisions apparié.

### 7.17 §6.1 traité : le théorème tient, et mon critère de falsification était sous-spécifié

11/08/2026, `code_emergent.py`. La loi nulle est d'abord corrigée : les codes
atteints n'étant pas bijectifs, la référence de chaque run est tirée uniformément
parmi les applications de **même profil de fibres**.

**Pourquoi cette classe-là, et c'est un argument exact et non un rapprochement.**
Le groupe `S₂₇ × S₂₇` agit par `(π, ρ)·c = π ∘ c ∘ ρ⁻¹`, et deux applications sont
dans la même orbite **si et seulement si** elles ont le même profil de fibres. La
paramétrisation tabulaire étant équivariante des deux côtés, la loi de sortie
conditionnée au profil est exactement uniforme sur ce profil. `z = 0` est donc un
théorème, pas une attente, et §6.1 devient un test sans paramètre libre.

| | concentration appariée | z | > q99,9 | distance au compositionnel |
|---|---|---|---|---|
| tabulaire | 0,1131 ± 0,0296 | **−0,12 ± 0,22** | 0/20 | 21,4 (min 20) |
| factorisé | 0,1091 ± 0,0343 | **−0,25 ± 0,25** | 0/20 | 21,8 (min 20) |
| structuré | 0,4240 ± 0,1056 | **+9,92 ± 0,78** | 19/20 | 15,8 (min 13) |

**Le théorème tient en distribution, pas seulement en moyenne.** Les centiles des
runs dans leur propre nulle doivent être uniformes sur [0, 1] : Kolmogorov-Smirnov
donne **D = 0,090, p ≈ 0,995** en tabulaire. Et l'écart-type des z vaut **0,97** —
la nulle appariée a la bonne **forme**, ce qui valide toute la construction, pas
seulement son centre. La distance de Hamming au compositionnel, qui n'utilise
aucune information mutuelle, confirme : 21,4 → 15,8.

**La correction sur laquelle j'ai insisté deux fois ne change rien.** Sur les onze
profils rencontrés, la nulle appariée s'écarte de la bijective de −0,0001 à
+0,0005, quand l'effet vaut 0,30. Elle était nécessaire à **vérifier** — sans quoi
l'écart mesuré aurait été suspect — et elle ne déplace pas la conclusion. Ce qui
change vraiment avec la non-bijectivité n'est pas la nulle, c'est le choix de la
statistique (§1.13).

**Le défaut sérieux du jour, et il est de moi.** Voir §4.7.

### 7.18 Ce que §6.1 ne prouve pas, et que je refuse de laisser croire

Le `z = +9,92` de la paramétrisation structurée n'est **pas** une émergence de la
compositionnalité. Cette paramétrisation ne peut pas écrire la plupart des
bijections (§7.16, contrôle de capacité), elle n'atteint pas le code compositionnel
(0,4240 et non 1,0), et elle paie sa structure en succès de tâche. La récompense
reste rigoureusement indifférente : c'est la paramétrisation qui sélectionne, et
c'est très exactement la thèse du projet depuis le test 2, démontrée cette fois des
deux côtés.

Ce qui est neuf n'est donc pas « la structure émerge », c'est :

1. qu'on peut **prouver** qu'elle ne peut pas émerger d'une paramétrisation
   équivariante, quel que soit l'algorithme (§7.15) ;
2. qu'une table d'embedding libre par référent suffit à garantir cette
   équivariance sans qu'on s'en aperçoive (§7.16) ;
3. et que la ligne de base contre laquelle tout ça se mesure est calculée
   exactement, orbite par orbite, plutôt que devinée.

### 7.19 §6.2 traité à 100 graines : un négatif avec sa borne, et un β qui aide

11/08/2026, `dynamique_uniforme.py`. §6.1 avait répondu à 20 graines, et c'était
insuffisant : sous le critère du document lui-même (unilatéral p < 0,001, puissance
80 %), 20 graines ne résolvent que **0,027**. Le scénario « une pression faible
soulève tous les runs de 0,02 », que Dipankar Sarkar décrivait comme bien plus
probable qu'un run isolé au-delà d'un seuil, y serait passé inaperçu. Conclure
« indiscernable » à 20 graines aurait été une conclusion que le dispositif ne
portait pas — c'est §1.6 sous une autre forme.

| | n | C appariée | z moyen | IC 95 % | KS *p* | > q99,9 |
|---|---|---|---|---|---|---|
| tabulaire | 100 | 0,1164 | **−0,01 ± 0,10** | [−0,21 ; +0,19] | 0,386 | 0/100 |
| factorisé | 100 | 0,1152 | **−0,05 ± 0,10** | [−0,25 ; +0,15] | 0,613 | 0/100 |
| structuré | 20 | 0,3971 | **+9,01 ± 0,60** | [+7,84 ; +10,18] | 0,000 | 20/20 |

**Le négatif est énoncé avec sa borne, pas comme une absence.** À 100 graines et
sd de nulle 0,0312 : détectable à partir de **0,0087** (bilatéral p < 0,05) ou
**0,0123** (unilatéral p < 0,001). Donc toute sélection résiduelle par la dynamique,
sur paramétrisation équivariante, est plus petite que 0,0087 de concentration.

**Balayage en β** — 20 graines par β, parce qu'un seul β n'est pas une propriété et
que [0,037 ; 0,170] est bistable. z = +0,12 · +0,41 · −0,01 · +0,12 · +0,02 pour
β = 0,005 · 0,010 · 0,020 · 0,030 · 0,037, KS *p* de 0,070 à 0,999. Aucun β ne sort.

**Observation non cherchée** : monter β jusqu'au seuil **améliore** le code, E[R] de
0,887 à 0,931 et collisions de 2,95 à 1,75. L'entropie aide la coordination tant
qu'elle ne détruit pas le code — l'inverse de l'intuition du test 2, où l'entropie
était la taxe.

**La phrase juste, et elle est plus étroite que celle que j'aurais écrite.** Ce
n'est pas « la dynamique tire au hasard parmi les codes ». C'est « la dynamique
tire au hasard **sur l'orbite**, quand la paramétrisation est équivariante ». Le
profil de fibres n'est pas tiré au hasard du tout : c'est la dynamique qui le
choisit. Un test qui ne conditionnerait pas dessus mesurerait ce choix et
l'appellerait sélection de code.

Les trois issues listées en §6.2 sont donc toutes réalisées, selon la
paramétrisation — sauf la troisième, « la dynamique fuit activement les codes
structurés », qui n'est réalisée nulle part.

### 7.20 §6.3 traité : personne n'écrit le code, et j'avais mesuré le mauvais écart

11/08/2026, `qui_ecrit_le_code.py`.

**La réponse à la question du titre est « ni l'un ni l'autre ».** Geler l'émetteur
et laisser apprendre le récepteur, ou l'inverse, donne **139 pas dans les deux
sens** et la même valeur finale **à huit décimales** (0,99992302). Le problème est
exactement symétrique : c'est la bilinéarité de §4 rendue visible. Et geler sur le
code compositionnel ou sur une bijection quelconque est le même problème, à
6 × 10⁻⁹ près — équivariance, pour la troisième fois de la journée.

**L'erreur, et elle était dans l'interprétation, pas dans le code.** J'avais
d'abord écrit « coût de la coordination = 0,049 », en comparant la paire libre
(E[R] = 0,911) à un agent gelé sur une **bijection** (0,9999). Deux défauts :

1. un code à *k* collisions plafonne **arithmétiquement** à (27 − *k*)/27, deux
   référents envoyés sur le même message étant indistinguables quoi que fasse le
   récepteur. Comparer une paire à 2,4 collisions à un agent qui a reçu une
   bijection compare deux **plafonds**, pas deux apprentissages ;
2. mon seuil de vitesse, « pas pour atteindre 0,99 », est **inatteignable dès la
   première collision**. Il mesurait une capacité en croyant mesurer une vitesse.

Le plafond est maintenant vérifié et non supposé : gelé sur un code à 2 collisions,
l'agent libre atteint 0,9259, soit exactement 25/27, à −0,0000 près.

**Le vrai résultat, une fois la mesure corrigée :**

| les deux libres | E[R] | collisions | plafond | E[R]/plafond |
|---|---|---|---|---|
| S tabulaire | 0,9111 | 2,40 | 0,9110 | **1,0000** |
| S structuré | 0,8777 | 3,30 | 0,8777 | **1,0000** |

> La paire libre exécute son code exactement aussi bien qu'un agent à qui on
> aurait donné ce même code tout fait. Le déficit n'est **pas** dans
> l'apprentissage : il est entièrement dans le code sur lequel les deux se posent.

La coordination coûte en vitesse (260 pas contre 139) et en qualité du code atteint
(2,4 collisions au lieu de 0), et **rien** en exécution. C'est exactement la
localisation que §6.3 cherchait, et elle est plus nette que ce que j'espérais.

**Et une condition où l'échec change de nature** : `R gelé aléatoire, S structuré
libre` plafonne à 0,5924. C'est la première du test 3 où l'échec vient de la
**représentabilité** et non de la coordination — §6.5 l'avait mesuré en supervisé,
on le retrouve dans le jeu.

**Leçon d'instrument, la deuxième du genre aujourd'hui.** Un seuil absolu de
réussite n'est comparable entre conditions que si toutes peuvent l'atteindre. Ici
0,99 était hors de portée d'une condition sur deux, par arithmétique et non par
difficulté. Voir aussi §1.12 : mesurer à travers un instrument qui ne peut pas
répondre, c'est mesurer l'instrument.

### 7.21 §6.4 traité : le gradient initial ne voit rien, et la préférence naît au pas 30

11/08/2026, `gradient_premier_pas.py`.

**La prédiction de §4 tient.** Coefficient de variation du gradient dans l'espace
des lois : 1,0 × 10⁻² pour `∂E[R]/∂S`, 9,8 × 10⁻³ pour `∂E[R]/∂R`. Aucune direction
préférée à l'initialisation, contrairement au test 2 où le déséquilibre du lexique
en imposait une dès le pas 1.

**La prédiction que j'avais ajoutée est fausse** (§1.14), et la courbe qui répare
la réfutation est le meilleur résultat de la section :

| pas | 0 | 10 | 30 | 100 | 300 | 1 000 | 3 000 |
|---|---|---|---|---|---|---|---|
| tabulaire | +0,07 | +0,07 | −0,30 | +0,30 | +0,19 | +0,16 | +0,19 |
| structuré | −1,18 | −0,29 | **+4,36** | +4,25 | +3,91 | +5,81 | +5,85 |

La tabulaire ne préfère **jamais** le compositionnel, à aucune profondeur. La
structurée s'y met **brutalement entre le pas 10 et le pas 30**. §6.4 demandait
qu'on nomme la direction si elle existait : elle n'existe pas au départ, elle est
créée par la concentration de la loi.

**L'issue est-elle écrite dans l'initialisation ?** Témoins appariés au profil de
fibres du code atteint, faute de quoi on mesurerait l'effet du profil.

| | z du code atteint | centile | argmax initial conservé |
|---|---|---|---|
| tabulaire | +6,80 ± 0,18 | 1,000 | 8,7 % ± 5,5 |
| structuré | −0,52 ± 0,18 | 0,333 | 3,7 % ± 3,7 |

**Deux lectures à concilier plutôt qu'à trier.** Le hasard vaut 3,7 %. En
tabulaire, l'initialisation classe le code final **premier sur 300** alternatives
appariées, mais n'en écrit que 8,7 %, soit 2,3 référents sur 27. « L'issue est
décidée à l'initialisation » serait donc exagéré ; la formulation juste est
« l'initialisation biaise fortement en agrégat, sans écrire le code ». J'ai ajouté
la mesure sans loi nulle ni cosinus précisément pour ne pas pouvoir me contenter du
z, qui est le chiffre le plus flatteur des deux.

En structuré, l'empreinte initiale est **exactement nulle**. Tout vient de la
trajectoire. C'est le miroir de §7.11sexies au test 2, et le partage
initialisation/trajectoire s'inverse d'une paramétrisation à l'autre.

**Deux défauts de protocole corrigés avant de lancer**, tous deux déjà commis
ailleurs aujourd'hui : les cosinus étaient lus sur la **dernière graine** au lieu
d'être moyennés (§1.6 encore), et les témoins de P4 étaient des bijections alors
que le code atteint a des collisions (§7.17 encore). Deux pièges que je venais de
corriger, retombés dedans à deux sections d'intervalle.

### 7.22 §6.6 traité : la seule chose qui a marché aujourd'hui est la paramétrisation

11/08/2026, `courbe_de_contrainte.py`. Dernière étape du programme du test 3.

**Le dispositif le plus pur que ce banc pouvait produire.** Le canal laisse
l'égalité des récompenses **exactement** intacte (§1.15) tout en brisant la
symétrie de l'objectif. Donc le certificat des optima à égalité continue de dire
que rien ne distingue les bijections, et le théorème d'équivariance ne s'applique
plus : toute sélection observée opérerait entièrement hors de la récompense.

**Et il ne se passe rien.** 15 graines par ε, émetteur tabulaire :

| ε | 0,00 | 0,05 | 0,10 | 0,20 | 0,30 | 0,50 |
|---|---|---|---|---|---|---|
| E[R] | 0,9333 | 0,8344 | 0,7598 | 0,6104 | 0,4840 | 0,0370 |
| z | −0,44 | −0,05 | −0,28 | +0,38 | +0,15 | −0,02 |

Borne à 15 graines : |z| < 0,72, soit 0,024 de concentration, contre +9,9 pour la
paramétrisation structurée — un facteur quatorze. À ε = 0,5 le canal détruit le
code avant de le structurer (babil pur, 9,4 collisions).

**Le renouvellement ne fait rien non plus, et c'était prédit** : z de −0,34 à +0,33
sur quatre périodes. L'équivariance survit à une opération échangeable, donc le
théorème s'applique encore. Si l'*iterated learning* produit de la
compositionnalité, ça ne peut pas venir du renouvellement seul sur ce banc — il
faut un biais inductif du réapprenant. Je n'ai pas fait la revue de littérature,
donc c'est une conclusion sur mon dispositif, pas sur les travaux des autres.

**La conclusion, et elle est plus dure que la courbe que j'attendais.**

> De tout ce qui a été testé aujourd'hui, une seule chose a produit de la
> compositionnalité : la **paramétrisation**. Et elle l'a fait en rendant les
> alternatives inécrivables, pas en les départageant. La seule contrainte
> d'environnement qui marcherait le fait en mettant la préférence dans la
> récompense. Sur ce banc, la compositionnalité n'a jamais été **sélectionnée** :
> elle a été soit impossible, soit spécifiée.

C'est la thèse du projet depuis le test 2, poussée aussi loin que ce banc le
permet, et elle survit à tout ce que je lui ai opposé aujourd'hui.

### 7.23 La revue de littérature, faite après coup : mon no-go est publié depuis 2021

12/08/2026, après la release 0.5.0. C'était le dernier blocage de fond que
j'identifiais, et il fallait le lever : l'article 3 écrivait « someone has very
likely written it down » à propos du théorème d'équivariance, ce qui est honnête
mais paresseux. Vérifié, quelqu'un l'a écrit.

**Kuciński, Korbak, Kołodziej, Miłoś — « Catalytic Role of Noise and Necessity of
Inductive Biases in the Emergence of Compositional Communication », NeurIPS 2021**
([arXiv:2111.06464](https://arxiv.org/abs/2111.06464)).

**Leur théorème 1 est mon no-go, sous une autre forme.** Ils montrent que pour une
loi uniforme μ sur les traits et une permutation π, la loi μ∘π⁻¹ est encore
uniforme — donc l'apprentissage non supervisé de la compositionnalité est
impossible sans biais inductif. C'est le même argument de symétrie que le mien.
Différence de portée, et je ne la surestime pas : leur théorème porte sur la
**loi des données**, le mien sur la **carte des paramètres** (équivariance de la
procédure entière, transitivité sur les 27! codes, donc P(compositionnel) =
1296/27! exactement). Le corollaire sur la table d'embedding libre par référent
est au niveau de l'architecture et je ne l'ai pas trouvé énoncé sous cette forme,
mais c'est une affirmation sur ma recherche, pas sur la littérature.

**Leur théorème 2 explique mon §6.6, et le confirme au lieu de le contredire.**
Énoncé vérifié sur deux rendus indépendants du papier : *« a language ℓ\* minimizes
J over all languages ℓ which are one-to-one mappings if and only if ℓ\* is
compositional »*, sous deux conditions conjointes. La perte
`J(ℓ,f) = 𝔼[H(ρ(f′,f))]` est bâtie sur la **distance de Hamming entre traits
corrompus et traits d'origine**, donc crédit partiel par trait et non
tout-ou-rien. Et le bruit doit vérifier **ε < (|𝒜|−1)/|𝒜|**, soit ε < 0,667 pour
mon alphabet de trois tokens — mon balayage allait à 0,5, dans leur domaine.

**Distinction à ne pas effacer** : leur théorème dit quel langage **minimise** la
perte sur les bijections. §6.6 dit lequel la dynamique **atteint**. Leur condition
satisfaite n'implique pas qu'une méthode de gradient y arrive, et à ε = 0,5 mon
système s'effondrait au babil avant de structurer quoi que ce soit. Or §6.6 mesure que le bruit seul ne produit rien, et démontre en une ligne
que sous une récompense **tout-ou-rien** l'égalité des codes survit à tout ε. Les
deux résultats sont la même chose vue des deux côtés :

> Ma preuve en une ligne établit **pourquoi leur condition de perte factorisée est
> nécessaire**, et mon +0,108 à ε = 0,2 sous crédit partiel est la contrepartie
> empirique de leur théorème 2.

Et leur formule « le bruit est nécessaire mais pas suffisant » est mot pour mot ma
conclusion de §1.16, atteinte par un autre chemin le même jour.

**Ce que ça change à l'article 3**, qui n'est pas encore publié : la section Limits
ne peut plus dire « pas de revue » ; le no-go doit être attribué ; et le §6.6 doit
dire que son négatif est la moitié complémentaire d'un théorème publié plutôt
qu'une découverte isolée. Les notes de la 0.5.0 restent telles qu'elles ont été
déposées sur Zenodo — la revue est postérieure, et réécrire un artefact archivé
serait pire que le laisser daté.

**La leçon d'ordre.** J'ai fait la revue en dernier, après sept expériences, un
article et une release. Elle a coûté vingt minutes et elle a changé le statut de
deux résultats sur trois. L'ordre correct était l'inverse, et je le savais : je
l'avais écrit le matin même comme « le premier pas ».

### 7.24 Sixième critique : annoter une borne périmée n'est pas la corriger

12/08/2026, après publication de l'article 3. Dipankar Sarkar fait remarquer que
les trois bornes de « Bounding the damage » ont été mesurées par montée locale sur
des **permutations**, alors que §6.5 et §6.7 établissent que les codes atteints ont
1 à 4 collisions. Je l'avais écrit — « conditionnelles à la bijectivité » — puis je
les avais laissées telles quelles. Sa remarque : la suite n'était pas de les
annoter mais de les **relancer**.

**Le diagnostic est pire que « mesuré dans le mauvais régime ».** Mon grimpeur
bougeait par **transpositions**, et une transposition d'une permutation est une
permutation. Le jeu de mouvements ne pouvait donc pas quitter le régime bijectif,
même en principe. Ce n'était pas une hypothèse que j'avais omis de vérifier :
elle était **soudée dans l'opérateur**. Troisième fois cette semaine que c'est
l'instrument, et non le raisonnement, qui décide de la réponse.

**Sa mesure reproduite** (`bornes_par_messages_distincts.py`, réaffectation d'un
référent, plancher R sur les messages distincts, budget identique par colonne) :

| plancher R | 27 | 26 | 25 | 24 | 23 |
|---|---|---|---|---|---|
| écart max, lui | 0,0396 | 0,1628 | 0,1850 | 0,2002 | 0,2112 |
| écart max, moi | 0,0526 | **0,1362** | 0,1783 | 0,1850 | 0,2152 |
| max en double compte, moi | 0,1943 | **0,6409** | 0,6530 | 0,6700 | 0,7015 |

**Une collision fait tout l'effet.** Et le second chiffre tombe à l'identique chez
nous deux : **0,6409 avec apparié 0,5812** au plancher 26, deux grimpeurs
différents, quatre décimales. Ma borne sur permutations valait 0,6314 : **une seule
collision bat une recherche restreinte à la bijectivité sur tout l'espace.**

**Sa question, et la réponse tient sur mes propres données.** Il demande si R = 26
est assez fréquent pour être le cas modal plutôt que la frontière. Non — **le mode
est R = 25**. Sur 100 graines tabulaires à β = 0,02 : R = 27 dans **3** runs, 26
dans 37, **25 dans 43**, 24 dans 16, 23 dans 1. Quatre-vingt-dix-sept sur cent sont
à la première collision ou au-delà, et le run médian est une collision plus loin
que le point où l'effet est déjà arrivé. Le balayage en β le confirme : R moyen de
24,05 à 25,25, jamais 27. Les deux autres paramétrisations sont plus profondes
encore, modes R = 21 et R = 23.

**Ce que ça durcit.** J'écrivais que la statistique appariée est la seule qui reste
interprétable une fois la bijectivité perdue. L'énoncé correct est le sien : le
double compte ne se dégrade pas graduellement, **il arrive presque entièrement à la
première collision** — et la première collision, c'est 97 % des runs. Donc la
statistique publiée n'est pas fragile au bord du régime : elle est **inutilisable
dans le régime où l'expérience opère**.

Les trois bornes sont remplacées par le tableau indexé par R, pas supprimées, avec
le diagnostic de l'opérateur écrit à côté.

### 7.25 Septième critique : j'avais corrigé le dénominateur, le mal était au numérateur

12/08/2026. Dipankar Sarkar croise **sa** loi nulle par R avec **ma** distribution
p(R) sur 100 graines, et le produit renverse mon diagnostic de §7.24.

**Sa mesure, 200 000 tirages par cellule.** La nulle appariée vaut 0,11679 à
R = 27 et 0,11697 à R = 24 ; mélangée sur mon p(R) elle donne 0,11696, soit
**+0,00016** par rapport à la nulle bijective, pour une erreur type de 7,0 × 10⁻⁵.
Autrement dit : **plate sur tout le support que mon procédé visite.** Ça recoupe
ma propre mesure par profil de fibres (onze profils, −0,0001 à +0,0005), obtenue
par un découpage différent — deux routes, même conclusion.

**Et son coup, qui est juste.** Le pire cas atteignable, lui, bouge énormément :
0,0396 à R = 27 contre 0,1751 mélangé sur p(R) chez lui. Le null bouge de 0,00016,
la mesure de 0,1355. **J'ai passé une section à reconstruire la référence quand le
dommage était dans l'ensemble atteignable de la recherche.** Le numérateur, pas le
dénominateur. Et mes trois bornes n'étaient donc pas trop généreuses : elles
étaient **trop petites**. Une rétractation qui allait dans le mauvais sens.

**Correction de magnitude, à moi.** Son « 22,6 % de votre propre effet » compare
*son* grimpeur à R = 27 (0,0396) à *son* mélange. Ma borne publiée valait 0,1443,
obtenue par transpositions, et vaut **90 % de mon propre mélange (0,1604)**. Le
sous-comptage est de 11 %, pas d'un facteur 4. La structure de son argument tient,
son chiffre ne décrit pas mon chiffre.

**Sa question : l'effet baisse-t-il quand β monte, ou R cesse-t-il de le prédire ?**
150 runs, 5 β × 30 graines, `effet_par_beta.py`, sur des codes **réellement
émergents** et non sur un pire cas de recherche.

| β | 0,005 | 0,010 | 0,020 | 0,030 | 0,037 |
|---|---|---|---|---|---|
| R moyen | 24,57 | 24,53 | 25,10 | 24,87 | 25,10 |
| écart observé | 0,0077 | 0,0094 | 0,0100 | 0,0175 | 0,0106 |

| R | 27 | 26 | 25 | 24 | 23 |
|---|---|---|---|---|---|
| écart observé | 0,0096 | 0,0120 | 0,0139 | 0,0076 | 0,0106 |
| n | 8 | 30 | 53 | 47 | 12 |

**Réponse : ni l'un ni l'autre.** L'effet ne baisse pas avec β (corrélation
**+0,158**, aucune tendance au-delà du bruit), et R n'a **jamais** prédit l'écart
observé — corrélation **+0,091**, et **4 % de variance expliquée**. Conditionner
sur R ne retire presque rien : la corrélation résiduelle avec β reste à +0,144
contre +0,158 sans conditionnement. Ce n'est pas que R *cesse* d'être la variable,
c'est qu'il ne l'a jamais été pour cette quantité-là.

**Et la distinction que la mesure impose, qu'aucun de nous deux n'avait posée.**

> Le **pire cas atteignable** sous un plancher R et l'**écart réellement produit**
> par la dynamique sont deux fonctions différentes de R. Le premier est fortement
> monotone — 0,0526 à R = 27 contre 0,2152 à R = 23. Le second est plat.

Rapport entre les deux : **15,2**. Les codes émergents ne s'approchent nulle part
du pire cas qu'une recherche adverse atteint dans le même régime.

Son mélange reste la bonne construction **pour une borne**, et sa réserve sur β se
règle par le calcul : mélangé sur le p(R) propre à chaque β, le pire cas va de
0,1564 à 0,1773, soit **0,0209 d'amplitude sur toute la plage**. Le déplacement de
p(R) avec β existe et il est petit. Ce qui reste vrai, et c'est le fond : une borne
doit être mélangée sur le régime visité, et une mesure d'effet ne doit pas être
lue sur la même table.

### 7.25bis Le seul contraste au-delà de deux sigma n'a pas survécu au second tirage

Même échange, tour suivant. Il relève dans le tableau par R un contraste qui dépasse
deux sigma — R = 25 contre R = 24, écart 0,0063, t = 2,43 — et note que sur dix
contrastes disponibles c'est exactement ce qu'on trouve en regardant. Il ne le
croit pas, et demande trente graines de plus dans ces deux cellules.

*(Deux corrections apportées par §7.26 : le t vaut 2,43 et non 2,53, la valeur 2,53
venant des erreurs types intra-cellule et non de l'écart-type mis en commun ; et les
contrastes disponibles ce jour-là étaient vingt et non dix, dont cinq au-delà de deux
sigma. Corrigé ici plutôt qu'annoté plus bas, par la leçon de §7.24.)*

**Détail qu'il ne pouvait pas voir : R est une sortie du run, pas un réglage.** Ces
deux cellules portent 100 runs sur 150, donc trente tirages de plus coûtent environ
quarante-cinq runs. J'en ai lancé soixante, graine indépendante.

**Le signe s'inverse.** Sur les soixante nouveaux seuls : −0,0053, SE 0,0033,
**t = −1,60**. Sur les 210 réunis, tout va vers le nul et rien ne s'en éloigne :

| | n = 150 | n = 210 |
|---|---|---|
| R = 25 contre R = 24 | +0,0063, t = 2,43 | **+0,0028, t = 1,35** |
| η² | 4,11 % | **1,2 %** |
| F | F(4,145) = 1,552, p = 0,19 | **F(6,203) = 0,419, p = 0,87** |
| pente par unité de R | +0,001180, t = 1,12 | **+0,000507, t = 0,61** |

**Deux reformulations à lui, que j'adopte, et que la réplication rend plus fortes.**
« R explique 4 % » devient « **R n'est pas distinguable de n'expliquer rien du
tout** » — à n = 210, η² = 1,2 % et p = 0,87. Et son cadrage par la puissance
remplace mon rapport de 15,2 : la pente observée vaut +0,000507 ± 0,000836, la
détectable à 80 % vaut 0,00234, celle du pire cas −0,04065, soit **17,4 fois ma
propre résolution** — contre 13,9 avant réplication. Le rapport invitait l'objection
« ce sont deux échelles » ; la puissance l'interdit.

**Ce que ça généralise, et ce n'est pas sur R.** Le contraste au-delà de deux sigma
que j'avais retenu était très exactement celui à ne pas croire, et le moyen le moins
cher de le savoir était de **retirer**, pas d'en discuter. Quatrième fois de cet
échange que la réponse est un second tirage, et la première où je le fais avant de
publier plutôt qu'après.

*(Ce paragraphe disait « le seul contraste au-delà de deux sigma dans un tableau qui
en offrait dix ». Les deux nombres sont faux : cinq contrastes sur vingt. §7.26.)*

---

### 7.26 Neuvième critique : le tableau offrait vingt contrastes, et le plus grand n'était pas celui que j'ai lu

14/08/2026. Il met ma procédure sous le nul contre lequel elle argumentait. Loi du
maximum de dix contrastes par paires, sur mes cellules 8/30/53/47/12, 400 000
tirages : E[max |t|] = 1,620, q90 = 2,427, P(max |t| ≥ 2,40) = 0,107. Le contraste
qui nous a fait regarder deux fois portait donc un p corrigé de la sélection de 0,10,
contre 0,016 nominal. Six fois et demie moins cher qu'il ne se lisait.

**Trois routes vers sa loi, parce qu'une seule ne vaut rien ici.** Paramétrique à
cellules fixes comme lui ; par permutation des 150 écarts sur les étiquettes du plan
réel, qui ne suppose ni normalité ni écart-type commun ni effectifs fixes ; et
paramétrique avec effectifs retirés au sort, puisque R est une sortie du run et que
8/30/53/47/12 est lui-même une réalisation.

| | E[max \|t\|] | q90 | P(≥ 2,40) |
|---|---|---|---|
| le sien | 1,620 | 2,427 | 0,1066 |
| σ connu | 1,619 | 2,427 | **0,1066** |
| σ réestimé à 145 ddl | 1,628 | 2,452 | 0,1130 |
| permutation sur le plan réel | 1,624 | 2,428 | 0,1069 |
| effectifs retirés au sort | 1,623 | 2,439 | 0,1100 |

Sa loi est juste, et sa ligne est ma ligne à σ connu aux trois chiffres sur les trois
colonnes. Le seul écart visible est qu'un σ réestimé a des queues un peu plus lourdes
qu'un σ connu : 0,025 sur le q90 et 0,006 sur le P. Les effectifs retirés au sort
bougent moins que ça, donc l'approximation à cellules fixes qu'il était obligé de
faire est gratuite — ce qui valait la peine d'être vérifié, puisque le fait que R
soit une sortie était la seule chose qu'il ne pouvait pas contrôler de l'extérieur.
Les magnitudes tombent
aussi : E[|d| du gagnant] 0,00658 et 0,00662 contre son 0,00657 ; E[|d| sachant que
le gagnant est la paire 25/24] 0,00415 et 0,00421 contre son 0,00414. Cette paire
gagne 13,7 % du temps, c'est pourquoi elle gagne pour moins cher.

**Trois de mes chiffres tombent, tous dans son sens.**

*Mon t vaut 2,43, pas 2,53.* Avec l'écart-type mis en commun sur les cinq cellules —
l'estimateur qu'utilise un test de contraste — c'est 2,430, et l'écart-type vaut
0,012969 contre son 0,012942. Mon 2,53 était la version intra-cellule, celle dont je
lui avais moi-même écrit qu'elle était le moins bon estimateur à n = 8 avant de
l'utiliser pour le chiffre de tête. Son 2,40 lu sur le tableau était plus près du
vrai que le mien.

*Sa mise en commun compare deux procédures différentes.* Mon +0,0028 n'est pas une
pondération par variance inverse de deux moitiés : c'est le contraste relancé sur les
210 runs bruts, qui réestime conjointement moyennes et écart-type. Mis en commun
comme son nul le fait, mes deux estimations indépendantes donnent **+0,00206, SE
0,00207, t = 1,00**, avec 63,5 % du poids sur la découverte. Contre son nul
conditionné sur la paire (+0,00263), P = 0,634 ; contre le nul non conditionné
(+0,00418), P = 0,822. Pas 0,434. **Mon nombre mis en commun est au 63ᵉ–82ᵉ centile
d'un tableau où il n'y a rien.** Seule la réplication seule reste publiable :
−0,0053, SE 0,0033. Et le t = 1,47 publié à n = 210 est lui aussi la version
intra-cellule : mis en commun, d = +0,00285, **t = 1,35**.

*Son ω² est exact et non approché.* Sous le nul, η² suit exactement une loi bêta de
paramètres ddl1/2 et ddl2/2, donc son espérance vaut exactement ddl1/(ddl1+ddl2).
Ses 2,68 % et 2,87 % sont 4/149 et 6/209 à la décimale — vérifié sur 400 000 tirages,
2,683 % et 2,867 %. À n = 150 le chiffre débiaisé valait déjà +1,45 % et non 4,11 % ;
à n = 210 il vaut **−1,69 %**, c'est-à-dire que mes sept niveaux de R sont moins
structurés qu'une partition au hasard des mêmes 210 runs.

**Ce que ni lui ni moi n'avions vu.** Il corrige pour dix contrastes. **Le tableau en
offrait vingt.** Les mêmes 150 runs portent une ligne beta, cinq niveaux, dix
contrastes de plus, imprimés dans la même réponse et lus le même après-midi. Son plus
grand vaut **|t| = 2,968** — plus grand que celui sur lequel nous discutions depuis
deux tours.

| | contraste | d | t |
|---|---|---|---|
| beta | 0,005 contre 0,03 | −0,00981 | **−2,968** |
| beta | 0,010 contre 0,03 | −0,00806 | −2,439 |
| R | 25 contre 24 | +0,00631 | +2,430 |
| beta | 0,020 contre 0,03 | −0,00744 | −2,253 |
| beta | 0,030 contre 0,037 | +0,00691 | +2,093 |

**Cinq contrastes au-delà de deux sigma, pas un.** Et le plus grand des cinq est dans
la ligne où j'avais écrit « aucune tendance au-delà du bruit sur la ligne elle-même »,
phrase posée sans test alors que l'omnibus de cette ligne valait F(4,145) = 2,595,
p = 0,039. Corrigé sur les vingt par permutation, mon contraste R passe de 0,10 à
**0,200** ; celui de beta est à 0,053.

**La ligne beta meurt de la même façon.** Les soixante runs indépendants la tranchent
aussi, et je ne l'avais jamais regardée : le contraste 0,005 contre 0,03 passe de
−0,00981 (t = −2,97) à −0,00135 (**t = −0,34**), et l'omnibus de F(4,145) = 2,595,
p = 0,039 à F(4,55) = 1,790, **p = 0,144**. Moyennes par cellule, découverte puis
réplication, sur 0,005 / 0,010 / 0,020 / 0,030 / 0,037 : 0,0077 0,0094 0,0100
**0,0175** 0,0106, puis 0,0087 0,0099 **0,0024** 0,0101 0,0121. Le pic que j'aurais
décrit comme « beta = 0,03 ressort » a disparu, et la cellule qui s'effondre au second
tirage est une autre. Le p = 0,029 à n = 210 est à 71 % de la découverte et ne
confirme rien. **J'ai eu raison sur cette ligne en ne regardant pas assez**, ce qui
est pire que l'erreur sur R : là au moins j'avais écrit le nombre.

**Sa question, et la réponse que je ne voulais pas donner.** Il demande si le carnet a
une règle sur le nombre de contrastes qu'un tableau offre avant qu'on ait le droit de
lire le plus grand, ou si c'est décidé par tableau après l'avoir vu. C'est décidé par
tableau après l'avoir vu, il n'y a jamais eu de règle, et la ligne beta le démontre :
j'ai regardé les deux lignes, jugé que celle en R méritait un contraste et celle en
beta une phrase, et retenu **le plus petit des deux maxima**. Une décision prise par
tableau après avoir vu le tableau ne garantit même pas qu'on en sélectionne le plus
grand élément.

**Règle adoptée, quatre lignes, avant la prochaine graine.** *K se déclare avant les
données* : tout contraste par paires que le plan offre, sommé sur tout facteur lu dans
la même séance, rapporté ou non — vingt ici, et les dix que je n'ai pas rapportés sont
ceux qui le font vingt. *Tout |t| ≥ 2 s'imprime*, pas le plus grand : cinq lignes ici,
et en taire quatre est ce qui m'a permis d'appeler une ligne plate pendant qu'elle
portait un 2,97. *Le p corrigé vient d'une permutation de la sortie sur les étiquettes
du plan* : quelques secondes, aucune hypothèse, et c'est l'arbitre quand la version
paramétrique et celle du lecteur divergent, comme ci-dessus de 0,025 sur le q90. *Un
contraste sélectionné ne se publie jamais mis en commun avec sa propre réplication* —
la réplication seule, avec son erreur type.

Pour ce plan la règle donne un seuil qu'il vaut la peine d'écrire, parce qu'il n'est
pas voisin de celui que j'utilisais : sur les vingt contrastes, q90 = 2,73,
**q95 = 2,99**, q99 = 3,56, contre 1,98 nominal.

*(Ce paragraphe se terminait par « tout mon tableau était sous le q90 corrigé ».
Faux : le maximum du tableau vaut 2,968 et le q90 vaut 2,73. Mon propre p de 0,053,
écrit deux lignes plus haut, le disait déjà — 0,053 tombe entre q95 et q90 par
construction. Relevé au dixième tour, §7.27. Et la règle 1 telle qu'écrite ici — « tout
facteur lu dans la même séance » — fait dépendre le p publié de mon défilement ; elle
est remplacée par Scheffé en §7.27.)*

**Sur le « zéro run ».** Il a raison que la loi du max de K et E[η²] sont des fonctions
du plan, calculables avant la première graine, et il concède lui-même que le second
tirage est la réponse la plus forte. Je durcis dans le sens qui me coûte : disponible
*avant la première graine*, ce calcul change le plan et pas seulement la lecture —
savoir que cinq niveaux de R et cinq de beta exigent |t| = 2,99 m'aurait forcé soit à
nommer un contraste d'avance, soit à budgéter la réplication dès le lancement. Je n'ai
fait ni l'un ni l'autre parce que je ne savais pas que je choisissais. Ce que la route
à zéro run ne pouvait pas faire : m'apprendre que le signe s'inversait, ni que la ligne
beta mourait aussi. C'est le moyen le moins cher de savoir qu'il ne faut pas croire un
nombre ; ce n'est pas un moyen de savoir ce que vaut le nombre.

Code : `src/test3_communication/correction_de_selection.py` et
`correction_de_selection_suite.py`. Réponse dans `docs/REPONSE_ORDRE10.md`.

### 7.27 Dixième critique : R n'est pas un facteur, c'est l'objectif divisé par 27

15/08/2026. `results_test3/` est gitignoré, mais le générateur est semé de bout en
bout et `monter()` ne tire jamais : il régénère mes 150 runs au bit près avec
`--graine 0`, puis lance mes propres scripts dessus. Tout reproduit — écart-type
0,012969, t = 2,430, les quatre routes de la loi du max, +0,00631 / −0,00535 /
+0,00206, 0,822 et 0,634, le p corrigé 0,2002, ω² à +1,45 % et −1,69 %. Ce sont mes
runs recalculés, pas une expérience à lui.

**Ma phrase était fausse et mon propre p le disait.** J'avais écrit « tout mon tableau
était sous le q90 corrigé » deux lignes sous un p de 0,053. Or 0,053 tombe entre q95
et q90 par construction, donc le nombre que je venais de publier contredisait la
phrase que j'écrivais dessus. 2,968 n'est pas sous 2,73.

**Son bump, vérifié.** Retrait d'une cellule à la fois sur les vingt-cinq : celle à
beta = 0,03, R = 25, treize runs, fait passer t(R) de 2,430 à **1,134** et t(beta) de
−2,968 à −1,250. Les cinq écarts internes à beta tombent aussi à la décimale : le
contraste R vaut +0,00003, +0,00284, +0,00161, **+0,02013**, +0,00267 par niveau de
beta. Un niveau à +0,0201 et quatre à +0,0022 en moyenne.

**Sa diagnostique est elle-même un maximum sur vingt-cinq retraits**, ce qui est
exactement la faute qu'il m'a apprise un étage plus haut. Loi nulle de la chute
maximale par permutation : E = 0,430, q99 = 1,070, chute observée **1,295**,
P = 0,0008. **Elle survit à sa propre correction.** Je suis allé la chercher et elle
n'est pas là.

**Ce qu'il a choisi.** Il annonce « les 19 autres cellules déplacent t(R) d'au plus
0,27 ». Il y en a vingt-deux, et le retrait de beta = 0,005, R = 25 déplace t de
**0,517** — plus que deux des trois cellules qu'il montre. Vers le haut, 2,430 →
2,947, ce qui explique probablement son absence de la liste. Son 0,27 est juste pour
les déplacements vers le bas seulement. Même forme que mon erreur : une phrase qui
couvre les lignes non montrées, fausse dans le sens qui sert le propos.

**Quatre crans plus loin.**

*Ce ne sont pas treize runs, ce sont deux.* Nombre de rupture — plus petit nombre de
runs dont le retrait adverse fait passer sous 1,98 : **2 sur 150 (1,3 %)** pour le
contraste R, 3 sur 150 pour celui de beta. Et la cellule n'est pas une cellule, elle
est bimodale : 0,0496 0,0488 0,0477 0,0445 puis 0,0253 0,0245 … Le saut vaut 0,0192
là où le plus grand saut ailleurs dans la cellule vaut 0,0084. Les quatre premiers
ont un max de concentration moyen de 0,182, au-dessus du q90 du plan entier (0,171) ;
les neuf autres sont à 0,123 et la médiane du plan vaut 0,124. **Quatre runs
inhabituels posés sur neuf parfaitement ordinaires.**

*L'interaction qu'il nomme a un test, et il ne passe pas sa propre barre.* Ajustement
des moyennes de cellules contre le modèle additif : **F(11,127) = 1,748**, p nominal
0,070, p par permutation 0,075 — avant toute correction de multiplicité. Et sur les
soixante runs indépendants, F(6,40) = 1,425, p = 0,229, avec la cellule qui s'inverse
de **+0,0201 à −0,0088**.

*R est l'objectif.* Le fichier porte une colonne que ni lui ni moi n'avions ouverte :
la récompense finale. Sur les 150 runs, |récompense − k/27| < 10⁻³ pour un entier k
dans **150 cas sur 150**, avec k = R dans 141 et k = R − 1 dans 9. corr(R, récompense)
= **+0,9725**. Chaque cellule du tableau a une seule valeur de récompense à cinq
décimales : 0,92586 sur les treize runs de la cellule du bump, 0,88882 une colonne
plus loin. **Stratifier par R, c'est stratifier les runs par la valeur de l'objectif
que l'optimiseur maximisait**, puis demander si un biais de mesure diffère entre ceux
qui ont fait 25/27 et ceux qui ont fait 24/27. Le bump n'est pas la raison pour
laquelle cette ligne n'aurait pas dû être publiée : elle n'aurait pas dû l'être même
si toutes les cellules avaient été plates.

*La grandeur est une fonction du max, pas de R.* corr(max, écart) = +0,4317, contre
corr(appariée, écart) = +0,0665 et corr(max, R) = +0,0172. L'écart mesure surtout où
le max non contraint est tombé, et cela n'a rien à voir avec R. C'est le mécanisme
sous le bump : un amas de runs à max élevé tombé dans une cellule d'un tableau indexé
par quelque chose d'orthogonal — ce qui explique aussi qu'il s'inverse au tirage
suivant, rien ne le tenait.

*Pour la ligne beta, qui reste légitime :* beta ne déplace pas R. χ²(16) = 15,67 à
p = 0,476 sur le croisement complet. Composante linéaire faible, corr = +0,197 à
p = 0,016, donc médiation marginale. Beta est un facteur que j'ai réglé, ses
problèmes restent ceux déjà au dossier : p corrigé 0,053, Scheffé 0,072, t = −0,34 en
réplication.

**Sa question, et pourquoi j'en réponds une autre.** Il demande s'il existe une règle
disant quand un contraste a le droit d'être rapporté comme un fait sur le facteur qui
l'étiquette plutôt que sur la seule cellule où il vit. La question présuppose que
l'étiquette est un facteur, et pour cette ligne elle ne l'est pas.

**Règle 5. Un contraste n'est un fait sur une colonne que si la colonne a été
réglée avant le run.** Pas mesurée, pas dérivée, pas « une sortie qui indexe
commodément » : assignée. R échoue au niveau le plus fort possible, étant l'objectif
sur une grille. Aucune diagnostique n'aurait pu sauver cette ligne, et aucune n'était
nécessaire : elle est disqualifiée avant l'arrivée des données, en lisant le
générateur. **Ça supprime toute la ligne R de §7.25 et §7.25bis**, ce qui est plus que
ne fait son bump, et c'est la première règle qui me coûte un résultat que je croyais
encore plutôt qu'un que j'avais déjà lâché.

**Règle 6, pour les colonnes qui passent la règle 5 : publier le nombre de rupture.**
Plus petit nombre de runs dont le retrait adverse fait passer sous la barre. Un entier,
aucune loi, aucune famille, quelques secondes, et il se moque de savoir si la
fragilité est une cellule, une valeur aberrante ou un amas. Ici 2 sur 150 et 3 sur
150. Un contraste qui meurt à 1,3 % de l'échantillon se rapporte avec cet entier
collé dessus, ou ne se rapporte pas.

Aucune des deux n'est une règle sur la sélection. Il a raison que mes quatre
premières l'étaient toutes, et c'était le défaut : j'avais construit quatre façons de
tarifer un contraste et aucune pour demander si la colonne méritait un tarif.

**Et la règle 1 avait bien un journal intime dedans.** « Tout facteur lu dans la même
séance » fait dépendre le p publié de mon défilement. Scheffé à 145 ddl demande 2,817
à famille 0,10 et donne 0,213 à mon contraste R contre mon 0,200 dépendant de la
séance, 0,072 à celui de beta contre 0,053. Mêmes nombres, fixés par le plan avant la
première graine, sans journal. La règle 1 devient : **K est celui du plan, et la barre
est celle de Scheffé.**

Sixième fois de cet échange que la réponse était déjà sur le disque. Cette fois
c'était une colonne du même fichier, à douze caractères de celle dont nous discutions.

Code : `src/test3_communication/anatomie_du_bump.py`. Réponse dans
`docs/REPONSE_ORDRE11.md`.

### 7.28 Onzième critique : le plancher de détection, et le seuil que personne n'a écrit

15/08/2026, même soir. Il attaque les deux règles nées la veille. La 6 publiait un
entier nu — « meurt à 2 runs sur 150 » ne veut rien dire tant qu'on ne sait pas à
combien de runs meurt un effet **vrai** de cette taille. Il plante donc l'effet et
mesure : résidus des moyennes par niveau de R, rééchantillonnés, décalage constant
sur les runs à R = 25 calibré pour E[t] = 2,430, effectifs 53 et 47.

**La règle 6 meurt plus salement qu'il ne la tue.** Je suis allé construire ce nul et
il n'est pas identifié. Même plan, même effet planté, même n, même retrait glouton ;
seule varie la provenance des résidus, qui ne devrait rien changer :

| source des résidus | médiane | moyenne | P(≤2) | P(≤3) | puissance |
|---|---|---|---|---|---|
| moyennes par niveau de R | 3 | 4,3 | 0,488 | 0,568 | 0,682 |
| moyennes des 25 cellules | 4 | 5,6 | 0,354 | 0,432 | 0,796 |
| gaussiens de même sd | 2 | 4,0 | 0,518 | 0,598 | 0,626 |
| les siens | 4 | 5,5 | 0,252 | 0,398 | 0,665 |

Sa médiane et sa moyenne tombent sur ma ligne « par cellule », sa puissance sur ma
ligne « par niveau de R ». Aucune spécification unique ne produit ses quatre nombres,
et je le rapporte plutôt que de régler jusqu'à ce que ça colle. **L'écart entre les
trois est le résultat** : le 2 observé se lit comme la médiane d'un nul, « la moitié
des effets vrais » dans un deuxième, « un quart » dans un troisième, et rien ne les
départage qu'un choix de modélisation sans rapport avec l'effet planté — l'asymétrie
des résidus vaut 1,37 par niveau de R contre 0,96 par cellule, et ce seul écart
déplace la référence d'un facteur deux. **Retirée, pas amendée** : c'était une
statistique de robustesse avec le modèle caché dans l'étape de calibration, ce qui
est la même faute qu'un entier nu avec le modèle caché dans la tête du lecteur.

**Son chiffre surprenant est son p en costume.** Il annonce « le plan a deux tiers de
puissance à la taille qu'il a trouvée, 0,665 ». La puissance calculée à la taille
d'effet **observée** est une fonction biunivoque décroissante du p (Hoenig et Heisey
2001) : elle ne peut rien contenir que t ne contienne déjà.

| t | p bilatéral | puissance à la taille observée |
|---|---|---|
| 1,98 | 0,0496 | 0,503 |
| **2,43** | **0,0163** | **0,675** |
| 2,97 | 0,0035 | 0,839 |
| 3,50 | 0,0006 | 0,935 |

C'est un fait sur l'observation, pas sur le plan, et la phrase l'attribue au plan.
Son *usage* est légitime — calibrer un effet planté à la taille observée n'est pas la
même chose. C'est la phrase isolée qui ne tient pas.

**Sa flèche, acceptée.** `R = len(np.unique(code))` est la taille d'alphabet du code
argmax, et la récompense est le succès aller-retour sur les mêmes 27 référents : R
symboles n'admettent au plus que R référents décodables. Dénombrement, pas régression.
recompense ≤ R/27 dans **150/150 et 60/60**, déficit dans {0, 1}, jamais 2, jamais
négatif sur les 210. Mon « R est la récompense fois vingt-sept » décrivait les 141
runs où la borne est serrée et laissait tomber en silence les neuf où elle ne l'est
pas — or ce sont les informatifs, ceux où le code porte R symboles et où R − 1 seulement
décodent. La direction fait mordre la règle 5 plus fort : une colonne qui **borne**
l'objectif est disqualifiée aussi dans les runs où l'optimiseur rate le plafond.

Un point qu'il n'a pas vérifié et que j'attendais déterminant : **aucun des neuf runs
à collision n'est dans la cellule du bump**, zéro sur treize, et leur écart moyen vaut
0,01142 contre 0,01102 pour les autres. La collision de récepteur n'est le mécanisme
de rien.

**Le plancher de détection, qui remplace la colonne des p.** Quantité du plan seul,
2,80 × SE :

| contraste | n | SE | plancher | observé | observé/plancher |
|---|---|---|---|---|---|
| R 27 contre 26 | 8+30 | 0,00516 | 0,01445 | −0,00249 | 0,17 |
| R 26 contre 25 | 30+53 | 0,00296 | 0,00830 | −0,00186 | 0,22 |
| R 25 contre 24 | 53+47 | 0,00260 | 0,00728 | +0,00631 | **0,87** |
| R 24 contre 23 | 47+12 | 0,00419 | 0,01175 | −0,00298 | 0,25 |
| beta 0,005 contre 0,03 | 30+30 | 0,00330 | 0,00925 | −0,00981 | **1,06** |
| beta 0,005 contre 0,037 | 30+30 | 0,00330 | 0,00925 | −0,00289 | 0,31 |

**Tout effet observé du tableau est à son plancher ou dessous**, et les deux qui ont
occupé quatre tours sont ceux à 0,87 et 1,06 : la définition du régime où ce qui
émerge est gonflé. Le plancher se calcule avant la première graine, ne demande aucune
donnée et ne se truque pas.

Pour porter le contraste beta à 90 % de puissance : **36 graines par niveau, 179
runs**, une demi-heure — commandable. Pour celui en R : 89 par cellule, 266 runs, mais
les cellules R ne se règlent pas. C'est la règle 5 sous un autre costume : **la
colonne qu'on ne peut pas alimenter en puissance est celle qu'on n'avait pas le droit
de contraster.**

**Ce que ni la puissance ni le plancher ne posent.** Le plancher répond « qu'aurais-je
pu voir ». Onze tours n'ont jamais demandé « qu'aurait-il fallu voir pour que quelque
chose change ». Le tableau de §7.25 existait pour savoir si l'écart produit par la
dynamique approche le pire cas atteignable : 0,1443 contre 0,0110, rapport 13,1.

| rapport visé | hausse nécessaire | contre le plus grand effet observé |
|---|---|---|
| 2 | 0,0611 | 6,2 × |
| 5 | 0,0178 | 1,8 × |
| 8 | 0,0070 | 0,7 × |

**Je m'arrête là plutôt que de prendre le chiffre qui m'arrange.** Le seuil est
choisi, pas dérivé, et le verdict bascule entre le rapport 5 et le rapport 8. Si
« nulle part près » veut dire un facteur deux, tout le tableau est six fois sous ce
qui pourrait compter. Si ça veut dire un facteur huit, les contrastes sont dans la
plage et l'exercice était légitime. Je ne peux plus nommer ce seuil sans qu'il ait
l'air choisi, **et c'est le résultat** : c'est le seul nombre de tout cet échange qui
devait être fixé avant les données et ne l'a pas été. Le plancher reste calculable
honnêtement aujourd'hui ; le seuil de pertinence devient incalculable dès qu'on a vu
les résultats.

Code : `src/test3_communication/plancher_de_detection.py`. Réponse dans
`docs/REPONSE_ORDRE12.md`. Les cinq questions de fond que cet échange dessine sont en
§8ter.

### 7.29 Douzième critique : le plancher était un p redimensionné, et la variable n'est pas continue

15/08/2026, tard. Il trouve une identité exacte dans le code que j'ai publié une
heure plus tôt. `plancher = 2,80 × se` et `t = d / se` dans le même fichier, donc la
colonne « observé / plancher » **est** |t| / 2,80, identiquement, avant qu'aucune
donnée n'existe. Vérifié sur mes six lignes : écart 0,00e+00 partout.

Donc « tout effet observé est à son plancher ou dessous » est |t| < 2,80, soit
**p > 0,0058** à 145 ddl : un alpha **8,6 fois plus strict** que le 0,05 inscrit dans
la définition du plancher. Et j'avais publié le même fait deux fois dans le même
message — la section C imprimait t = 2,97 → p = 0,0035, qui est la ligne à 1,06 de la
section D — en n'en signalant qu'un. **J'ai remplacé une colonne de p par une colonne
de p redimensionnée un message après avoir expliqué que la puissance observée est un
p redimensionné.** Colonne retirée. Correction à la question 3 de §8ter : « est-ce
une fonction du plan » se vérifie sur l'expression imprimée, pas sur l'intention.

**Le plancher absolu survit, et il condamne quelque chose de plus ancien.** Son usage
prospectif est la vraie trouvaille, et il remonte plus loin que la ligne où il
l'applique :

| | n | plancher | effet à confirmer | rapport |
|---|---|---|---|---|
| beta .005/.03 découverte | 30+30 | 0,00925 | 0,00981 | 1,06 |
| beta .005/.03 **réplication** | 12+12 | **0,01462** | 0,00981 | **0,67** |
| R 25 vs 24 découverte | 53+47 | 0,00728 | 0,00631 | 0,87 |
| R 25 vs 24 **réplication** | 30+12 | **0,01240** | 0,00631 | **0,51** |

**Les soixante runs de §7.25bis ne pouvaient confirmer aucun des deux contrastes,
même vrais.** Je les ai rapportés comme un test — « le signe s'inverse », « ça ne
réplique pas », « l'omnibus passe à p = 0,144 » — alors que ce n'était qu'un
estimateur. La conclusion survit ; la raison que j'en donnais, non : j'ai lu un échec
à franchir une barre que le tirage ne pouvait pas franchir.

**Sa solution échoue ici, et pour une raison structurelle.** Il propose
`2,80 × σ_pilote × √(1/na + 1/nb)` dans le document de conception. Or σ n'est pas
transférable : Bartlett χ² = 19,176 à **p = 0,0007** entre niveaux de beta, rapport
des sd 2,07, et un σ commun mésestime le plancher de 12 % à 38 % selon la cellule.
Le mécanisme est ce qui compte : **l'écart est borné par zéro en bas** — le max non
contraint est toujours au moins la valeur appariée — et sur les 18 cellules à n ≥ 4,
corr(moyenne, sd) = **+0,874** Pearson, +0,917 Spearman, pente 0,82, CV médian 1,07.
**σ n'est pas une échelle de nuisance, c'est à peu près la grandeur mesurée.** Un
pilote ne fixe le plancher que si sa moyenne coïncide avec celle du run, c'est-à-dire
seulement si on connaît déjà l'effet.

**Mais plus de la question 4 remonte avant le run qu'il ne le proposait, dans une
autre unité.** Si sd ≈ CV × moyenne avec CV stable près de 1, alors
plancher/moyenne = 2,80 × CV × √(1/na + 1/nb), qui ne demande aucun σ. À 30 graines
par cellule et CV ≈ 1,2 : **0,89**, vérifié contre le tableau réel (0,00925 / 0,01035
= 0,894). Écrit avant la première graine, en une ligne, sans pilote : *à trente
graines par cellule, ce plan voit un quasi-doublement de l'écart et rien de plus
petit.* C'est la forme honnête pour toute grandeur positive massée en zéro, et c'est
**CV** qu'il faut piloter, pas σ.

**Sa reformulation de la question 4, que j'adopte.** Il écrit que le seuil de
pertinence est devenu inécrivable *par moi*, ce qui n'est pas la même chose
qu'incalculable : le défaut n'est pas que les données soient arrivées d'abord, c'est
que « qu'est-ce que cette mesure alimente » n'a jamais été posé — et cette
question-là ne référence pas les résultats, donc elle reste répondable aujourd'hui à
pleine honnêteté. Il a raison, et c'est un meilleur diagnostic que le mien.

Alors je l'ai posée, et **elle a une réponse datée.** L'écart max − appariée mesure
l'inflation due à publier la statistique de concentration sous sa forme argmax non
contraint plutôt qu'appariée. Son consommateur était le seuil de 0,35 de TEST3 §6.1.
**Ce seuil a été retiré le 11/08/2026, §1.9.** Les tours six à douze ont tarifé une
mesure dont le consommateur avait été supprimé au tour cinq. Distinction qui garde
ça honnête : la **borne** a encore un consommateur, puisque je publie la forme max et
qu'un lecteur doit savoir qu'elle peut être gonflée de 0,14. Le **tableau de
contrastes** n'en a jamais eu — aucune décision ne change à aucune valeur de cette
dépendance, ce qui est exactement pourquoi le rapport de pertinence était libre de
valoir 2 ou 8.

Le plan le dit aussi, si on l'interroge sur la question qui avait un consommateur.
Celle-là est à un échantillon, pas un contraste : moyenne 0,01035, SE 0,00085, IC 95 %
[0,00869 ; 0,01201], distance au pire cas **158 erreurs types**, plancher à un
échantillon 0,00237 contre 0,00728–0,01445 pour les contrastes. **Les mêmes runs sont
trois à six fois plus fins sur la question à consommateur que sur les contrastes qui
n'en avaient pas**, et ils y avaient répondu à 158 sigma avant que tout ceci commence.

**Règle 7 : quand une affirmation est retirée, lister toute mesure dont elle était
l'unique consommateur, et arrêter de la mesurer.** Une rétractation se propage vers
l'aval et rien dans mon processus ne la faisait se propager. §1.9 a tué le seuil ; la
grandeur qu'il justifiait a continué d'être mesurée, contrastée, corrigée pour la
multiplicité, répliquée, re-corrigée et défendue sur sept tours — tout cela correct,
rien de tout cela rattaché à quoi que ce soit. Contrairement aux six précédentes, elle
ne coûte rien à exécuter et ne se truque pas : c'est une liste, écrite au moment du
retrait.

Code : `src/test3_communication/plancher_de_detection.py` et `masse_en_zero.py`.
Réponse dans `docs/REPONSE_ORDRE13.md`.

### 7.29bis Ce que ni lui ni moi n'avions ouvert : la variable n'est pas continue

Trouvé en vérifiant son pilote, le même soir, et c'est la plus grosse chose du
fichier. **63 des 210 runs ont un écart exactement nul.** Trente pour cent. L'écart
vaut zéro précisément quand l'argmax non contraint est déjà une bijection : aucune
position de message ne réclame l'attribut qu'une autre a réclamé. Ce n'est donc pas
une grandeur continue avec un plancher, c'est **une masse ponctuelle plus une partie
positive asymétrique** — et tous les t, toutes les permutations, toutes les barres de
Scheffé et tous les bootstraps de douze tours l'ont traitée comme n'étant ni l'un ni
l'autre.

**Première conséquence, sur le nombre que je défends depuis le septième tour.** La
borne 0,1443 vient du grimpeur par transpositions cherchant le **pire** code, qui a
nécessairement une collision d'argmax : c'est un pire cas **sachant collision**. Mon
0,0104 est non conditionnel, à 30 % de zéros. **Le rapport publié compare un mélange
à une conditionnelle.**

| quantité | valeur | rapport à 0,1443 |
|---|---|---|
| E[écart] sur 210 runs (publié) | 0,01035 | **13,9** |
| E[écart \| écart > 0] — comparaison appariée | 0,01479 | **9,8** |
| médiane des écarts > 0 | 0,01254 | 11,5 |
| q95 des écarts > 0 | 0,04049 | 3,6 |
| maximum observé sur 210 runs | 0,05927 | **2,4** |

Le rapport que je cite depuis quatre tours vaut 13,9 contre 9,8 apparié — et si l'on
compare les deux objets de même nature, un pire cas contre un pire cas, **2,4**.
« Les codes émergents n'approchent nulle part ce qu'une recherche adverse atteint »
reposait sur l'appariement qui l'arrange, et j'ai choisi cet appariement sans
remarquer qu'il y en avait un à choisir.

**Deuxième conséquence : la grandeur en est deux, à consommateurs distincts.**
P(collision d'argmax) = 0,700 [IC95 0,633 ; 0,761] et E[inflation | collision] =
0,01479 (SE 0,00101), dont le produit vaut exactement la moyenne publiée 0,01035. À
quelle fréquence la statistique publiée est fausse, et de combien quand elle l'est.
Deux questions différentes pour un lecteur, et les moyenner n'en répond à aucune.
Jamais séparées dans aucune version du document.

**Troisième conséquence : les tests.** Asymétrie +1,34, et ni la variable ni son
logarithme ne passent Shapiro (1,7 × 10⁻⁹ et 2,2 × 10⁻⁶) — ce n'est donc pas non plus
une log-normale.

| contraste | Student brut | sur log(écart > 0) | Mann-Whitney |
|---|---|---|---|
| R 25 contre 24 | t = +2,462, p = 0,0156 | t = +1,471, **p = 0,1459** | **p = 0,0619** |
| beta .005/.03 | t = −2,589, p = 0,0122 | t = −2,113, p = 0,0405 | p = 0,0285 |

**Le contraste en R — quatre tours, une correction de sélection, une réplication, une
analyse de bump, un argument de dénombrement — passe à p = 0,062 dès qu'on le pose
sous une forme que la variable peut porter.** C'était en partie la machinerie
gaussienne lisant une masse ponctuelle comme de la donnée. Et la décomposition dit
que les contrastes ne portent pas sur le taux de collision : 13 % et 17 % viennent de
la proportion de zéros, Fisher p = 0,66 et 0,55. Ils portent sur la taille de
l'inflation quand elle a lieu, c'est-à-dire sur la moitié à plus petit n — 39 et 32
runs, pas 53 et 47.

**Un nombre que je ne revendique pas.** Le taux de zéros observé, 0,300, diffère de
6/27 = 0,2222 à p = 0,0098 binomial, 6/27 étant le taux de permutation si les trois
argmax étaient indépendants et uniformes. Ils sont les argmax d'informations
mutuelles corrélées, donc cette référence n'est pas un nul justifié et le p n'est pas
un résultat. C'est un nombre, imprimé comme tel.

**Leçon.** `min`, `max` et un compte de zéros exacts l'auraient attrapé au premier
tour, sans coûter un calcul ni demander un argument — et avant la comparaison à la
borne dont tout le reste découlait. Douze tours d'inférence de plus en plus correcte
sur une variable que personne n'avait tracée.

### 7.30 Treizième critique : la réponse était dans un champ nommé `inflation_moyenne_globale`, écrit le 11/08

16/08/2026. Il relève que j'ai lu un Fisher p = 0,66 sans son plancher — très
exactement la faute de tout l'échange, une section après avoir retiré la même lecture
pour la puissance observée.

**Ses trois jambes, vérifiées.** L'identité sur le CV du mélange est exacte sur mes
210 : CV_pos²/p = 0,9736, (1−p)/p = 0,4286, somme 1,4022 contre un CV_mélange² mesuré
à 1,4022. Donc **30,6 % du CV² qu'on écrirait dans un document de conception est le
taux de collision**, sans échelle dedans. Planchers relatifs : mélange à 30+30 =
0,8561, conditionnel à 21+21 = 0,7133, gain de 17 %. Fenêtres sur la proportion :
détectable seulement si le taux monte à 0,956 ou descend à 0,361 à 30+30 — soit
**15 % de la marge disponible vers le haut**. IC bootstrap sur la part portée par les
zéros : [1 %, 53 %] et [0 %, 56 %]. Une part dont l'intervalle va jusqu'à la moitié ne
soutient pas « ils portent sur la taille de l'inflation ». **Phrase retirée.**

**Une correction qui améliore sa solution.** Il veut préinscrire CV_pos. Il n'est pas
assez stable non plus : 0,57 à 0,95 entre niveaux de beta, rapport 1,67, Bartlett
p = 0,0004. **Le log stabilise** — Levene sur log de la partie positive p = 0,071
contre 0,020 en brut. La quantité préinscriptible est l'écart-type du log, pas le CV.

**Sa question, et le fichier où elle mène.** Il demande s'il existe une règle qui se
déclenche à la *création* d'une mesure, ou si la liste ne s'écrit jamais qu'à la
descente. Je suis allé l'écrire. Avant de l'écrire, j'ai cherché quel aurait été le
consommateur de cette mesure au 11/08 — et j'ai trouvé que **la question avait déjà
une réponse**.

`results_test3/loi_nulle_longue_n10000000_g0.json`, généré le 11/08/2026, dix
millions de codes. Bloc `double_compte` :

```json
"taux_global": 0.7464519,
"inflation_moyenne_globale": 0.010049794802284647,
"inflation_maximale": 0.10807050074977963
```

| quantité | nulle 10⁷ (11/08) | 210 runs (12–15/08) | |
|---|---|---|---|
| P(collision d'argmax) | 0,7465 | 0,7000 | binomial p = 0,13 |
| E[inflation] | **0,01005** | **0,01035** | z = 0,36 |
| E[inflation \| collision] | 0,01346 | 0,01479 | z = 1,32 |
| inflation maximale | 0,10807 | 0,05927 | |

**L'écart que j'ai mesuré, borné, contrasté, corrigé pour la multiplicité, répliqué,
défendu sur huit tours et décomposé hier comme une nouveauté est l'inflation propre
de la loi nulle.** Pas « proche de » : c'est elle.

Et le contrôle de premier ordre que personne n'a lancé non plus : Kolmogorov-Smirnov
des 210 concentrations max contre 20 000 tirages de la nulle, **D = 0,0508,
p = 0,638**. La distribution observée est sur la nulle à chaque quantile — 3 % sous
q5, 50 % sous q50, 95 % sous q95, 99 % sous q99. Ce que **mon propre §6.2 avait établi
le 11/08** sur la concentration appariée, à 100 graines : z = −0,0098 ± 0,1025,
KS p = 0,386. Je l'ai publié, puis j'ai passé huit tours à contraster une différence
entre deux statistiques calculées sur des codes que j'avais déjà montrés
nul-distribués.

**Troisième correction du rapport de tête en deux jours.**

| comparaison | rapport |
|---|---|
| moyenne observée contre pire cas de recherche (publié) | **13,9** |
| moyenne observée contre moyenne de la nulle | 0,97 |
| max observé sur 210 runs contre pire cas de recherche | 2,4 |
| **max de la nulle sur 10⁷ contre pire cas de recherche** | **1,34** |

Mon 15 était une **moyenne** contre un **maximum**, sur des tailles d'échantillon
séparées de cinq ordres de grandeur. La comparaison de même nature est le pire cas
qu'une recherche adverse trouve contre le pire que dix millions de tirages au hasard
produisent seuls : **0,1443 contre 0,1081**. La recherche fait 34 % de mieux que le
hasard, pas quinze fois. « Les codes émergents n'approchent nulle part ce qu'une
recherche adverse atteint » est faux tel qu'écrit — ils tombent exactement où
tombent les codes au hasard, et les codes au hasard arrivent au tiers de la recherche.

**Règle 8, et c'est l'autre sens que celui qu'il suppose.** *Quand un résultat est
établi, lister toute grandeur encore mesurée à laquelle il répond, et arrêter de la
mesurer.* La règle 7 se déclenche au retrait : une affirmation meurt, on liste ce
qu'elle alimentait — direction bon marché, et elle a attrapé une chose, le seuil de
0,35 mort le 11/08. La règle 8 se déclenche à l'établissement, et c'est elle qui
aurait tout arrêté le jour où ça a commencé : §6.2 a atterri le 11/08 en disant que
les codes émergents sont tirés de la nulle, et le fichier de la nulle, généré le même
jour, contenait l'inflation de cette nulle à sept chiffres sous un nom de champ qui
**est** la grandeur. Entre les deux, tout le programme §7.24–§7.29 était répondu avant
d'être lancé — la moyenne, le taux, la conditionnelle et le maximum, les quatre.

Personne n'écrit la liste d'établissement parce qu'établir un résultat donne le
sentiment de finir quelque chose, pas d'en contracter une obligation. Un retrait
s'annonce ; un résultat qui répond en silence à trois autres questions ouvertes
n'annonce rien, et j'en avais deux dans le même répertoire avec des horodatages
identiques.

**Ce qui survit.** La décomposition reste publiable, pour la raison inverse de celle
que j'avançais : P(collision) = 0,700 et E[inflation | collision] = 0,01479 ne
décrivent pas une propriété de ce que produit la dynamique, ils décrivent **la nulle,
reproduite par la dynamique** — ce qui est le vrai résultat et concorde avec tout
§6.1 à §6.6. L'inflation n'est pas une donnée sur les codes émergents, c'est un fait
sur la statistique, valable pour n'importe quel code. Et ça règle le seuil de
pertinence que je ne pouvais pas nommer deux tours plus tôt : il n'y avait rien à
choisir parce que la grandeur ne dépend de rien que l'expérience fasse varier. La
colonne honnête à côté n'est ni un p, ni un plancher, ni un seuil — c'est la valeur
de la nulle.

Treize tours. Chaque correction était juste, chacune plus fine que la précédente, et
la chose qu'elles affinaient toutes avait été calculée correctement le premier jour
et classée sous son propre nom.

Réponse dans `docs/REPONSE_ORDRE14.md`.

### 7.31 Quatorzième critique : le maximum publié était celui du premier cinquième

17/08/2026. Il lit `tirer` dans `loi_nulle_longue.py` et trouve que `inflation_maximale`
sort du réservoir, lequel cesse de se remplir à 2 000 000 tirages, puis s'imprime sous
« toute la loi ». Le maximum publié est donc celui du premier cinquième du tirage.

**Vérifié en relançant les 10⁷, même graine, même flux :**

| n | vrai max | max réservoir | n réservoir | max du pool |
|---|---|---|---|---|
| 100 000 | 0,097594 | 0,097594 | 100 000 | 0,097594 |
| 500 000 | 0,103746 | 0,103746 | 500 000 | 0,103746 |
| 2 000 000 | 0,108071 | 0,108071 | 2 000 000 | 0,108071 |
| 5 000 000 | 0,111111 | **0,108071** | 2 000 000 | 0,111111 |
| 10 000 000 | **0,122365** | **0,108071** | 2 000 000 | 0,122365 |

**13 tirages sur 10⁷ sont au-dessus du nombre que j'avais publié comme maximum.** Le
rapport d'hier passe de 1,34 à **1,18**. Et sa lecture du pool est juste sur les deux
points : il porte bien le vrai maximum à chaque jalon, et c'est de la chance — il est
retenu sur `conc_max`, l'inflation n'est pas monotone en `conc_max`, et la garantie
que le docstring de `quantile_exact` gagne pour les quantiles appariés n'existe pas
pour cette colonne. Corrigé dans la source : `inflation_max`, `inflation_moyenne`,
`taux_inflation`, `inflation_moyenne_si_positive` et un dict `inflation_depassements`,
tous accumulés dans la boucle. Vérifié contre son jalon : à 500 000 le code corrigé
rend 0,10374624404828912.

**Le second défaut du même bloc, qu'il n'a pas relevé.** `taux_global` n'est pas le
taux auquel j'ai comparé mes runs. Dans `statistiques`, `dc = ~distincts` compte
**P(deux positions partagent un argmax)** = 0,7464519. Sur les mêmes 10⁷ tirages,
**P(inflation > 0) = 0,6762074**. Sept points d'écart : dans 7 % des tirages l'argmax
collisionne et l'appariement égale le max exactement, donc la collision ne coûte rien.

Mon 0,700 observé est calculé comme `ecart > 0`. Le tableau d'hier comparait donc un
taux observé fondé sur le coût à un taux nul fondé sur la structure, et la moyenne
conditionnelle en héritait. Apparié :

| | nulle (comme publiée) | nulle (appariée) | observé | |
|---|---|---|---|---|
| P(collision qui coûte) | 0,74645 | **0,67621** | 0,70000 | p = 0,507 |
| E[inflation] | 0,01005 | 0,01005 | 0,01035 | z = 0,36 |
| E[inflation \| > 0] | 0,01346 | **0,01486** | 0,01479 | **z = −0,07** |

La ligne conditionnelle passe de z = 1,32 à **z = −0,07**, et le taux de p = 0,13 à
p = 0,51. **La seule ligne du tableau d'hier qui ne tombait pas sur la nulle était un
mélange de définitions à moi**, et la corriger rend l'accord exact sur les trois. Les
deux taux sont maintenant exportés séparément.

**La clôture, qui est tout ce que huit tours de contrastes ont acheté.** La nulle ne
contient que des bijections — `np.argsort` d'un vecteur aléatoire est une permutation,
donc R = 27 pour les 10⁷ tirages. Neuf de mes 210 runs le sont ; le R médian observé
vaut 25. Comparer mes runs à cette nulle exige donc que l'inflation ne dépende pas de
R, et c'est exactement le contraste sur lequel huit tours ont porté : Welch R = 27
contre R < 27, **t = −0,65, p = 0,53**, et le sous-ensemble apparié des neuf bijections
donne 0,00849 contre 0,01005, z = −0,55. **Le contraste en R valait précisément une
chose — autoriser la comparaison à une nulle de bijections — et en huit tours aucun de
nous ne l'a dit.** Ce n'était jamais un résultat sur R, c'était une condition de
validité pour la comparaison qui rend R sans objet.

**Ce que j'ai construit puis n'ai pas publié.** Son compte de dépassement à 0 est une
borne, pas une estimation, donc j'ai ajusté la queue pour la convertir : log-linéaire
sur neuf seuils, R² = 0,990, longueur caractéristique 0,00753. Puis j'ai réajusté sur
des sous-fenêtres — 2,60e−08, 1,85e−08, 1,38e−08, 1,06e−08, 7,45e−09 selon le seuil de
départ, soit **un facteur 3,5** en extrapolant 5,9 longueurs caractéristiques au-delà
du dernier seuil à comptes utilisables. **Point d'estimation non publié.** La colonne
honnête reste la sienne : 0 sur 10⁷, p < 3,0e−7 par la règle de trois, plus une phrase
disant que la forme de la queue place la vraie valeur un à deux ordres de grandeur
dessous sans dire lequel. C'est le piège de §1.9 douze tours plus tard : une grandeur
dérivée qui a l'air de converger, issue d'un ajustement dont j'aurais choisi la
fenêtre après l'avoir vue.

**Sa question : la règle 8 se déclenche-t-elle aussi sur une méthode ?** Oui, et c'est
là qu'elle mord — `depassements` avait établi le 11/08, sur la colonne voisine, dans
la même fonction et le même commit, que le tarif de cette nulle est le compte de
dépassement.

**Règle 9. Un choix de méthode est une affirmation sur une classe de grandeurs. Écrire
la classe, dans le même commit que la méthode.** Avec sa limite, que je ne crois pas
résolue : écrire la classe déplace le jugement vers l'endroit où passe la frontière,
tracée une fois, tôt, par celui qui voit le moins ce qui viendra s'en approcher. Je ne
peux pas construire la phrase que j'aurais plausiblement écrite en août couvrant
« toute grandeur qu'on comparera un jour à un optimum adverse » — en août il n'y avait
pas d'optimum adverse dans le projet. **La classe qui devait exister n'existait pas
encore.** Les règles 7 et 8 se déclenchent sur un événement déjà survenu ; celle-ci
demande de la prévoyance, et toutes ses versions se réduisent à « sois plus prudent »
dès que la frontière est contestée.

**Quatre questions posées en retour**, parce que quatorze tours où il trouve et je
vérifie sont eux-mêmes un plan que je n'ai pas audité : ce qui l'a fait ouvrir `tirer`
ce tour-ci et pas aux tours 7, 8 ou 13 (déclencheur post-hoc de même forme que celui
qu'il m'a reproché ?) ; borne ou extrapolation instable, laquelle va dans l'article ;
**quel est son propre taux d'erreur et comment le saurait-il**, lui qui n'a pas
d'adversaire là où j'en ai un — l'argument de l'article 3 retourné ; et s'il existe une
version vérifiable de la règle 9, ou s'il faut la laisser tomber plutôt que d'expédier
une règle qui se lit comme un conseil.

Code : `src/test3_communication/queue_de_inflation.py`, correctif dans
`loi_nulle_longue.py`. Réponse dans `docs/REPONSE_ORDRE15.md`.

### 7.32 Quinzième critique : le numérateur aussi est un maximum d'échantillon, et ce qu'ils estiment tous deux se calcule en 43 secondes

17/08/2026. Il montre que le rapport 0,1443 / max(nulle) décroît avec n par
construction, le dénominateur étant une statistique d'ordre : 1,335 à 2·10⁶, 1,179 à
10⁷, 1,038 à 3·10⁹. Vérifié sur ma machine — max 0,097594 / 0,103746 / 0,122365 à
10⁵ / 10⁶ / 10⁷, et ses deux grandeurs stables tombent sur ma trajectoire à cinq et
six chiffres (E[inflation] 0,0100490 contre son 0,01005099 ; P(infl > 0) 0,676207
contre son 0,6761844).

**Son 3·10⁹ est-il vérifiable ?** Question de Théo, et le seul angle que je n'avais
pas pris : j'ai vérifié ses chiffres un par un depuis neuf tours, jamais l'ensemble.
Tirage à 3·10⁸ — un dixième de son n, monoprocesseur, flux de graines différent :

| | lui (3·10⁹) | moi (3·10⁸) | écart relatif |
|---|---|---|---|
| E[inflation] | 0,0100510 | 0,0100522 | 0,012 % |
| P(inflation > 0) | 0,6761844 | 0,6761580 | 0,004 % |

| seuil | son compte à 3·10⁹ | attendu à 3·10⁸ | mon compte | p de Poisson |
|---|---|---|---|---|
| 0,125 | 42 | 4,2 | 5 | 0,41 |
| 0,130 | 7 | 0,7 | 0 | 0,50 |
| 0,135 | 3 | 0,3 | 0 | 0,74 |

Rien hors du bruit de Poisson, et le maximum est sur la trajectoire : 0,122365 à 10⁷,
**0,127468 à 3·10⁸**, son 0,139048 à 3·10⁹. Ce contrôle vaut pour le comptage et pas
pour la définition : il partage mon `statistiques`.

**Audit de tous ses chiffres depuis le début**, puisque lui comme moi pouvons avoir
tort depuis le départ. Une vingtaine reproduisent exactement. Un était faux — le
« 0,27 » du dixième tour, qu'il a concédé. **Deux n'ont jamais reproduit** : le
P = 0,434 du neuvième tour (j'obtenais 0,634 et 0,822) et le P(≤ 2) = 0,252 du onzième
(j'obtenais 0,354 / 0,488 / 0,518 selon la provenance des résidus). Ce sont
précisément les deux nombres qui **portaient l'argument de leur tour**. J'avais
signalé les écarts et poursuivi parce que la conclusion ne bougeait pas — mais elle ne
bougeait pas parce que mes chiffres allaient toujours **plus loin dans son sens** que
les siens, jamais moins. Un désaccord qui ne renforce jamais que l'autre partie est un
désaccord qu'on n'a jamais vraiment testé.

**Sa règle appliquée à l'autre moitié, ce qu'il n'a pas fait.** Il propose : *tout
nombre entrant dans un rapport voit sa provenance relue, car un rapport cache le n
des deux moitiés.* Il a relu une moitié. `recherche_pire_cas(objectif, generateur,
n_restarts=24, n_pas=60)` — **le 0,1443 est le meilleur de vingt-quatre montées**, et
l'artefact sauvegardé porte `"inflation_maximale": 0.14429720912767127` à la graine 7.
Ma graine 0 aux mêmes vingt-quatre départs donne 0,146685.

| départs | meilleure inflation |
|---|---|
| 6 | 0,143824 |
| 24 | 0,146685 |
| 48 | 0,151461 |
| 384 | **0,154322** |

Le rapport a donc deux axes, et il en fait varier une colonne :

| départs | n = 10⁵ | n = 10⁶ | n = 10⁷ |
|---|---|---|---|
| 6 | 1,474 | 1,386 | 1,175 |
| 24 | 1,503 | 1,414 | 1,199 |
| 384 | 1,581 | 1,487 | 1,261 |

**Le nombre publié était une case sans coordonnées.**

**Et la limite du rapport vaut 1.** La nulle tire `np.argsort` d'un vecteur aléatoire,
soit une bijection uniforme. Le grimpeur part d'une permutation et bouge par
transpositions, qui préservent la bijectivité. Même espace, même objectif `cm - ca`,
la fonction identique dans les deux. **Ce sont deux estimateurs du même supremum** —
le maximum de l'inflation sur les 27! bijections, l'un par tirage uniforme, l'autre
par recherche locale. Le rapport ne converge pas vers quelque chose sur « la recherche
adverse contre le hasard » : il converge vers 1, et toute valeur publiée par l'un ou
l'autre dit seulement quel budget était le plus grand. Entraîné par deux choses déjà
au carnet : §1.9 (un maximum d'échantillon n'estime rien) et la trouvaille du sixième
tour (le voisinage par transpositions ne peut pas quitter le régime bijectif).
Troisième fois en trois tours qu'un résultat ne se propage pas à la colonne voisine.

**Alors j'ai calculé le supremum.** 1500 départs, deux voisinages indépendants —
transpositions seules, et transpositions plus 3-cycles échantillonnés pour sortir des
optima que le premier ne peut pas quitter. **Les deux plafonnent à 0,154322**, atteint
par 1 départ sur 600, en 43 secondes.

| | valeur | part du supremum |
|---|---|---|
| supremum (recherche, deux voisinages, 1500 départs) | **0,154322** | 100,0 % |
| ma borne publiée (24 départs, graine 7) | 0,144297 | 93,5 % |
| son max à 3·10⁹ tirages | 0,139048 | 90,1 % |
| mon max à 10⁷ tirages | 0,122365 | 79,3 % |
| max des 210 runs émergents | 0,059270 | 38,4 % |
| moyenne des 210 runs émergents | 0,010350 | 6,7 % |
| moyenne de la loi nulle | 0,010049 | 6,5 % |

Cette colonne ne bouge avec le n de personne.

**La structure de l'optimum, et une correction que je me suis faite en cours de
route.** La matrice d'information du code maximisant n'a **qu'une ligne non nulle** :
les trois positions portent de l'information sur un seul attribut (0,4156 / 0,4383 /
0,3182) et zéro sur les deux autres. Le max glouton la ramasse trois fois, l'appariement
une. conc_max 0,2465, appariée 0,0922.

Ma première lecture : « la pire inflation vit donc sur des codes dégénérés à faible
concentration absolue, elle ne peut tromper personne ». **Vérifié, et faux.** Inflation
maximale sous plancher sur conc_max :

| plancher | inflation max | conc_max atteint | appariée |
|---|---|---|---|
| 0,30 | 0,140207 | 0,420620 | 0,280413 |
| 0,35 | 0,140207 | 0,385139 | 0,244932 |
| 0,40 | 0,140207 | 0,474035 | 0,333828 |
| 0,50 | 0,140207 | **0,613747** | **0,473540** |
| 0,60 | 0,132570 | 0,613747 | 0,481177 |

Quasi plate. **Un code affichant 0,6137 sur la statistique publiée peut valoir 0,4735
apparié.** La borne est sérieuse à tous les niveaux où un lecteur agirait, et j'ai
failli écrire le contraire parce que le premier optimum regardé était dégénéré.

**Sa règle contre mon carnet : elle marche, et elle en attrape une qu'il n'a pas
nommée.** Le test n contre n/10 sur le fichier de la nulle sépare proprement :
moyenne, écart-type, q50, q99, E[inflation] et P(infl > 0) bougent de 0,00 à 0,03 % ;
q99,9999 bouge de **2,45 %**, le maximum de 12,96 %, le max d'inflation de 17,95 %.
Or `q99,9999` est exporté sous `quantiles_queue_exacts`, avec un docstring où
j'argumente qu'il est exact et non estimé sur un sous-échantillon. Il l'est — et à
n = 10⁷, le quantile 1 − 10⁻⁶ est le dixième plus haut tirage. **Mon propre docstring
confond deux exactitudes**, et son test les sépare en une ligne.

**Sa question, et la réponse est oui.** Une grandeur qui bouge avec n et reste juste à
publier : **une borne**. `p < 3/n` change à chaque n et est correcte à chaque n, parce
que le n est dans la phrase. Ce qui affûte sa règle au lieu de la casser : le test
n'est pas *est-ce que ça bouge avec n*, c'est **est-ce que la phrase publiée porte son
n**. `inflation_maximale = 0,1081` est un énoncé vrai sur 2 000 000 tirages, publié
sous un nom qui ne mentionne aucun tirage. `inflation_max_1e7 = 0,122365` n'a besoin
d'aucune colonne compagne. Publier deux colonnes demande au lecteur de faire le
classement ; nommer le n rend le nombre juste tout seul.

**Quatre questions posées en retour :** qu'est-ce qui se déclenche sur « calcule-le
plutôt » — le supremum a pris 43 secondes après huit tours de dispute sur deux de ses
estimateurs, dans un projet qui existe pour calculer exactement plutôt qu'estimer ;
est-ce qu'un plafond dur à 0,154322 explique sa sur-prédiction hors échantillon, un
ajustement exponentiel sur-prédisant forcément près d'un bord de support fini, auquel
cas c'est une erreur de modèle et non de taille ; est-ce que mon `valider()` est un
contrôle ou un miroir, puisqu'il compare mon vectorisé à mon propre scalaire, deux
implémentations mais un auteur et une lecture de la définition — et si `statistiques`
se trompe, quinze tours de nombres des deux côtés se trompent identiquement ; et
l'offre de mettre son pilote 3·10⁹ dans le dépôt sous son nom, puisque l'asymétrie ne
se corrige pas par le soin mais par les artefacts.

Code : `src/test3_communication/deux_budgets.py`. Réponse dans
`docs/REPONSE_ORDRE16.md`.

### 7.33 Seizième critique : son habitude est réelle, la mienne en est le miroir

17/08/2026. Il répond à l'audit d'ensemble et **concède deux choix de
conditionnement non publiés**, tous deux reproduisant exactement depuis son code.

**Le premier, `sub = counts[counts > 0]`**, écarte les réplicats où le contraste
simulé n'a jamais franchi la barre. Son P(≤ 2) = 0,252 est donc conditionnel à
« le contraste est rapportable », et la phrase publiée ne le disait pas. Vérifié
avec recalibration par bras : atteinte 0,665–0,684, P(≤ 2 | atteint) 0,242–0,285,
P(≤ 2) inconditionnel **0,496–0,514**.

**Et ça expose une faute à moi, qui portait un argument.** Mon écart du douzième
tour — 0,488 / 0,354 / 0,518 selon la provenance des résidus, publié comme « le nul
de la règle 6 n'est pas identifié » — était une erreur de calibration. Je fixais
`delta` une fois, depuis le sigma par niveau de R, puis tirais des résidus dont
l'écart-type était 10 % plus petit dans le bras « cellules » : ce bras portait donc
un effet effectif plus grand (puissance 0,796 contre 0,682) et résistait
mécaniquement mieux. **Recalibré par bras, les trois tombent entre 0,496 et 0,514.**
La provenance ne déplace rien. §1.20 meurt donc pour **sa** raison et non la mienne :
la moitié des effets vrais de cette taille cassent à deux runs.

**Le second**, sur la mise en commun : cinq de ses six lignes reproduisent, dont
0,8013 exact à quatre chiffres et 0,9251 à trois. **La deuxième ne reproduit pas** —
0,7066 chez moi contre 0,4499 chez lui — et elle contredit le mécanisme qu'il énonce
dans le même message : conditionner sur la paire à plus petite erreur type fait
tomber E[d de découverte] de 0,00657 à 0,00414, soit 37 % d'une moitié qui porte
63,5 % du poids. Ses propres lignes cinq et six montrent cet écart sur la statistique
non mise en commun (0,8013 → 0,9251). La ligne deux montre 0,0155.

**Et sa question « lequel des deux notiez-vous » se répond seule** : ni un autre
objet ni un autre conditionnement, **un autre seuil**. Il notait mon +0,0028 publié
(repool brut des 210 runs), je notais +0,00206 (variance inverse des deux moitiés,
l'objet apparié à sa procédure — ce que j'avais signalé au douzième tour). Contre son
seuil j'obtiens ses nombres : 0,4523 et 0,7066. Aucun de nous n'avait tort ; aucun
n'avait écrit contre quoi il mesurait.

**Son mécanisme, retourné contre moi.** Il écrit que conditionner sur un événement de
sélection rétrécit la statistique vers le nul à chaque fois, donc que ses nombres
étaient systématiquement conservateurs — ce qui, dans cet échange, veut dire
favorables à moi. Une habitude appliquée deux fois, pas deux accidents.

Vérifié sur mon propre carnet. Vingt-cinq hypothèses mortes datées : §1.9 retire mon
unique seuil chiffré, §1.11 le certificat qui portait le projet, §1.14, §1.15 et
§1.16 chacune une trouvaille positive, §1.19 une ligne entière, §1.20 et §1.21 des
règles proposées le jour même, §1.22 fait passer 13,9 à 9,8, §1.23 rend la grandeur
propriété de la nulle, §1.25 dissout le rapport. **Un seul contre-exemple net sur
vingt-cinq** : §1.3, où j'annonçais le hasard (0,5) et où j'ai mesuré 0,9966.

Mais « toutes mes corrections affaiblissent mes affirmations » ne prouve pas un
biais : c'est aussi ce à quoi ressemble la convergence depuis un départ trop
confiant. **Le test qui sépare les deux : ai-je déjà dépensé du calcul pour rendre un
résultat négatif plus fort ?**

Jamais. §6.2 publie « toute sélection résiduelle vers le compositionnel est sous
0,0087 », et cette borne est une pure fonction du nombre de graines, 100. Six cents
graines la divisent par deux. La resserrer renforce l'affirmation négative centrale
du projet, coûte moins que ce que j'ai dépensé cette semaine à affaiblir des
affirmations positives, et en six jours personne ne l'a proposé, moi compris.

**Donc l'habitude est symétrique et la mienne est la moitié la plus laide.** La
sienne était un conditionnement conservateur. La mienne est que **je n'audite que
dans la direction où je pourrais surestimer**, parce que c'est celle où avoir tort
est gênant. Un audit qui ne peut trouver qu'une espèce d'erreur rapporte cette
espèce à 100 %, et c'est exactement à quoi ressemblent vingt-cinq entrées de ce
carnet.

**Sa question : une version bon marché de l'audit d'ensemble, qui tourne chaque
tour.** Deux, et la seconde est meilleure.

*Le registre.* Le mien a coûté quinze tours uniquement parce qu'il n'existait pas et
qu'il a fallu le reconstituer. Une ligne par tour le rend gratuit ensuite : `tour |
grandeur | sa valeur | la mienne | reproduit ? | sinon, quel côté l'écart favorise`.
La dernière colonne est tout le contrôle — sans biais systématique les signes se
répartissent, donc une série de même signe est un test binomial gratuit. Le mien
était trois sur trois avant que je le remarque, p = 0,125 : non significatif, et
suffisant pour aller voir.

*La règle de nommage, qui se déclenche tout de suite et unifie les quatre derniers
tours.* `inflation_maximale` cache son n. `P(rupture ≤ 2)` cache son conditionnement
à l'atteinte. `E[|d| du gagnant]` cache son jeu de sélection. Mon
`quantiles_queue_exacts` à 0,999999 cache qu'il est le dixième plus haut de dix
millions. Chacune est une fonction dont les arguments manquent au nom.

**Règle 10 : le nom d'une statistique porte tout argument dont sa valeur dépend.**
Pas une colonne compagne, le nom. `inflation_max_1e7`. `P(rupture ≤ 2 | atteint)`.
`E[|d| du max de dix]`. Le désaccord devient alors visible **là où le nombre est
utilisé**, par qui lit la ligne, au moment de l'écriture et non de la comparaison, et
personne n'a besoin de penser à auditer. Si je ne devais garder qu'une règle de seize
tours, ce serait celle-là : les règles 7 et 8 se déclenchent sur un événement et ne
coûtent rien, la 9 demandait de la prévoyance et je l'ai dit en la proposant, la 10
coûte un nom de variable.

**Trois questions posées en retour :** quelle procédure produit sa ligne deux ; où
s'arrête la règle 10, puisque je peux mettre un n et un conditionnement dans un nom
mais pas « et les résidus venaient des moyennes par niveau de R » — qui est justement
la dépendance illégitime, donc la dangereuse ; et surtout, **existe-t-il un endroit
du dépôt où il pense que je sous-estime un résultat**, pas où je surestime — une
borne à resserrer, un négatif surcouvert, un contrôle qui rendrait un nul plus fort
et que je n'ai pas lancé. En seize tours personne ne l'a cherché, moi compris, et
je viens de mesurer pourquoi je ne peux pas être celui qui le fait.

Réponse dans `docs/REPONSE_ORDRE17.md`.

### 7.34 Dix-septième critique : la lecture extérieure arrive, l'énumération ne tranche pas, et le négatif que j'ai enfin resserré a mordu

18/08/2026. Il fournit la lecture extérieure que je demandais en question 3, et
c'est la chose la plus utile de tout l'échange.

**L'objectif reconstruit depuis la définition, pas depuis mon code** :
`sklearn.metrics.mutual_info_score` au lieu d'un `p log p` maison,
`scipy.optimize.linear_sum_assignment` au lieu de l'énumération des six
permutations, le monde relu dans `TEST3.md` au lieu d'être importé de
`grammaire3`. Accord à **3,05 × 10⁻¹⁶** sur 3000 bijections, et exactement 1,0 sur
les 1296 codes compositionnels. Quinze tours de nombres ne sont donc pas deux
expressions d'une seule mauvaise lecture de la définition.

**Et le trou de `valider()`, que je n'avais pas vu.** Ses deux moitiés importent
`ATTRIBUT`, `TOKEN` et `INFORMATION_TOTALE`, et prennent toutes deux l'argmax par
colonne : c'est un contrôle de vectorisation contre la table `TERME`, rien d'autre.
Surtout, **il ne tire que des permutations**, donc `matrices_information_generale`
— la fonction qui porte tous les résultats sur codes émergents, dans
`code_emergent`, `effet_par_beta`, `courbe_de_contrainte`, `qui_ecrit_le_code`,
`dynamique_uniforme` et `bornes_par_messages_distincts` — n'y apparaît **jamais**.
Six scripts validés par rien pendant six jours, et il a fallu un lecteur extérieur
pour remarquer que les *entrées* du validateur avaient la mauvaise forme, pas sa
logique. Son contrôle sur 2000 codes à collisions : 2,50 × 10⁻¹⁶. Et le prix du
garde-fou, s'il sautait : **0,0205**, soit deux fois E[inflation]. C'est le nombre
qui aurait dû être publié à côté du garde depuis le début.

**Sa question sur le treillis, et la réponse est non.** L'argument des marges est
juste : 1540 tables, 55 valeurs distinctes, et ses trois entrées gagnantes sont sur
le treillis à **0,00e+00**.

*Une contrainte que ni lui ni moi n'avions écrite, et elle est gratuite.* Pour un
code bijectif sur 27 référents uniformes, le message est une image bijective du
référent, donc (M₁, M₂, M₃) est uniforme sur 3³ : **les trois positions sont
mutuellement indépendantes**. Pour des Yⱼ indépendants, I(X ; Y₁..Yₙ) ≥ Σⱼ I(X ; Yⱼ),
et les trois positions déterminent le référent, donc I(Aᵢ ; M) = H(Aᵢ) = log₂3. D'où
**chaque ligne et chaque colonne de la matrice d'information somme à au plus log₂3**.
Sans énumérer : si les trois maxima de colonne sont dans la ligne r, alors
Σⱼ maxᵢ M[i,j] ≤ log₂3 et l'appariement vaut au moins (somme de ligne)/3, donc
**inflation ≤ (2/3)·log₂3/log₂27 = 2/9 = 0,2222**.

*L'énumération, avec cette contrainte.* Triples du treillis vérifiant a+b+c ≤ log₂3
et dépassant 0,154321642873 : **3123 candidats survivent**, le sommet étant
(0,521362144 ×3) à 0,219295 — soit la borne relâchée presque exactement. Testé
directement, en maximisant minⱼ I(Aᵣ ; Mⱼ) sur 400 départs par attribut :
**0,340006701** pour les trois attributs, contre une cible de 0,521362144. Court
d'un facteur 1,53.

**Donc le treillis plus toutes les contraintes de marge ne ferme pas la question :
ce qui mord est la réalisabilité conjointe, que le treillis ne voit pas.**
L'énumération transforme « redémarrer indéfiniment » en « 3123 candidats dont
presque aucun n'est un code », ce qui est une position **pire** que la recherche.
À noter, puisque j'ai proposé « calcule-le plutôt » comme règle deux tours plus
tôt : ici la route exacte existe, est bon marché, et donne une réponse plus lâche
que la route par échantillonnage qu'elle devait remplacer.

**Son certificat hors-ligne, et la règle 10 retournée contre lui.** Le mécanisme est
juste et vérifié. Mais trois de ses quatre masses ne reproduisent pas, et la raison
est visible dans mes propres runs :

| inflation | la sienne | moi, 600 montées | moi, 1500 montées |
|---|---|---|---|
| 0,154321642873 | 0,000000000000 | 0,000000000000 | 0,000000000000 |
| 0,151460867637 | 0,069167547890 | 0,072625925285 | 0,072625925285 |
| 0,146684666683 | 0,131042430405 | **0,145251850570** | **0,108938887927** |
| 0,144297209128 | 0,099860647267 | 0,072625925285 | 0,072625925285 |

La troisième ligne bouge **entre deux de mes propres runs**. La masse hors ligne est
une propriété du **code**, pas de la valeur d'inflation, et plusieurs codes distincts
atteignent la même valeur en portant des masses différentes. Son tableau indexe une
grandeur par une étiquette qui ne la détermine pas — c'est la règle 10 pointée sur
lui, et je ne l'ai vue que parce que mon propre nombre a bougé entre mes deux runs.

Et je ne trouve pas du tout `0,147337819489` : **94 optima distincts sur 1500
montées, il n'en fait pas partie**. C'est sa démonstration du « nécessaire mais pas
suffisant » ; je ne peux ni la confirmer ni la contredire, seulement rapporter que
sur 1500 montées j'ai exactement un optimum à matrice propre et que c'est le maximum.

**Sa réponse à ma question 2, prise telle quelle.** Deux formes à R² de 0,9950 et
0,9975 sur la même fenêtre, en désaccord de six ordres de grandeur à 0,1443, et
mettre le vrai bord **inverse** le signe de l'erreur au lieu de la corriger — la
sensibilité montrant que le corps veut un bord à 0,18 quand le vrai est 0,1543. La
phrase est : **R² sur la fenêtre d'ajustement ne porte aucune information sur
l'extrapolation.** C'est un échec de choix de modèle, qu'aucun n ne répare.

**Et le contrôle dont j'avais dit que je ne l'avais jamais lancé.** §6.2 à 600
graines contre une nulle de 200 000, graine indépendante :

| bras | graine | n | z moyen | SE | IC 95 % | KS p | détectable |
|---|---|---|---|---|---|---|---|
| tabulaire | 0 | 100 | −0,0098 | 0,1025 | [−0,211, +0,191] | 0,386 | 0,00874 |
| tabulaire | 11 | 600 | +0,0099 | 0,0421 | [−0,073, +0,092] | 0,249 | **0,00359** |
| tabulaire | 907 | 600 | +0,0195 | 0,0412 | [−0,061, +0,100] | 0,455 | **0,00359** |
| factorisé | 0 | 100 | −0,0514 | 0,1014 | [−0,250, +0,147] | 0,613 | 0,00869 |
| factorisé | 11 | 600 | **+0,0935** | 0,0429 | **[+0,009, +0,177]** | 0,179 | **0,00356** |
| factorisé | 907 | 600 | **−0,0280** | 0,0417 | [−0,110, +0,054] | 0,384 | **0,00356** |

La borne se resserre d'un facteur **2,45**, ce qui est tout l'intérêt du run.

**Et le 2,18 σ n'a pas survécu.** La graine 907 rend le bras factorisé à
z = −0,0280, signe inversé et zéro dans l'intervalle. Réunis sur 1200 graines :
**z = +0,0327 ± 0,0299, IC [−0,026 ; +0,091], |z/SE| = 1,09**. Tabulaire réuni :
+0,0147 ± 0,0294. **Seizième contraste de cet échange à mourir au second tirage**, et
le premier où j'ai retiré avant d'écrire une ligne d'interprétation.

Une chose que la mise en commun cache et qu'il faut dire : les deux runs à 600 graines
diffèrent sur le bras factorisé de +0,1214 avec SE 0,0598, soit **t = +2,03**. Seul,
c'est un événement à 4 % et sans intérêt. Mais ça veut dire que **la SE d'un run
unique sous-estime la variabilité inter-run sur ce bras**, donc qu'une borne citée
depuis un seul run de 600 graines est optimiste même quand le run est honnête. C'est
l'intervalle groupé qui se publie.

Donc la conclusion de §6.2 tient, et elle tient désormais sur **1200 graines avec un
déplacement détectable de 0,00356** au lieu de 100 graines à 0,00874.

**Ce qui corrige ce que j'ai écrit au tour précédent au lieu de le confirmer.** J'avais
écrit que le premier calcul jamais dépensé à resserrer un négatif avait mordu
immédiatement, et que c'était ce qu'un biais d'audit directionnel prédit. Cette phrase
reposait sur un bras à 2,18 σ : **retirée**. L'affirmation en dessous survit intacte —
je n'avais jamais payé pour resserrer un négatif, et un audit qui ne peut trouver que
des surestimations en rapportera à 100 % — mais la preuve que j'en donnais était très
exactement le genre de nombre que seize tours m'ont appris à ne pas croire, et je l'ai
produit dans l'acte même de décrire le biais.

Ce que le run a réellement acheté est l'inverse de ce que j'annonçais, et c'est mieux :
**c'est la première entrée sur vingt-cinq où payer une mesure a rendu un de mes
résultats négatifs plus fort.** Borne 2,45 fois plus serrée, conclusion tenue, rien à
retirer.

Réponse dans `docs/REPONSE_ORDRE18.md`. Code :
`src/test3_communication/treillis_inflation.py` et `realisabilite_treillis.py`.

### 7.35 Dix-huitième critique : une forme fermée pour son U, et une phrase de moi qui contredit sa propre source

18/08/2026. Il fournit explicitement le code témoin que je disais introuvable.
Vérifié par mon propre `matrices_information` : matrice à une seule ligne non nulle
(0,340006701169 ; 0,360568055315 ; 0,360568055315), masse hors ligne **exactement
nulle**, inflation **0,147337819489**. Sa démonstration « nécessaire mais pas
suffisant » tient donc, et je retire mon « je ne peux ni confirmer ni contredire ».

*Une nuance qui change ce qu'était le quasi-manque.* Ma ligne pour l'attribut 0 porte
ces trois valeurs, mais ma matrice n'est pas la sienne — mon script imprimait
`inflation de ce code = 0,060758294` juste à côté, parce que la recherche maximisait
minⱼ I(A₀ ; Mⱼ) sans jamais exiger que les trois maxima de colonne soient dans cette
ligne. **La ligne était dans mon message, la matrice non.** Ce n'est pas « j'avais le
témoin et je l'ai raté », c'est « j'avais un nombre qui aurait été le témoin sous une
contrainte que je n'avais pas posée ». Et son arithmétique de rareté est juste : 0 sur
1500 à un taux de 1/600 vaut p = 0,082, donc ma recherche vide n'était pas une preuve
contre lui.

**Sa concession sur les masses, et une forme fermée qu'il n'avait pas.** Il reconnaît
avoir publié des **moyennes par valeur** et non des masses par code : 600 montées, 600
codes, 568 matrices, 57 optima, et la masse est une fonction déterministe du code. Mes
deux nombres sont ses deux modes (21 et 23 montées sur 69), et son 0,131042430405 est
une moyenne que **aucun** des 69 codes ne porte.

Il relève que sur les sept plus hauts optima, toutes les masses sont des multiples
entiers de U = 0,018156481321, sans forme fermée. **Elle existe.** La table donnant 2U
est [[2,3,4],[3,3,3],[4,3,2]], marges toutes à 9, d'information mutuelle
(4/27)log₂(2/3) + (8/27)log₂(4/3). En divisant par deux :

> **U = (2/27)·log₂(32/27) = (2/27)(5 − 3·log₂3) = 0,018156481321225**

contre son 0,018156481321, à 2,25 × 10⁻¹³. Et 32/27 = 2⁵/3³ n'est pas arbitraire :
log₂(32/27) = 5 − log₂27 est exactement le jeu entre cinq bits et la largeur du monde.
**L'unité dans laquelle ses masses sont quantifiées est la quantité par laquelle ce
monde rate cinq bits de large**, étalée sur 27 référents et doublée. Son ensemble de k
— {0, 2, 4, 5, 6, 7, 8, 9, 10}, trous en 1 et 3 — reste sans forme.

*Et une phrase fausse de moi.* J'avais écrit « même valeur d'inflation, même code,
montée différente ». Impossible, pour la raison qu'il donne : la masse étant fonction
du code, quand mon nombre a bougé entre mes deux runs, c'est le code qui a bougé. Le
paragraphe suivant du même message disait la bonne chose ; c'est cette phrase-là qui
était fausse. Corrigée.

**Sa question — mon pipeline garde-t-il les atomes ? Non, et voici le compte.** Audit
des 26 artefacts de `results_test3/` : **neuf sur vingt-six publient des moyennes sans
garder ce sur quoi elles portent.** Le plus grave :

```
6_4_gradient_premier_pas_b0.02_20graines_g0.json
  premier_pas.structure.z_moyen        -0.07508
  premier_pas.structure.z_erreur_type   0.23527
```

Vingt graines contre 300 bijections témoins, soit **6000 cosinus réduits à deux
flottants** — et c'est la mesure qui a tué §1.14. Inauditable par quiconque, moi
compris. Même chose pour `6_3_qui_ecrit_le_code`, `6_6_courbe_de_contrainte` et
`certificat_deux_agents`.

**Et ce qui est tombé de cet audit, que personne n'avait vu.** J'ai ouvert `6_4` pour
vérifier s'il gardait ses atomes. Il ne les garde pas. Mais trois clés plus bas :

| pas | 0 | 10 | 30 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|---|---|
| z, structure | −1,18 | −0,29 | **+4,36** | +4,25 | **+3,91** | **+5,81** | **+5,85** |

Or §1.14, publiée le 11/08, dit : « z passe de −1,18 au pas 0 à +4,36 au pas 30, **et
n'en bouge plus** ». **Elle bouge** : elle creuse à +3,91 au pas 300 puis monte à
+5,85, soit **+34 % au-delà du point où j'annonçais l'arrêt**, et elle monte encore au
dernier pas mesuré. Le nombre qui contredit ma phrase est dans le même dictionnaire
que celui qu'elle cite, depuis le jour où je l'ai écrite.

La conséquence n'est pas cosmétique. Mon mécanisme disait : la contrainte ne mord pas
près de l'uniforme, se met à mordre quand la loi se concentre, donc la préférence est
**construite par la trajectoire en quelques dizaines de pas**. L'amorce est juste,
l'achèvement est faux — elle continue d'être construite pendant trois mille pas, et je
n'ai aucune mesure au-delà ni aucune raison de croire que c'est là qu'elle s'arrête.
§1.14 corrigée.

**Comment ça a été trouvé est le point.** Sa question était « gardez-vous les
atomes ». La réponse était non, et en l'établissant j'ai trouvé une erreur d'une autre
espèce, dans un fichier que j'avais déjà exploité pour un résultat publié. **L'audit
n'a pas trouvé ce pour quoi il était conçu.**

**Et le carnet se contredit lui-même.** §7.21 imprime la courbe complète et juste —
+4,36 / +4,25 / +3,91 / +5,81 / +5,85 — dans le même document où §1.14 la résume par
« n'en bouge plus ». La bonne donnée était en §7.21 depuis le 11/08. Ce n'est donc pas
« je n'ai pas lu l'artefact » : je l'ai lu, publié correctement, puis résumé faux
douze lignes plus haut dans le même fichier.

### 7.35bis Quatre pistes vérifiées, quatre déjà publiées, et une qui ne l'est pas

Cherché, sur demande de Théo, ce qui serait sous notre nez depuis le début. Cinq
candidats, vérifiés contre le dépôt avant toute affirmation. **Quatre étaient déjà
documentés**, et le dire est la moitié utile du résultat :

- le coût en récompense de la paramétrisation structurée (0,930 tabulaire contre
  0,861 structurée) — déjà en §7.19 et dans `TEST3.md`, « elle la paie » ;
- la statistique de concentration suit-elle la compositionnalité — déjà mesuré,
  Spearman 0,814, concordance 0,863 ;
- le résultat « l'issue est écrite dans l'initialisation », z = +6,80 au centile
  1,000 — déjà en §7.21 et `TEST3.md` §6.4 ;
- la conception à gradient exact serait cachée — non, `TEST3.md` l'écrit deux fois.

**Le cinquième ne l'est pas.** `reinforce()` est défini une fois dans tout le test 3
et appelé **depuis un seul site** : ligne 362 de `representable_atteignable_stable.py`,
dans la branche *stable*, à partir d'un état où la montée exacte l'avait déjà mis,
pour demander s'il y reste. Tout le reste — §6.1 à §6.7, les distributions de
concentration, les comparaisons à la nulle, et les dix-huit tours de cet échange —
est `torch.optim.Adam` sur le `E[R]` en forme close. Aucun échantillonnage, aucune
variance de récompense, aucune assignation de crédit.

C'est un choix de conception assumé, et l'exactitude est la raison d'être du banc.
Mais la conclusion publiée est « sur ce banc, la compositionnalité n'a jamais été
sélectionnée », et la question du projet est de savoir si **l'apprentissage par
renforcement** la sélectionne. Ce qui a été mesuré est ce qu'Adam atteint sur un
objectif analytique. **REINFORCE n'a jamais été lancé depuis une initialisation
aléatoire.**

Et il y a une raison précise de l'avoir signalé plutôt que supposé inoffensif :
**§1.12, morte le 11/08** — j'avais mesuré un beta critique à 0,0381, accusé la
taille de la perturbation, et la réponse était **Adam**, le hessien donnant
1/27 = 0,037037037 à 2,4 × 10⁻¹¹. La leçon écrite ce jour-là est qu'une propriété de
l'objectif mesurée à travers une boucle d'optimisation mesure l'optimiseur. Puis
§6.1 à §6.7 ont mesuré où la dynamique atterrit, **à travers Adam**, et dix-huit
tours ont affiné la statistique de cette mesure sans que l'un ou l'autre demande
laquelle des deux on regardait.

Je n'affirme pas que la conclusion est fausse. L'équivariance est une propriété de
l'objectif et survit à tout optimiseur ; le no-go de §6.7 s'en moque. J'affirme que
« jamais sélectionnée » n'est aujourd'hui étayé que **pour un optimiseur**, que le
projet s'est déjà fait prendre une fois par exactement cette confusion, et que le run
qui trancherait — REINFORCE depuis l'aléatoire, mêmes graines, même mesure — n'a
jamais été lancé et coûte une nuit.

### 7.35ter La dynamique n'atteint jamais l'ensemble dont parle la prémisse

18/08/2026, sur deux relances de Théo : *« cherche ce qui était sous notre nez »* et
*« on n'est jamais sûr que le code soit juste depuis le début »*. La seconde donne
l'angle : la relecture extérieure du relecteur a validé la **mesure**
(`matrices_information`, `statistiques`, à 3,05 × 10⁻¹⁶). **Personne n'a jamais
vérifié la dynamique**, qui est la moitié qui produit les codes.

**La cible est inatteignable pour 95 % des runs, par construction.** Les 1296 codes
compositionnels sont **tous** des bijections (vérifié : 1296, tous bijectifs). Un run
qui finit avec des collisions ne peut donc pas être compositionnel, quelle que soit
sa concentration.

| bras | n | bijectifs | part | compositionnels | borne sup 95 % |
|---|---|---|---|---|---|
| tabulaire | 1200 | 60 | **5,0 %** | 0 | 6,0 % |
| factorisé | 1200 | 1 | **0,1 %** | 0 | 97,5 % |
| structuré | 40 | 1 | 2,5 % | 0 | 97,5 % |

Et la borne publiée porte sur les 1200. Sur la population où la question est
posable : z = +0,0507 ± 0,1585 contre +0,0147 ± 0,0294, soit **5,4 fois plus lâche**.
La forme directe de la question du plan — parmi les runs ayant atteint l'ensemble
lié, combien sont compositionnels — n'a jamais été calculée : **0 sur 60, borne
supérieure 6,0 %**, contre un nul de 1,19 × 10⁻²⁵. Vingt-quatre ordres de grandeur de
jeu. Ce test n'a aucune puissance, et c'est celui que le cadrage décrit.

**Puis le test de la dynamique elle-même.** D'abord, le plateau à 0,93 est une vraie
convergence : 3000 pas donnent 0,92896, 12 000 donnent 0,92901, 30 000 donnent
0,92901. Dix fois le budget déplace la cinquième décimale, et un pas plus grand fait
pire. Ensuite :

| état | J | E[R] | collisions |
|---|---|---|---|
| convergé depuis l'aléatoire | **0,96395** | 0,96290 | 1 |
| ajusté sur un compositionnel | 0,99980 | 0,99973 | 0 |
| ajusté puis 3000 pas de montée | **1,00000** | 1,00000 | 0 |

**La montée depuis l'aléatoire converge vers un point strictement pire de son propre
objectif**, de +0,03605. Ce n'est pas le terme d'entropie qui refuserait de récompenser
le déterminisme : J atteint exactement 1,00000 à la bijection déterministe et l'y
tient. **Le paysage a des optima locaux, et la montée depuis l'aléatoire tombe dedans
environ 95 % du temps.**

**Ce que ça change.** La prémisse dit : les 27! bijections sont à égalité à récompense
1, donc la récompense ne peut pas trancher entre elles, donc un résultat compositionnel
viendrait d'ailleurs. **La dynamique ne tranche pas entre elles : elle n'arrive
jamais.** Elle converge dans un bassin sous-optimal à ~1,8 référents non décodables,
et les codes compositionnels sont à l'optimum global qu'elle n'atteint pas.

**Et une correction que je me suis faite dans l'heure, dans le sens qui me coûte le
plus.** Mon premier jet accusait Adam, sur le précédent de §1.12. J'ai donc appliqué
la leçon de §1.12 au lieu de la citer, et regardé le gradient plutôt que la boucle :

| pas | E[R] | ‖grad J‖ | relatif | collisions |
|---|---|---|---|---|
| 0 | 0,037037 | 2,107e−05 | 5,58e−05 | 10 |
| 1000 | 0,888496 | 5,005e−05 | 2,69e−07 | 3 |
| 3000 | 0,888833 | 6,594e−06 | 3,04e−08 | 3 |
| 30 000 | **0,888889** | **3,653e−07** | **1,13e−09** | 3 |

**Le gradient tombe à zéro**, et 20 000 pas de SGD à lr = 1,0 depuis le plateau
déplacent E[R] de 7 × 10⁻⁵. C'est un **vrai point critique de J**, pas Adam qui cale —
et 0,888889 vaut exactement 24/27, la récompense d'un code à 24 messages distincts.

Donc la critique n'est pas celle que j'avais saisie : **les points critiques
sous-optimaux sont une propriété du paysage de l'objectif**, pas un artefact d'Adam.

*(Et j'ai poursuivi cette phrase, le 18/08, par « aucune méthode locale n'en sort » et
« ce n'est pas réparable en changeant d'optimiseur ». **Les deux sont faux, réfutés le
19/08 en §7.36.** Le gradient tombe bien à 7 × 10⁻¹¹, donc le point critique est réel
— mais un point critique n'est pas un attracteur fort pour une méthode bruitée :
REINFORCE en sort et atteint une bijection **11 fois sur 12** là où la montée exacte
fait 0 sur 12, à budget, graines et lr identiques, p = 9,6 × 10⁻⁶. C'était la phrase
la plus forte que j'avais écrite ce jour-là.)*

**Ce qui survit intact :** le no-go d'équivariance de §6.7, propriété de l'objectif
et valable pour tout optimiseur ; et l'uniformité intra-classe de fibres, bien mesurée
sur la population qu'elle décrit.

### 7.35quater Trois vérifications qui passent, et une hypothèse fausse sans conséquence

Même jour, en réponse à *« on n'est jamais sûr que le code soit juste depuis le
début »*. La relecture extérieure couvrait la mesure. J'ai vérifié le reste.

**`tirer_profil` — validé, deux fois, et ça n'avait jamais été fait.** C'est le
tirage qui produit la loi nulle de **chaque** z de §6.2. Sur un profil bijectif il
doit coïncider avec `np.argsort(random)` de `loi_nulle_longue`, écrit
indépendamment : KS D = 0,0027, **p = 0,9985**, écart des moyennes −0,20 SE, et la
table marginale 27 × 27 est uniforme (χ² = 678,0 à 676 ddl, p = 0,47). Pour les
profils **non** bijectifs — 95 % des runs — il n'existait aucun second
échantillonneur, donc j'en ai écrit un par construction différente (permuter les
tailles sur les messages, puis partitionner les référents). Accord sur trois
profils : KS p = 0,78 / 0,21 / 0,95, plus grand écart 2,17 SE sur trois comparaisons.

**`objectif()` — validé par reconstruction depuis la définition.** E[R] = tr(SR)/N
reconstruit à **0,00e+00**, à l'initialisation comme après 3000 pas, et
J − E[R] = 0,02 × (H_S + H_R) en nats à la dernière décimale. À l'initialisation
E[R] = 0,037037078, soit 1/27.

**Une hypothèse fausse, et sa conséquence est petite.** §6.2 moyenne des z venant de
lois nulles différentes, ce qui suppose z ∼ N(0,1) sous H0. Faux : l'asymétrie vaut
**+0,48 à +0,58** selon le profil et Shapiro rend p ≈ 10⁻²⁰. Mais l'échelle est
bonne (sd 0,99–1,01, P(|z| > 1,96) = 0,040–0,046), et après cumul sur 1200 le
théorème central limite absorbe presque tout : l'intervalle empirique vaut
[−0,0534 ; +0,0581] contre [−0,0564 ; +0,0564] nominal. **Environ 4 % sur chaque
queue, la droite plus longue** — donc un z positif est légèrement moins surprenant
que publié, ce qui va dans le sens du 2,18 σ d'hier qui n'a pas survécu.

C'est la première fois de cet échange que je rapporte une hypothèse violée dont la
conséquence est négligeable. Ça vaut d'être nommé comme catégorie : le contrôle qui
passe et le contrôle qui échoue coûtent le même prix, et seul le second se raconte.

### 7.35quinquies Un nombre qui ne peut pas exister, imprimé dans mon propre artefact

18/08/2026. Sa trouvaille sur les percentiles — un p10 publié sans convention, que
seule la méthode `nearest` de numpy reproduit — **ne transfère pas** à mon dépôt :
les treize méthodes rendent le même chiffre chez moi, la statistique de concentration
étant assez discrète pour qu'elles coïncident. Vérifié, pas supposé.

Mais la chercher m'a envoyé dans §6.3, où il y a autre chose.

`qui_ecrit_le_code.py` publie **`plafond_beta = 0,9999230227241369`** — la récompense
maximale atteignable à β = 0,02, mesurée en gelant un agent sur une bijection et en
laissant l'autre apprendre, médiane 139 pas. **Cette constante est le dénominateur de
tous les `ratio_au_plafond` de la section.**

C'est une mesure de boucle. En la laissant tourner :

| pas de montée | E[R] | ‖grad J‖ |
|---|---|---|
| 0 | 0,9997270898 | 2,86e−05 |
| 139 | 0,9999945548 | 5,12e−07 |
| 1000 | 0,9999999604 | 3,18e−09 |
| 20 000 | **0,9999999990** | **7,17e−11** |

**Le vrai plafond vaut 1,0 à neuf décimales.** Le publié est court de **7,70 × 10⁻⁵**
en relatif, parce que la boucle s'est arrêtée sur son critère de convergence et que le
critère n'était pas serré.

Et la conséquence était imprimée :

```
les deux libres, S tabulaire :
  E[R] 0,911055   plafond 0,911041   ratio_au_plafond 1,000016
```

**Un ratio à un plafond ne peut pas dépasser 1.** Le nombre est dans l'artefact, et
§7.20 le rend par `1,0000` — c'est là que l'impossibilité a disparu. Avec le plafond
corrigé il vaut 0,999939, donc la conclusion tient à la cinquième décimale : la paire
libre exécute bien son code presque parfaitement. Ce qui ne tient pas, c'est qu'un
nombre qui se réfute lui-même soit resté sept jours à l'écran.

**Quatrième fois que ce projet mesure une propriété de l'objectif à travers une
boucle d'optimisation et rapporte la boucle.** §1.12 en est morte le 11/08. §7.35ter
ce matin, c'était moi qui recommençais et qui l'ai rattrapé par le gradient. Et
`plafond_beta` le fait depuis le premier jour, au dénominateur de la statistique de
tête d'une section.

**Règle 11 : toute grandeur dont le nom implique une borne arithmétique est
contrôlée contre cette borne là où elle est calculée.** `ratio_au_plafond <= 1`. Une
probabilité dans [0, 1]. Une corrélation dans [−1, 1]. Une entropie sous log n. C'est
un `assert`, pas une habitude, ça coûte une ligne et ça échoue fermé. Sa règle
attrape un nombre qui n'a rien derrière lui ; celle-ci attrape un nombre qui
contredit sa propre définition. Ce sont les deux moitiés.

### 7.36 Dix-neuvième critique : le 5 % est une propriété de mon optimiseur, pas du banc

19/08/2026. Il me demande si REINFORCE depuis l'aléatoire est un contrôle de fidélité
ou un mécanisme d'échappée, et argumente qu'un estimateur à plus grande variance du
même gradient atteindrait l'ensemble lié **moins** souvent. Plutôt que d'en discuter,
je l'ai lancé. Deux fois, et le second run renverse le premier.

**Premier run, 25 graines, budgets par défaut :** montée exacte 0,93920 avec 1/25
bijections, REINFORCE lot 64 0,89240 avec 0/25, lot 8 0,37436 avec 0/25 et 6,76
collisions. Monotone dans le sens qu'il prédisait. **J'ai failli le lui envoyer comme
« vous avez raison ».** Ce qui m'a arrêté est que les bras n'étaient pas appariés —
`monter` est Adam à lr = 0,05 sur 3000 pas, `reinforce` est lr = 0,01 sur 4000 — et
que publier cet écart comme de la variance aurait été `plafond_beta` une fois de plus.

**Second run, balayage, 12 graines par cellule :**

| lot | pas | lr | E[R] | bijections | collisions |
|---|---|---|---|---|---|
| 64 | 4000 | 0,01 | 0,89542 | 0/12 | 1,92 |
| 64 | **20 000** | 0,01 | **0,99178** | **11/12** | **0,08** |
| 64 | 4000 | 0,05 | 0,97458 | 5/12 | 0,58 |
| 64 | 20 000 | 0,05 | 0,98902 | 9/12 | 0,25 |
| 8 | 4000 | 0,01 | 0,38557 | 0/12 | 6,75 |
| 8 | 20 000 | 0,01 | 0,92874 | 3/12 | 1,08 |

Le 0,374 était du **sous-entraînement**, pas de la variance. Tout le premier tableau
était un artefact de budget.

**Et la cellule appariée**, 20 000 pas, mêmes graines, mêmes lr, une seule différence :

| méthode | lr | E[R] | bijections | collisions |
|---|---|---|---|---|
| montée exacte | 0,05 | 0,94753 | **0/12** | 1,42 |
| montée exacte | 0,01 | 0,92593 | **0/12** | 1,83 |
| REINFORCE lot 64 | 0,01 | **0,99178** | **11/12** | 0,08 |
| REINFORCE lot 64 | 0,05 | 0,98902 | **9/12** | 0,25 |

Fisher exact à lr égal : 11/12 contre 0/12, **p = 9,6 × 10⁻⁶** ; 9/12 contre 0/12,
p = 3,4 × 10⁻⁴.

**Il a tort, et moi plus que lui.** Le sien : l'estimateur échantillonné atteint
l'ensemble lié bien plus souvent, aux deux taux d'apprentissage. Le mien, écrit la
veille : « les attracteurs sous-optimaux sont une propriété du paysage, aucune méthode
locale n'en sort, ce n'est pas réparable en changeant d'optimiseur ». Le gradient
tombe bien à 7 × 10⁻¹¹, donc le point critique est réel — **mais un point critique
n'est pas un attracteur fort pour une méthode bruitée.** Les pièges existent et ne
mordent pas sur un gradient échantillonné.

**Ce que ça rouvre.** Le 5 % de §7.35ter est une propriété de la **montée exacte**,
pas du banc. La question dont parle la prémisse — 27! codes à égalité, 1296
compositionnels — est atteignable dans 92 % des runs sous REINFORCE, et elle n'a
jamais été posée, puisque `reinforce()` n'est appelé que depuis un état que la montée
exacte a déjà trouvé (§7.35bis). **Ce n'était ni un contrôle de fidélité ni une nuit
perdue : c'est l'expérience, et elle était cachée derrière un nombre de pas.**

Ce qui reste à faire, et c'est maintenant la première ligne de la suite : relancer
§6.2 sous REINFORCE lot 64 à 20 000 pas, où 92 % des runs entrent dans l'ensemble
lié, et mesurer la concentration **sur la population dont parle le plan**.

**Et la leçon de procédure.** Le seul réflexe qui a servi aujourd'hui est le caveat
posé avant publication : *les bras ne sont pas appariés*. Sans lui j'envoyais un
artefact de budget comme confirmation, à quelqu'un qui l'aurait cru parce qu'il allait
dans son sens. Un résultat qui confirme l'interlocuteur ne se vérifie pas moins qu'un
résultat qui le contredit — il se vérifie plus, parce que personne ne le contestera.

### 7.37 Vingtième critique : mon voisinage n'était certifié qu'à 47,3 %, et sept optima n'en étaient pas

20/08/2026. Je lui avais demandé où s'arrête sa boucle. Il répond que les deux
critères d'arrêt de `monter()` — un plafond de 300 tours et une tolérance de 1e−12 —
sont sains (le plafond ne se déclenche jamais, le plus petit gain d'une échappée vaut
1,247 × 10⁻⁴, huit ordres au-dessus de la tolérance), et que **ce qui est cassé est
ailleurs** : le voisinage est de 351 transpositions plus 2925 3-cycles, et la montée
n'en échantillonne que 1200. Chaque arrêt certifie une non-amélioration contre
**1551 mouvements sur 3276, soit 47,3 %**.

**Le diagnostic porte sur mon code** — `voisins_3cycle` avec `echantillon=1200` est
une fonction que j'ai écrite dans `supremum_inflation.py`. Relancé de mon côté, sur
mes graines :

| | lui | moi |
|---|---|---|
| arrêts qui ne sont **pas** des optima | 88/600 | **85/600** |
| nature de l'échappée | triples 88, paires 0 | **triples 85, paires 0** |
| gain minimum | 0,000124724928 | **0,000124724928** |
| gain médian | 0,003818488095 | **0,003818488095** |
| gain maximum | 0,015306403306 | 0,016356390168 |

**Les gains min et médian coïncident à la douzième décimale** — ce sont des quantités
du treillis, donc deux flux de graines indépendants tombent sur les mêmes atomes ; le
maximum diffère parce qu'on termine sur des codes différents. Et la structure
reproduit exactement : **toute échappée est un 3-cycle, aucune n'est une
transposition.** La moitié exhaustive du voisinage ne peut pas échouer et n'échoue
pas ; la moitié échantillonnée est le mécanisme, pas un corrélat.

Continué sous le voisinage complet :

| | lui | moi |
|---|---|---|
| optima distincts | 57 → 50 | **52 → 45** |
| maximum | inchangé | **inchangé** |
| destinations au-dessus du maximum | 0 | **0** |
| montées sous les sept premiers | 103 → 138 | **101 → 134** |

On perd sept optima chacun, tous dans la moitié basse, et on gagne chacun un tiers de
preuve sous le sommet. **Le supremum de §7.32 survit** et il est désormais certifié
contre 3276 au lieu de 1551. Mais je l'avais publié comme « 1500 départs, **deux
voisinages indépendants**, tous deux plafonnent à 0,154322 » — **sans jamais imprimer
les 47,3 %.** Le second voisinage était à moitié énuméré et la phrase laissait croire
que c'en était un.

**Sa règle, adoptée : une valeur produite par une recherche se rapporte avec la
fraction de l'espace contre laquelle elle a été certifiée.** Et son cadrage est ce qui
la rend générale : *une valeur convergée n'est pas une mesure, c'est une affirmation
de non-existence* — aucun meilleur voisin, aucune ascension possible — et le conteneur
d'une non-existence est **l'espace de réfutation réellement énuméré**. Imprimer 1e−12
à côté d'un plafond qui n'a jamais servi décrit la règle d'arrêt et ne dit rien de la
recherche.

Il relève aussi que `plafond_beta` n'était pas un nombre sans conteneur : **‖grad J‖
était à l'écran juste à côté**, quatre ordres de grandeur lâche, et je l'ai lu comme
un diagnostic au lieu du certificat qu'il était. C'est pire que de manquer un
conteneur : c'est en avoir un et ne pas le reconnaître.

**Les trois règles, dont aucune ne subsume les autres :** la sienne du tour 19 — tout
nombre résout son conteneur, un conteneur irrésolvable est un échec — attrape un
compte sans fichier. La 11 — toute grandeur dont le nom implique une borne est
contrôlée contre elle — attrape `ratio_au_plafond = 1,000016`. La sienne de ce tour
attrape sept optima qui n'en étaient pas, et mes 47,3 %.

**Et sa critique de mon balayage REINFORCE tient.** Le lot fait varier variance et
volume de tirages **ensemble** — 64 × 20 000 fait 1 280 000 tirages contre 160 000
pour 8 × 20 000 — donc chaque cellule est compatible avec les deux récits. Son
contrôle : apparier les tirages, lot 8 × 20 000 contre lot 64 × 2500, tous deux à
160 000.

*Ce que je pose avant les chiffres, parce que ça contraint les deux conclusions
possibles :* **apparier les tirages désapparie les mises à jour.** Lot 64 × 2500 fait
2500 pas de gradient contre 20 000 pour le lot 8. Son contrôle fixe les tirages et
fait varier les mises à jour d'un facteur 8 ; le mien fixait les mises à jour et
faisait varier les tirages d'un facteur 8. **Aucun des deux n'isole la variance** —
les trois quantités sont liées par une identité et il n'y a que deux axes. La cellule
départagera son récit du mien, mais pas contre un troisième que ni lui ni moi n'avons
nommé : les **mises à jour**.

Grille en cours : ligne iso-échantillons à 160 000 tirages (lot 8 × 20 000, lot
16 × 10 000, lot 64 × 2500), aux deux taux d'apprentissage, douze graines par
cellule, plus lot 64 × 20 000 hors ligne comme ancre.

**Et un constat sur l'échange lui-même.** Quatre audits d'affilée, des deux côtés,
n'ont pas trouvé ce pour quoi ils étaient conçus : son audit des atomes a trouvé un
fichier manquant, le mien a trouvé une phrase contredisant son propre artefact, sa
question sur la tolérance de boucle a trouvé un voisinage non certifié, et mon
contrôle de fidélité REINFORCE a trouvé l'expérience. Question posée en retour :
reste-t-il autre chose à en conclure que **la cible d'un audit est la chose qu'il a le
moins de chances d'attraper** — et si oui, faut-il auditer au hasard plutôt que
d'auditer ce qu'on soupçonne ?

Réponse dans `docs/REPONSE_ORDRE21.md`.

### 7.38 Vingt-et-unième critique : sa règle appliquée au fichier que je lui avais écrit, et une ligne de référence qui fait l'inverse de ce qu'elle promet

24/08/2026. Il refuse ma conclusion des quatre audits d'affilée et la sépare en deux
lectures qui donnent des conseils opposés. *Lecture une :* un audit rate sa cible
parce que la cible avait déjà reçu de l'attention — alors seule la nouveauté de la
région compte et auditer au hasard suffit. *Lecture deux :* un audit trouve dans sa
**traversée**, et la cible n'est qu'un point dedans ; la probabilité que l'unique
défaut soit sur le point nommé vaut environ un sur la taille de ce qu'il a fallu
toucher. Il tranche pour la deux, sur les quatre vérifications de la semaine classées
par largeur, et en tire : *choisir la revendication dont la vérification force à
ré-énumérer le plus.*

**Il a raison de refuser ma conclusion, et sa lecture deux ne survit pas non plus.**
J'ai vingt-neuf points au lieu de quatre — les vingt-neuf entrées de §1 — chacune
nommant la mesure qui l'a tuée. Codage intégral dans
`src/test3_communication/anatomie_des_audits.py`, une justification par ligne, pour
qu'il puisse recoder.

| | n | visé | instrument | preuve déjà sur le disque |
|---|---|---|---|---|
| tout le carnet | 29 | 18 (62 %) | 11 (38 %) | 9 (31 %) |
| avant le 14/08 | 17 | 13 (76 %) | 1 (**6 %**) | 2 (12 %) |
| à partir du 14/08 | 12 | 5 (42 %) | 10 (**83 %**) | 7 (58 %) |
| sa semaine (ses 4 + mes 2) | 6 | 0 (0 %) | 6 (100 %) | 5 (83 %) |

**La visée ne rompt pas : Fisher p = 0,119.** Sur tout le carnet, 62 % des morts sont
dues à une vérification qui les visait. « La cible est ce qu'un audit a le moins de
chances d'attraper » n'est même pas une propriété de ce projet.

**Ce qui rompt, c'est l'objet : 6 % à 83 %, Fisher p = 3,3 × 10⁻⁵.** Et « la preuve
dormait déjà sur le disque » passe de 12 % à 58 %, p = 0,014. Croisé : parmi les morts
de MONDE la preuve dormait 3 fois sur 18, parmi les morts d'INSTRUMENT 6 fois sur 11.

**D'où une troisième lecture, qui n'est ni la sienne ni la mienne.** Tôt, les
propositions fausses portent sur le monde, et une vérification visée les tue — on peut
viser ce dont on a une hypothèse. Tard, les propositions qui restent ont déjà survécu
aux vérifications visées, et ce qui meurt est **l'instrument** : un plafond de
réservoir, un `n_restarts=24`, une colonne qui n'est pas un facteur, un voisinage
échantillonné. Personne n'a d'hypothèse sur un argument par défaut, donc personne ne
peut le viser. **La visée marche sur le monde et ne peut pas s'appliquer à
l'instrument** — non parce qu'elle échoue, parce qu'elle n'a pas de prise.

Et pour l'instrument, la largeur ne fait rien : sept des douze dernières morts sont
venues de rouvrir un fichier, lire un site d'appel, imprimer un min et un max.
**Traversée nulle, rendement maximal.**

**Son tableau ne porte pas le gradient qu'il lui prête.** Les rendements de ses quatre
lignes valent 1, 1, 3, 1 pour des largeurs étroite, étroite, large, large. Moyenne
1,0 contre 2,0, et tout repose sur une ligne qui rend 3 au lieu de 1 — sous un nul de
Poisson de moyenne 1,5, la probabilité qu'au moins une des quatre rende ≥ 3 vaut 0,57.
Pire : **la granularité de la colonne rendement a été choisie après avoir vu les
résultats.** Il compte « 7 non-optima, les 47,3 %, le déplacement du top-sept » pour
trois, et « la conception du balayage » pour un — alors que le même contrôle REINFORCE
a produit l'artefact de budget, le renversement de §1.29, le 11/12, et le fait que la
question du plan n'avait jamais été posée. Recompté à granularité égale, le gradient
disparaît.

**Et son cinquième point n'est pas un point.** Il dit que ma reprise de
`voisins_3cycle` a touché sa cible en plein centre, donc que la lecture une tombe. Ce
n'était pas une recherche : c'était une **réplication de son propre résultat**. La
région avait été choisie parce qu'un défaut venait d'y être démontré. Conditionner sur
la réponse puis compter le taux de réussite est exactement le défaut de sélection que
cet échange passe vingt tours à nommer.

**Sa règle, appliquée là où ni lui ni moi n'avions regardé — et c'est mon fichier.**
`realisabilite_treillis.py`, que j'ai écrit **pour lui au tour vingt, dans le message
où j'adoptais sa règle**, fait monter ses deux campagnes par **transpositions seules :
351 mouvements sur 3276, soit 10,7 %**. Or §7.37 a établi que sur 85 faux optima, 85
s'échappent par 3-cycle et **zéro** par transposition. Le fichier n'utilise que la
moitié du voisinage qui ne trouve jamais rien.

Re-certifié contre les 3276, puis poursuivi jusqu'à l'optimum sous le voisinage
complet (`recertifier_les_bornes.py`) :

| campagne | départs | arrêts qui ne sont pas des optima | échappée | optima distincts | maximum |
|---|---|---|---|---|---|
| A. attribut 0 | 400 | **121 (30,2 %)** | 3-cycle 121, transposition 0 | 11 → 8 | 0,340006701169 inchangé |
| A. attribut 1 | 400 | **134 (33,5 %)** | 3-cycle 134, transposition 0 | 9 → 8 | 0,340006701169 inchangé |
| A. attribut 2 | 400 | **143 (35,8 %)** | 3-cycle 143, transposition 0 | 11 → 8 | 0,340006701169 inchangé |
| B. inflation | 600 | **247 (41,2 %)** | 3-cycle 247, transposition 0 | 72 → 51 | 0,154321642873 inchangé |

**Les quatre maxima survivent, et zéro arrêt sur 1800 était une troncature de budget.**
Ce qui ne survit pas est la fraction de certification, jamais imprimée. Et le gain
médian d'échappement de la campagne A2 vaut **0,018156481321** — c'est exactement
`U = (2/27) log₂(32/27)`, la forme fermée que j'avais dérivée pour SON quantum au tour
dix-huit. Son quantum réapparaît comme atome modal du treillis des gains d'échappement,
par un chemin qu'aucun de nous n'avait emprunté.

**Sa colonne LOO fait le contraire de ce qu'elle promet, et ça se mesure sans
optimiseur.** Il propose une ligne de référence leave-one-out dans le lot, à tirages
et mises à jour fixes, pour ne bouger que la variance. Mesurer une réduction de
variance à travers une boucle d'optimisation serait la **cinquième** fois de ce projet
(§1.12, `plafond_beta`, mon 5 %, ma phrase sur les méthodes locales). La variance est
une propriété du point et de l'estimateur : elle se mesure à θ fixe, sur des lots
répliqués, sans une seule mise à jour. `variance_du_gradient.py`, 20 000 lots par
cellule, trois points de la dynamique.

D'abord deux contrôles que personne n'avait faits. **Le gradient analytique contre
autograd : 2,5 × 10⁻²⁰, 1,7 × 10⁻¹⁷, 8,5 × 10⁻¹⁸** aux trois points. **Et l'estimateur
échantillonné est non biaisé pour le gradient exact** — l'écart vaut 0,27 à 1,52 erreur
type de Monte-Carlo sur les trente-six cellules, donc REINFORCE monte bien le même
objectif. Le récit « ce n'est pas le même objectif » est mort avant d'être écrit.

Puis la mesure. Variance totale, lot 8 :

| point | aucune ligne | EMA (canonique) | LOO | constante optimale |
|---|---|---|---|---|
| θ init | 8,78 × 10⁻³ | 8,47 × 10⁻³ | **1,02 × 10⁻²** | 8,64 × 10⁻³ |
| θ milieu | 4,59 × 10⁻³ | 6,21 × 10⁻³ | **6,45 × 10⁻³** | **2,86 × 10⁻³** |
| θ piège | 9,23 × 10⁻³ | 9,21 × 10⁻³ | **9,41 × 10⁻³** | **5,12 × 10⁻³** |

**LOO monte la variance de 2 à 20 % selon le point**, jamais ne la baisse, quand mon
axe du lot la divise par 7,9 à 8,1. La raison est arithmétique : l'avantage LOO vaut
(n·r_i − S)/(n−1), de variance p(1−p)·n/(n−1), soit 8/7 fois celle de l'avantage
centré, à n = 8. Et il **annule entièrement 39 à 73 % des lots** — tous ceux où les
huit récompenses sont égales — donc il coupe aussi les mises à jour effectives.

**Ce qui marche, et qui n'est pas ce que le manuel dit :** la constante **optimisant la
variance** divise par 2,26 en milieu de montée et 1,84 au piège. LOO estime E[R] ;
b* = E[R‖score‖²]/E[‖score‖²] en diffère parce que la magnitude du score est corrélée à
la récompense. **La ligne de référence à la moyenne n'est pas la ligne de référence à
variance minimale ici, et LOO estime la mauvaise.**

Conséquence de plan, posée avant les chiffres de sa cellule : sa colonne ne peut pas
conclure d'un résultat nul, parce qu'elle déplace la variance de +2 % là où l'axe du
lot la déplace de −87 %. C'est le **plancher de détection** de §1.21, appliqué à un
contrôle avant de le lire.

### 7.39 Vingt-deuxième critique : il retire sa propre affirmation sur son quantum, et les scripts cités n'ont toujours pas d'adresse

25/08/2026. Il retire les trois points où j'avais raison — granularité, cinquième
contrôle, LOO — sans réserve, et retourne son propre AST sur son dépôt : 409 fichiers,
35 où un générateur unique traverse plusieurs consommations. Sur `c580-rdt-quantum.py`,
le fichier de son quantum, trois ordres du même flux donnent le même maximum
(0,154321642873) et le même ensemble de k, mais un septième optimum diffère selon
l'ordre — masse hors ligne 0,155054165625, à **0,46 U** du treillis. Sa propre
affirmation du tour dix-huit, « les sept plus hauts sont tous des multiples entiers
de U », ne survit pas au réordonnancement de son propre flux.

**Vérifié ce qui pouvait l'être sans son code : l'arithmétique tient.**
0,155054165625 / U = 8,539879665, écart au multiple le plus proche −0,46 U — accord à
trois chiffres avec son `4,60e-01` publié. Pas une coquille de transcription.

**Et une confirmation gratuite dans l'autre sens.** Son 0,154321642873 est exactement
le nombre que mon propre `recertifier_les_bornes.py` a produit ce même tour, sur un
script différent, un générateur différent, une construction de voisinage différente.
Deux codes indépendants tombent sur le même supremum à la treizième décimale — la
meilleure preuve que ce nombre est une propriété de l'objectif, produite par accident
plutôt que par vérification croisée délibérée.

**Ce qui reste ouvert :** aucun des scripts cités (`c580-rdt-quantum.py`,
`c510-rdt-selection.py`, `c578-rdt-bound.py`, les quatre `c674-*`) n'a d'adresse.
Demandé explicitement dans `docs/REPONSE_ORDRE23.md`, avec l'adresse du mien en
échange. Proposé aussi une quatrième question d'instrument, distincte des trois
siennes parce qu'elle ne suppose aucun hasard : une colonne est-elle fixée avant le
run, ou lue sur la propre sortie du run — le défaut de §1.19 et §1.21, tous deux
reproductibles sans generateur.

Réponse dans `docs/REPONSE_ORDRE23.md`.

### 7.40 Vingt-troisième critique : son discriminateur reproduit à l'octet près, et le dépôt reste demandé

25/08/2026. Il répond zéro sur douze — les deux fichiers les plus sûrs de son
recensement ne threadent pas un générateur, ils **réinitialisent** le flux global de
`random`, ce que sa syntaxe d'AST ne distingue pas de `default_rng`. Rejoué son
discriminateur mot pour mot : hachages identiques aux siens jusqu'au dernier chiffre,
UNCHANGED avec la réinitialisation quel que soit le nombre de tirages injectés,
MOVED dès un seul tirage sans elle. **Vérification indépendante complète, pas une
lecture.**

Il propose une cinquième question — une grandeur mesure-t-elle sa cible directement,
ou une corrélation moins chère qui peut diverger sans que rien sur la page ne puisse
le voir. Acceptée, avec un exemple déjà dans ce carnet et sans aucun hasard :
`plafond_beta`, §7.35quinquies, une boucle arrêtée à 139 pas dont rien sur la page ne
contredisait la valeur avant l'étendre à 20 000.

Vérifié chez moi : RDTRL réinitialise à deux endroits
(`rl_copie.py:331`, `rl_grammaire.py:111`), et l'écart au premier tirage est nul par
lecture de `PolitiqueGRU.__init__` — **lu, pas mesuré par son discriminateur**, dit
comme tel. Aucun script de test 3 ne réinitialise en cours de fichier.

**Toujours ouvert :** son dépôt n'a pas d'adresse. Redemandé explicitement, fichier
par fichier, dans `docs/REPONSE_ORDRE24.md`.

Réponse dans `docs/REPONSE_ORDRE24.md`.

### 7.41 Vingt-quatrième critique : huit fichiers reçus, la table de §6.5 ne bouge pas, et un plafond en cachait un autre

25/08/2026. Huit scripts collés en clair. Rien pris pour argent comptant :
re-dérivé chaque affirmation structurelle depuis ma propre source avant de la
croire.

**La bissection non appariée, confirmée à l'entier.** Rejoué l'ordre de tirage
réel de `certificat_deux_agents.py` — trois permutations, la boucle de mélange K,
vingt-quatre tirages de phase 1 — en pur numpy, sans torch : les quatre premières
graines par niveau de bruit tombent exactement sur les siennes
(25970514, 826555961, 763435854 pour bruit=0,01…), et les quatre niveaux
partagent zéro graine sur douze, sur les six paires. Confirmé depuis ma propre
lecture du fichier, pas depuis son extrait.

**L'inventaire de mon dépôt vérifié indépendamment.** `git ls-files` : 86 fichiers
suivis, 55 sous `src/`, exactement 1 JSON — ses trois chiffres, exacts. Et son
audit des 60 chemins cités dans ce carnet, rejoué avec ma propre méthode : 40
résolvent, 20 non, dont 17 gitignorés par règle documentée et 3 qui sont ses
propres fichiers `c510`/`c578`/`c580` cités dans mes réponses. Zéro non expliqué,
comme lui.

**Le défaut de dérivation existe cinq fois chez moi, pas une.**
`torch.Generator().manual_seed(int(generateur.integers(1 << 30)))` : trouvé par
grep après sa découverte, présent aussi dans `representable_atteignable_stable.py`
aux quatre constructeurs `EmetteurTabulaire`, `EmetteurFactorise`,
`EmetteurStructure`, `Recepteur`. Son censeur classerait les cinq occurrences
UNSEEDED pour la même raison qu'il a mal classé le sien.

**Sa question fermante, testée plutôt que lue.** Phase 3 de ce même fichier — le
tableau §6.5, source du 5 % de §1.28 — hérite la position de flux de phases 1 et
2. Avancé le flux de 72 tirages sans exécuter un seul pas de montée (les
constructions ne tirent qu'un entier chacune, confirmé), puis lancé les 30 vraies
montées de phase 3 sur cette position et sur un générateur frais :

| paramétrisation | bijections expédié/indép. | E[R] expédié/indép. | concentration appariée |
|---|---|---|---|
| tabulaire | 0/10 — 1/10 | 0,9185 — 0,9481 | 0,1226±0,033 — 0,1234±0,031 |
| factorisé | 0/10 — 0/10 | 0,7888 — 0,7814 | 0,1201±0,036 — 0,1321±0,037 |
| structure | 0/10 — 0/10 | 0,8518 — 0,8518 (identique) | 0,4301±0,109 — 0,4243±0,086 |

**Le tableau ne bouge pas** — tous les écarts tiennent dans l'écart-type imprimé,
à n = 10 par bras. Le couplage est un vrai défaut d'écriture, pas une menace
vivante pour §1.28. Corrigé quand même : un `default_rng` frais avant la boucle
de phase 3 coûte une ligne.

**`plafond_beta`, la forme de la courbe.** Ni géométrique ni logarithmique
proprement : décélération loin de l'optimum (Adam à pas normalisé), puis
géométrique à la précision machine une fois dans le bassin. Conséquence
opérationnelle : étendre le budget aurait attrapé le sous-comptage, le gradient
étant déjà à six ordres de grandeur sous son départ dès 2000 pas. Et un plafond
en a caché un autre pendant la mesure : ma réplique gèle le récepteur à force = 8,
dont le plafond de décodage vaut exactement e⁸/(e⁸+26) = 0,991353 — une constante
non nommée qui a joué le même rôle que `plafond_beta` avant que je l'étende.

Réponse dans `docs/REPONSE_ORDRE25.md`.

### 7.42 Vingt-cinquième critique : le tableau structure n'était pas un point fixe, c'était un compte de collisions

25/08/2026. Il prix les deux phrases du tableau de §7.41 l'une contre l'autre :
l'écart de 0,0296 sur tabulaire dit que l'écart-type par graine est assez grand
pour rendre l'écart ordinaire ; l'accord à 1e-4 sur structure dit qu'il est assez
petit pour que le même accord soit un événement à 1 chance sur 56 au mieux. Aucun
écart-type unique ne rend les deux phrases ordinaires en même temps.

**Il a raison, et la raison n'est ni la sienne ni la mienne.** Imprimé les dix
valeurs par graine des deux bras : l'écart-type individuel vaut 0,047 et 0,050,
pas un point fixe. Mais groupé par nombre de collisions, cinq classes sur six
saturent (27−collisions)/27 à 1e-4 près — le même résidu de convergence que
`plafond_beta` et sa réplique du tour précédent, pour la troisième fois. `E[R]`
n'est pas continu ici : c'est une fonction quasi déterministe d'un entier, et cet
entier a un écart-type par graine de ~1,3. Moyenne des collisions : 4,0 côté
expédié, 3,9 côté indépendant, écart bien sous une erreur type — l'accord à
quatre décimales sur `E[R]` n'est donc pas une coïncidence à 1 sur 56, c'est un
événement ordinaire sous le bon modèle génératif, que ni lui ni moi n'avions le
bon dès le départ.

**Une classe casse le motif, gardée plutôt que lissée :** une graine du bras
indépendant a 3 collisions par argmax (24 messages distincts) mais un `E[R]`
qui atterrit sur la cible à 4 collisions — en retard d'un quantum entier sur son
propre compte de collisions. La quantification fait l'essentiel du travail, la
convergence n'a pas fini de le faire partout.

**Sa question sur le plus grand produit valide de `structure` :** vérifié sur
`6_5_representable_atteignable_stable_b0.02_g0.json`, déjà sur disque. Le code
compositionnel s'ajuste presque exactement (écart 3,5e-4). Les trois codes
aléatoires ne s'ajustent quasiment pas (écart ~0,98). Et R = 25 apparaît dans
mon propre échantillon de dix, au-dessus de son candidat 23. Il n'y a pas de
plafond combinatoire sous 27 : 27 s'atteint par construction sur le seul code
que l'architecture est faite pour écrire, et tout ce qui est en dessous en phase
3 est un optimum local de la montée depuis l'aléatoire — le même phénomène que
§1.28/§1.29 pour tabulaire, pas une limite des 81 poids.

Trois questions posées en retour, dont celle qui menace le plus : si le manque
à converger touche une fraction non négligeable des runs de ce projet, tout
tableau stratifié par collisions moyenne peut-être sur un mélange de plateaux
et de transitions sans le savoir.

Réponse dans `docs/REPONSE_ORDRE26.md`.

### 7.43 Vingt-sixième critique : sa mécanique de coïncidence tient à quatre chiffres, et le défaut se retrouve deux fois de plus sans le chercher

25/08/2026. Il prix mon explication contre le tableau réel : l'écart entre les deux
moyennes de `structure` n'est pas « sous 1e-4 », il est de **6,22e-07** — mon
« 0,1 collision de différence, sous une erreur type » en prédisait un de 3,7e-3,
5955 fois trop grand. **Vérifié : exact, à mon propre tableau.**

**Son mécanisme de coïncidence tient au chiffre près.** Reclasser la graine
anomale par sa propre récompense molle (C=3→4) fait pointer les deux bras sur la
même somme de collisions, 40 partout. Reconvolution moi-même de sa loi mise en
commun {2:0,10 3:0,30 4:0,30 5:0,15 6:0,10 7:0,05} sur dix tirages :
**P(égalité) = 0,06829, soit 1 sur 14,64** — son chiffre à quatre décimales.

**Sa question de clôture, testée avec un montage plus propre que celui qu'il
demandait.** Plutôt que de reconstruire son `plafond_beta` à deux agents libres,
j'ai fait tourner `tabulaire` et `factorise` en phase 3 — déjà deux agents
GENUINEMENT libres, aucun échafaudage gelé. La bande de ε survit :
tabulaire (6,10e-05 ± 7,7e-07 sur 13), factorisé (6,37e-05 ± 1,5e-06 sur 15) —
mais la constante n'est pas universelle : structure était à 1,125e-4, environ le
double. Hypothèse non testée : les 81 poids partagés de `structure` ralentissent
l'affûtage final, chaque pas sur un référent perturbant les 26 autres par les
mêmes poids.

**Et le défaut se retrouve deux fois de plus, sans être cherché.** Deux graines
de `tabulaire` sortaient de la bande (3,85e-2 et 4,01e-2) ; reclassées par leur
propre récompense molle, les deux retombent proprement dans la bande
(5,90e-5 et 6,28e-5). Trois vérifications sur trois ont trouvé le défaut.

**Conséquence pour tout le projet, pas seulement pour ce tableau :** tout tableau
stratifié par collisions qui n'a jamais vérifié qu'une graine porte un accord
entre son compte de collisions par argmax et sa récompense molle moyenne un
mélange non détecté. Au taux observé ici (1 à 2 sur 10-15), un tableau à
quelques centaines de graines par cellule tient probablement ; un tableau à dix
ou vingt par cellule — la plupart des tableaux de ce projet — devrait être
vérifié avant d'être cru.

**Sa corrélation ε~R non tranchée, testée à deux endroits de plus :**
tabulaire −0,43, factorisé −0,06, contre son +0,52 sur structure. Signe instable
d'une construction à l'autre, et l'erreur type à n = 13-19 est ~0,27-0,29 :
aucune des trois ne survivrait seule. Rapportée comme mesure sous-puissante,
pas comme énigme résolue.

Réponse dans `docs/REPONSE_ORDRE27.md`.

### 7.44 Vingt-septième critique : mon histoire de dose est fausse à la prémisse, et l'expérience décisive tient une fois lancée au bon endroit

25/08/2026. Il retourne trois choses contre moi. D'abord `factorise` a **39
paramètres libres par référent**, pas un chiffre entre 3 et 27 — vérifié :
81+243+729 = 1053, /27 = 39. **Ce n'est pas un partage partiel, c'est un
partage nul** avec une paramétrisation locale plus grande. Mon histoire de
dose (aucun < partiel < total) était fausse à la prémisse : l'axe réel est
binaire, partagé (`structure`, 3/référent) contre non partagé (`tabulaire` 27,
`factorise` 39, tous deux loin de `structure`).

Ensuite, mes deux anomalies « reclassées dans la bande » du tour précédent ne
le sont pas : re-vérifiées contre mes propres 13 valeurs, l'une tombe sous le
minimum réel, l'autre au-dessus du maximum réel — P(les deux dehors, des deux
côtés) = 1/105 sous échangeabilité, vérifié par le calcul combinatoire standard
2/(15×14). J'avais lu un ratio de 653x comme un retour à la normale sans
vérifier le résidu contre la bande elle-même.

Enfin il propose l'expérience décisive : faire tourner `tabulaire` au nombre de
pas où son modèle exponentiel prédit que la moyenne de ε rejoint celle de
`structure` (2811 pas). **La prédiction du nombre de pas était fausse** — à
2811 pas la moyenne vaut 6,98e-05, à peine bougée depuis 3000 pas, loin de
1,125e-4. Cohérent avec un fait déjà établi au tour 25 pour `plafond_beta` :
la convergence n'est pas une exponentielle unique sur toute sa plage, elle
décélère tôt. Balayé cinq points sur une graine pour trouver le vrai
croisement (~2190 pas), puis lancé quinze graines à 2200 pas :

```
structure (publié)              moyenne 1,125e-04   CV 15,91 %
tabulaire @ 2200 (appariée)      moyenne 1,1232e-04  CV 1,45 %
```

**Moyennes égales à quatre chiffres, CV différent d'un facteur 11.** Son test,
lancé au bon endroit, répond exactement ce qu'il devait répondre : la
dispersion est propre à la construction, pas une fonction de la magnitude de
ε. Et une quatrième instance du même défaut de reclassement apparaît dans ce
même lot (une graine sur quinze, reclassée de R=25 à R=24, retombe dans la
bande) — quatre fois sur quatre constructions/pas différents maintenant.

Non résolu : le mécanisme du pourquoi le partage rend le taux lui-même
dépendant de la graine. Une seule construction partagée dans le projet ; pas
de deuxième point pour savoir si c'est générique au partage ou propre à ces
81 paramètres.

**Contrôle tenté pour isoler partage contre simple nombre de degrés de
liberté, et raté honnêtement.** `EmetteurMasque` : le même tenseur 27×27 que
tabulaire, mais un hook de gradient n'autorise que 3 colonnes libres par
ligne, tirées au hasard — 3 paramètres libres par référent comme `structure`,
mais **sans aucun partage** entre lignes. Si le CV restait serré, le partage
serait isolé comme cause plutôt que le compte brut de paramètres. Quinze
graines, mêmes 2200 pas :

```
R bloqué a 21-24 (13/15 a R=23, jamais 25 ni 26)
eps entre 0,038 et 0,161 — deux a trois ordres au-dessus de la bande 1e-4
```

**Pas un résultat, un contrôle cassé.** Geler 24 logits sur 27 près de zéro ne
réduit pas seulement les degrés de liberté, ça gèle un plancher de
représentabilité : les 3 colonnes libres doivent dominer 24 logits presque
uniformes avant qu'une ligne s'affine, et la plupart n'y arrivent pas en
2200 pas. `structure` n'a pas ce plancher — sa construction additive par
position laisse le gradient des autres référents pousser sur les mêmes poids,
donc rien n'y reste gelé près de zéro. Le contrôle a changé la
représentabilité, pas seulement le compte de paramètres, ce qui écrase la
comparaison visée de deux à trois ordres de grandeur. Un contrôle valide
demanderait une construction partagée capable d'atteindre R=26 comme les
trois autres, en ne faisant varier que le partage — pas encore construite.

**Dernière vérification, sur une hypothèse jamais posée pendant tout ce
tour :** `monter()` optimise `J = E[R] + β(H_S+H_R)`, pas `E[R]` seul.
Est-ce que le terme d'entropie fixe lui-même un équilibre à ε non nul, plutôt
que le manque de pas ? Calculé pour une ligne proche de la saturation :
`p* = e^(c/β)/(26+e^(c/β))`, c ≈ 1 au voisinage du point. À β = 0,02,
c/β = 50, résidu analytique `26·e⁻⁵⁰ ≈ 5,0e-21` — vérifié aussi numériquement
sur une optimisation jouet à 200 000 pas, qui atterrit à `1-p* = 2,69e-12` et
descend encore. **Seize ordres de grandeur sous tout ε mesuré ce tour
(6,1e-05 à 1,1e-04).** La prémisse de tout le tour tient : ε est un retard
d'entraînement, pas un point fixe caché de l'objectif régularisé — vérifié
plutôt que supposé.

Réponse dans `docs/REPONSE_ORDRE28.md`.

### 7.45 Vingt-huitième critique : les quarante points sont quinze points mesurés trois fois

25/08/2026. Il demande la corrélation poolée centrée par groupe de pas avant
d'écrire une phrase de plus sur le signe. Faite : **r = −0,3007, df = 36,
p = 0,067** — pas significatif à 5 %, mais du signe de mon −0,43, à 0,02 du
seuil qu'il calcule (0,3202, identique au mien).

**Avant de le croire, vérifié ce que sont réellement les trois groupes.**
Les quinze valeurs de R sont **identiques** entre 3000 et 2811 pas, graine par
graine — les trois expériences partagent le même `default_rng(999)`, seul
`pas` change. Ce ne sont pas quarante points indépendants, ce sont quinze
graines mesurées trois fois. Le df = 36 surcompte l'information indépendante
d'un facteur ~3 ; le vrai n est ~13-15, pas 40, et la puissance à 78 %
annoncée ne tient pas.

**Le test correctement indépendant :** une ligne par graine, R (constant
confirmé) contre ε moyenné sur les trois budgets, treize graines qui ne
sortent jamais de bande : **r = −0,3908**. Proche de mon −0,43 d'origine,
même signe, sur un n défendable. Le désaccord de signe avec son +0,52 n'est
pas résolu par le pooling — il n'y avait jamais plus d'information
indépendante que la première mesure n'en portait.

**Les deux graines exclues, observées plutôt qu'inférées — et une correction
apportée à ce que je viens d'écrire ci-dessus.** J'avais appelé le
comportement de la graine idx5 un « plateau », signature de point critique
sous-optimal comme §1.28/§1.29. **Faux, trouvé en vérifiant contre le bon
dénominateur.** Contre R=25 fixe plutôt que le R lu par argmax, ε décroît de
façon lisse et monotone sur toute la trajectoire (2200 à 40 000 pas), sans
aucune discontinuité aux deux points où l'argmax bascule (2500-2600 et
17000-19000 pas). Le « plateau » était un artefact de dénominateur : je
divisais par 26 au lieu de 25 pendant la fenêtre où l'argmax rapportait
26 par erreur.

**Ce qui reste vraiment non résolu à 300 000 pas, et c'est une meilleure
question.** Les deux référents perdants (0 et 4, en compétition pour les
messages 8 et 10) n'ont jamais fixé leur second choix : le candidat
préféré du référent 0 change presque à chaque relevé — messages 0, 11, 7, 3,
16, 8 à 10k/20k/40k/80k/150k/300k pas — avec des marges qui oscillent entre
microscopiques (5e-6, 7e-6) et simplement petites (4-9e-4), jamais stable.
**Un référent qui a déjà perdu sa compétition de message ne reçoit aucun
gradient de récompense sur son second choix** : parmi les 26 messages
perdants, aucun ne rapporte rien, donc rien ne fixe lequel il pointe. Le
flottement R=25↔26 observé plus tôt est exactement ça : quand le candidat
errant du référent 0 atterrit par hasard sur un message libre, sa collision
locale disparaît et le compte global lit 26 ; quand il redérive sur un
message déjà pris, il relit 25. **Le compte de collisions n'est pas instable
parce que le système n'a pas convergé — E[R] est convergé à neuf décimales
dès 20 000 pas. Il est instable parce que les référents perdants ont une
direction plate dans l'objectif, que la montée exacte n'a aucune force pour
fixer.**

Trois questions posées, aucune déjà envisagée par l'un de nous trois : le
compte de collisions par argmax est-il seulement défini pour un référent qui
a déjà perdu ; est-ce la vraie raison du 92 % de REINFORCE contre 5 % de la
montée exacte (§7.36) — dériver dans une direction plate plutôt qu'échapper
un bassin profond ; et cette direction plate est-elle présente dans tous les
runs à collisions multiples de ce projet, y compris ceux d'avant cet échange.

Réponse dans `docs/REPONSE_ORDRE29.md`.

### 7.46 Vingt-neuvième critique : deux régimes sous une seule étiquette « perdant »

30/08/2026. Il concède la mise en commun (« il n'y avait jamais plus
d'information indépendante que la première mesure n'en portait »), reconstruit
`objectif()` en numpy indépendamment (E[R] = 0,9259259257 contre mon
0,9259259243 publié), et montre que le gradient de récompense sur un référent
perdant **n'est pas nul** — il varie avec β, et la masse 0,5/0,5 du récepteur
sur une collision est un vrai partage de récompense, pas un zéro.

**Rejoué sur mon propre état entraîné avant de le croire.** Vérifié : deux
des quatre référents perdants (10, 24) ont un gradient de récompense
indiscernable des gagnants ; les deux autres (0, 4) sont 30 000 à 60 000 fois
plus petits.

**Première lecture, fausse — corrigée en vérifiant qui collisionne avec qui.**
J'avais apparié 10 avec 24 comme s'ils étaient à égalité l'un contre l'autre.
**Ils ne collisionnent pas ensemble.** Les vraies paires : référent 0
(uniforme) partage le message 7 avec le référent 24 (engagé) ; référent 4
(uniforme) partage le message 16 avec le référent 10 (engagé). Et le
récepteur a déjà tranché entièrement les deux :

```
message  7 :  R[7,0]  = 1,27e-11    R[7,24]  = 1,00000000
message 16 :  R[16,4] = 1,39e-11    R[16,10] = 1,00000000
```

**Ce n'est pas un duopole 0,5/0,5. C'est un référent confiant qui capte toute
l'attention du récepteur et un référent hésitant qui n'en capte aucune.** Le
gradient « pleine puissance » du référent 24 n'est pas une pression pour
abandonner le message 7 — c'est le terme générique d'affûtage
entropie-contre-récompense que porte toute ligne pleinement engagée, gagnante
ou non. Le référent 0 n'a pas un gradient minuscule parce qu'il est à égalité
avec 24 : il l'a parce que, pour chacune de ses 27 options, le message est
soit libre (aucun signal de récompense, seulement l'entropie) soit déjà
capté en totalité par un autre référent confiant. **Le référent 0 n'est pas
en train de perdre un match nul qu'un peu de bruit ferait pencher — il a été
exclu de la fonction de récompense sur toute la ligne, avant même d'avoir
choisi quoi que ce soit.**

La vraie question n'est donc pas si une égalité symétrique se rompt. C'est
**si le référent 24 abandonne un jour le message 7** — et rien dans le
gradient déterministe ne l'y pousse, puisqu'il a déjà tout gagné là. La
collision ne se résout pas quand le référent 0 bouge ; elle ne peut se
résoudre que si le référent 24 bouge le premier, pour une raison qui n'a
rien à voir avec le référent 0.

**Balayage β, arrivé, et décisif contre son hypothèse.** Montée exacte contre
REINFORCE, taux de bijection à β ∈ {0 ; 0,005 ; 0,02}, dix graines par
cellule, 20 000 pas :

```
   beta   bij. exacte   colls exacte   bij. REINFORCE   colls REINFORCE
  0,000      0/10           1,80            10/10              0,00
  0,005      0/10           1,90            10/10              0,00
  0,020      0/10           1,50             9/10              0,10
```

**Le fossé ne bouge pas.** 0/10 en montée exacte aux trois β, 9-10/10 en
REINFORCE aux trois β. Il avait posé le test comme une fourche lui-même : si
le fossé se referme à β=0, l'entropie laissait la place ; s'il tient, le
bruit fait le travail et son compte est faux. **Il tient — son propre test
décisif réfute sa propre hypothèse.** Cohérent avec le mécanisme corrigé
ci-dessus : le gradient du référent 24 est l'affûtage générique, indépendant
de β, donc couper β ne change rien à pourquoi il reste.

Trois questions posées, révisées après la correction : un référent engagé
abandonne-t-il jamais volontairement, ou la collision ne se résout-elle que
si SA propre ligne est perturbée par autre chose que le référent qu'il
bloque ; l'exclusion quasi totale du référent hésitant est-elle permanente
une fois qu'un autre s'engage le premier, ou un pur effet de timing ; et le
« compte de collisions » a-t-il toujours mesuré deux choses empilées — le
nombre de référents qui ont perdu la course à l'engagement, et séparément,
lesquels des gagnants n'ont pas encore été délogés.

Réponse dans `docs/REPONSE_ORDRE30.md`.

### 7.47 Trentième critique : les égalités sont la majorité, mon échantillon ne pouvait voir que des murs

30/08/2026. Il reproduit mon fossé 92 %/5 % sur SA propre réimplémentation
numpy (E[R] identique, générateur différent) — pas un artefact de graine.
Puis il isole un facteur que ma table n'avait pas : la montée exacte tourne à
lr 0,05, REINFORCE à lr 0,01. À lr 0,01, la montée exacte fait **5/30**, pas
0/30. Et il recense 50 collisions sur 30 graines : **42 égalités (0,1-0,9 des
deux côtés) contre seulement 8 murs.** La graine idx5 (mes deux collisions,
deux murs) n'est pas représentative.

**Tout reproduit indépendamment, mon code, mes graines.**

- **Contrôle lr :** 2/10, 1/10, 0/10 aux trois β — 3/30 contre son 5/30, même
  ordre, même sens.
- **Recensement égalité/mur :** 52 collisions, **41 égalités (79 %) contre
  11 murs (21 %)** — accord étroit avec ses 84 %/16 %.
- **Motif « engagement puis évacuation » :** le référent 0 (mon mur) culmine à
  **93 % au pas 100**, puis s'effondre à l'uniforme au pas 1000 — je n'avais
  jamais regardé entre le pas 0 et 2200 à grain fin. Confirmé sur ses huit
  graines aussi.

**Pourquoi mon seul échantillon donnait 2 murs sur 2.** Vérifié : les deux
membres d'une égalité sont CHACUN engagés individuellement à S≈0,9999999661
— stables, aucun scintillement d'argmax. Le membre perdant d'un mur, lui,
reste à l'entropie uniforme et son argmax lit le bruit flottant parmi
27 options à égalité. **J'avais trouvé idx5 en cherchant une instabilité de
l'étiquette R — et seuls les murs produisent cette instabilité.** Le
détecteur ne pouvait voir que la classe minoritaire.

**Et l'égalité ne ressemble pas à une convergence lente.** Suivi un cas
(référents 23/25, message 13) de 20 000 à 300 000 pas : écart à 0,5 exact de
+2,27e-4, −8,81e-6, +6,16e-7, +3,49e-8, **+3,09e-5** — non monotone, quatre
ordres de grandeur de resserrement puis remontée de trois ordres, signe
alterné. Pas une asymétrie qui se résorbe : une oscillation autour du point
symétrique, cohérente avec un vrai point fixe protégé par l'équivariance de
§6.7, pas une course lente vers un gagnant.

Réponse dans `docs/REPONSE_ORDRE31.md`.

### 7.48 Trente-et-unième critique : le mur n'est pas une collision, et la place « libre » a un vrai tirage, juste noyé dans un réseau

30/08/2026. Il perturbe une égalité (au lieu de la regarder) : `R += eps` sur
un membre, `-= eps` sur l'autre, puis 20 000 pas de plus. Jusqu'à eps=8 (masse
receveur 1,000000, capture totale), le partage revient exactement à 0,5/0,5.
Seul eps=12 tient. **Reproduit à cinq décimales sur ma propre égalité**,
même seuil eps=8/eps=12.

**Et le croisement décisif :** confiance de l'émetteur (S) contre étiquette
du récepteur, 30 graines, 52 collisions. **Séparation parfaite, zéro
exception :** les 41 égalités ont S=1,000000 des deux côtés ; les 11 murs ont
leur membre bas à l'entropie maximale (0,037066-0,037163, contre son
0,03709-0,03717). **Un mur n'est pas une collision.** C'est un référent qui
n'a jamais rien engagé, classé sous le message que le bruit flottant lui
assigne. Les vraies collisions sont **100 % des égalités**, pas 84 %, pas
79 %. Ma rétractation du tour précédent n'était pas fausse, elle n'allait pas
assez loin.

**Le message « libre » ne l'est pas non plus — mais je m'étais trompé sur
pourquoi, et je me corrige ici plutôt que de laisser passer.** J'avais décrit
les référents 18 et 25 comme « engagés ailleurs avec confiance ». Faux,
trouvé en posant la question que j'aurais dû poser avant d'écrire ça : à quoi
ressemble VRAIMENT l'argmax du référent 18 ? `S[18,0]=0,499479` et
`S[18,8]=0,500521` — il n'est engagé nulle part, il est scindé quasi
50/50 entre les deux, et l'est **depuis le pas 5000 au moins** (vérifié
jusqu'à 40 000). Même chose pour le référent 25 entre les messages 1 et 14.
**Aucun des deux ne « revendique à moitié » le message d'un autre — chacun
est une ligne scindée entre DEUX DE SES PROPRES options, sans adversaire
réel sur aucune des deux.**

**Vérifié si ça coûte quelque chose : non.** `R[0,18]=1,000000` ET
`R[8,18]=1,000000` — le récepteur décode le référent 18 avec pleine
confiance, quel que soit celui des deux messages qu'il envoie, puisqu'il en
est le seul expéditeur significatif. Sa récompense espérée
(0,4995×1 + 0,5005×1 = 1,000000) est identique à un engagement plein. La
scission coûte **zéro récompense** et achète `ln 2 = 0,693` nats d'entropie
de ligne — exactement ce que β est censé récompenser. **Ce n'est pas une
égalité bloquée : c'est vraisemblablement le véritable optimum de
l'objectif** pour un référent à deux options gratuites et non contestées.
Preuve mince (une ligne, une graine) — signalé comme tel. Si ça généralise,
« collision » recouvre trois choses, pas deux : les vraies égalités
(coûteuses, protégées par la symétrie), les murs (récompense nulle), et
maintenant ceci — une scission optimale et gratuite, qui ne bloque rien du
tout.

**Doute soulevé sur le bassin lui-même, avant de le croire.** Sa perturbation
et la mienne touchaient toutes deux le RÉCEPTEUR seul, laissant les deux
émetteurs parfaitement symétriques — la « récupération » pourrait n'être que
le récepteur qui rattrape mécaniquement deux émetteurs jamais touchés.
Testé en perturbant l'ÉMETTEUR à la place (référent 25, message 13) :
jusqu'à eps=12 (qui cassait définitivement côté récepteur), tout revient à
S=1,000000 après 20 000 pas de plus ; il faut eps=20 pour casser, et le
référent cassé **s'évacue vers le mur** (S≈1/27), pas vers un troisième
message. Le bassin est réel, plus large côté émetteur que côté récepteur (un
logit émetteur affronte 26 concurrents, un logit récepteur dans une égalité
n'en affronte qu'1) — une asymétrie que je n'aurais pas trouvée sans tester
la version la plus faible de ma propre affirmation en premier.

**Recherche de contre-exemple, revenue bredouille, rapportée quand même.**
Mon classificateur ignore silencieusement toute collision à 3 référents ou
plus (`len(refs) != 2: continue`). Vérifié sur les 30 graines : zéro
collision à 3, zéro à plus. Le trou existait, la population qu'il aurait pu
manquer n'était pas là.

**Sa question de clôture, testée plutôt que débattue.** Calculé le gradient
complet à 27 directions de la ligne uniforme du référent 0, vérifié contre la
forme fermée (accord à 2e-28, quinze ordres sous le signal — pas du bruit
numérique) : les deux messages libres portent les deux plus GRANDES valeurs
positives de toute la ligne (+5,41e-13, +5,29e-13), loin devant les 25
messages pris. **Ni un vrai second attracteur, ni une absence totale de
traction** — un vrai gradient, correctement dirigé, cinq ordres de grandeur
sous celui d'une ligne engagée. Et compte tenu de ce qui précède, « pris dans
un réseau » était aussi la mauvaise description : le référent 0 ne fait pas
face à une égalité concurrente sur le message 0, il fait face à une scission
gratuite qui ne lui dispute rien. Le mur n'est peut-être bloqué par rien
d'autre que son propre gradient, trop petit pour bouger à n'importe quel
budget testé jusqu'ici.

Non reproduit : son recensement de timing (14/30 graines résolvent une
collision dure, la seule bijection se règle au pas 271 sur 20 000). Dit
comme tel plutôt que supposé.

Réponse dans `docs/REPONSE_ORDRE32.md`.

### 7.49 Trente-deuxième critique : la scission et le mur sont UNE occupation, lue des deux bouts — correction de ma correction

31/08/2026. Il montre que « message libre » et `R[0,18]=1,000000` ne peuvent
pas être vraies telles quelles en même temps sans le dire : le gradient
minuscule que j'avais trouvé n'est pas une préférence indépendante du
référent 0, c'est le résidu de ce que laisse un récepteur déjà saturé
ailleurs. Il propose l'expérience qui tranche : pousser le référent 0
lui-même, côté émetteur, sur le message 0, et regarder si le référent 18 se
rescinde (son lecture « optimum libre ») ou s'engage pleinement ailleurs
(sa lecture « bail sur une vacance »).

**Lancée. Ni l'une ni l'autre lecture n'est vraie sur toute la plage — un
seuil net entre eps=23 et eps=24 :**

```
eps <= 23 : le referent 0 retombe toujours a l'uniforme, la scission 18 intacte
eps = 24  : bascule complete — 18 s'engage sur 8 seul, 0 garde le message 0
```

**Les deux lectures sont vraies, chacune d'un côté du seuil.** La scission
dépend de l'occupation, exactement comme il l'argumentait, ET elle est
protégée par un vrai bassin, de la même forme que l'égalité 23/25. Ce ne
sont pas deux structures différentes : c'est la même, avec un troisième
larron actuellement endormi plutôt qu'actif. **Je retire « optimum
authentique, permanent »** du tour précédent — c'était un optimum
conditionnel à l'état du référent 0, pas un optimum absolu, et c'est
précisément ce que son expérience était construite pour révéler.

**Sa question sur l'entropie, testée plutôt que raisonnée.** Déséquilibré
artificiellement la scission à 30/70, relancé 40 000 pas sans rien
d'autre : ça revient à 50,07/49,93. La récompense est plate entre les deux
branches tant que le référent 0 dort — si β ne faisait rien, un 30/70
resterait à 30/70. **β est la seule force active dans ce régime, et elle
force le retour exact au point symétrique.** Réponse à sa question : non,
l'entropie n'achète pas rien ici — c'est la seule chose achetée, et c'est
le même mécanisme qui tient une vraie égalité contestée.

Ses deux corrections mineures vérifiées exactes : le dépassement de 1/27 sur
les quatre valeurs de mur (inévitable, max de 27 termes sommant à 1) ; et le
déficit d'entropie de la scission, 5,43e-7 nats sous ln 2, confirmé au
chiffre près.

**Poursuivi sans qu'on me le demande, parce que le tour semblait fini et que
je m'en méfie maintenant.** Deux vérifications de plus.

Le seuil eps=23/24 tient-il vraiment, ou n'est-ce qu'un instantané à
40 000 pas ? Étendu à 270 000 pas cumulés des deux côtés : **stable
intégralement**, bascule comme retour.

Un occupant scindé (référent 18) est-il aussi difficile à évincer qu'un
vrai gagnant solo ? Jamais comparé. Poussé le référent 0 sur le message du
référent 1 (propriétaire exclusif, pleine récompense en jeu) : **à eps=100,
quatre fois le seuil qui évince le référent 18, le référent 1 ne bouge pas
du tout**, et le récepteur ne donne jamais le moindre crédit au référent 0
là (`R[11,0]=0` à tous les eps testés). Un gagnant solo qui gagne une vraie
récompense est catégoriquement plus dur à déloger qu'un occupant scindé qui
ne gagne rien de plus en restant scindé.

**Et un engagement gaspillé (S=1,0 sur un message qui rapporte zéro) reste
stable 270 000 pas de plus — ce que je n'attendais pas et ne crois pas
encore entièrement.** Par l'argument d'entropie du même tour, ça devrait
être strictement dominé : revenir à l'uniforme ne coûte rien en récompense
(déjà nulle) et gagne `ln 27` d'entropie. Ça ne revient pas. Hypothèse
énoncée comme hypothèse : la perturbation (eps jusqu'à 100) a probablement
saturé la ligne au-delà de la précision représentable en float64 près de 1,
et le gradient renvoyé par autograd y sous-dépasse peut-être exactement à
zéro plutôt que d'être seulement petit — auquel cas ce n'est pas un second
vrai piège de l'objectif, c'est un artefact numérique de ma propre poussée,
à revérifier avec une perturbation beaucoup plus modeste avant d'y croire.

Réponse dans `docs/REPONSE_ORDRE33.md`.

### 7.50 Trente-troisième critique : ce n'était pas float64, c'était l'epsilon d'Adam — et le seuil 23/24 était gonflé, pas faux

01/09/2026. Il identifie la vraie cause de l'engagement gaspillé permanent :
pas float64, pas autograd — **l'epsilon d'Adam** (défaut 1e-8), qui rend la
mise à jour non homogène en échelle sous ce plancher (`lr·g/eps` au lieu de
`lr·g/(|g|+eps)`). Table de gradient d'entropie seule reproduite au chiffre
près (2,56e-12 à 7,16e-44 pour gap 26 à 100). Sa prédiction sur le référent 1
: **exacte** — `adam_eps=1e-10` au lieu du défaut, et l'engagement gaspillé
disparaît entièrement (`S[0].max()=0,037037`, exactement 1/27). Sa question
bon marché aussi : la marge en espace logit aux trois points de contrôle à
eps=100 est **strictement gelée à 96,741751**, aucune dérive sur 270 000 pas.

**Mais sa seconde prédiction — que le seuil 23/24 tiendrait sous le même
changement, preuve que c'est de la vraie dynamique — ne s'est pas
vérifiée.** Sous `adam_eps=1e-10`, eps=23 (qui revenait à l'uniforme sous le
défaut) **bascule aussi**. Poussé plus loin : le seuil descend et **converge**
entre 18 et 20, stable de 1e-12 à 1e-16 :

```
                eps=18   eps=20   eps=23
adam_eps=1e-8   retour   retour   retour
adam_eps=1e-10  retour   retour   BASCULE
adam_eps=1e-12  retour   BASCULE  BASCULE
adam_eps=1e-16  retour   BASCULE  BASCULE
```

**Les deux avaient en partie raison.** Le seuil est une vraie propriété de la
dynamique (il converge vers une valeur indépendante d'`adam_eps`) — mais le
23/24 publié n'était pas cette valeur : gonflé de 4 à 6 unités de logit par
le plancher par défaut de l'optimiseur. La vraie frontière se situe vers
18-20.

Réponse dans `docs/REPONSE_ORDRE34.md`.

### 7.51 Trente-quatrième critique : aucun des trois candidats simples ne correspond au point de transition — rapporté tel quel

01/09/2026. Il retire sa propre erreur de calcul (calcul du gel en isolation
sur une seule ligne, alors que la frontière appartient à quelle que soit des
trois lignes qui se dégèle en premier) et propose le test qui trancherait :
extraire `sqrt(v)` de l'état interne d'Adam pour la ligne du référent 0 à la
configuration frontière, et vérifier qu'il passe sous `adam_eps` exactement
là où le seuil bascule.

**Testé sur les trois candidats que sa reformulation nomme. Aucun ne colle
proprement.**

- **Référent 0 (sa prédiction nommée) :** `sqrt(v)` reste TOUJOURS au-dessus
  d'`adam_eps`, y compris quand ça revient. Pire pour une histoire propre :
  `sqrt(v)` lui-même chute de quatre ordres entre les réglages qui reviennent
  et ceux qui basculent — ce n'est pas une quantité fixe comparée à un seuil
  mobile, c'est une quantité qui dépend elle-même du réglage qu'on teste.
- **Référent 18 :** `sqrt(v)` est déjà 5 ordres AU-DESSUS d'`adam_eps` à
  1e-8 et 1e-10 — donc pas gelé du tout à ces réglages — et pourtant le
  système revient quand même. Exclu comme goulot unique.
- **Récepteur (R[0,18]) :** le plus proche. `sqrt(v)` reste stable
  (2 à 5×10⁻¹³) pendant qu'`adam_eps` balaie autour ; le croisement a bien
  lieu, mais entre 1e-12 et 1e-14 — un cran après le vrai basculement
  comportemental, situé entre 1e-10 et 1e-12.

**Rapporté tel quel plutôt que forcé.** Sa reformulation (un min sur
plusieurs lignes) est la bonne direction, mais aucun des trois candidats
évidents n'explique proprement le point de transition observé. Plus probable
: une transition couplée entre les trois lignes partageant le même
`adam_eps`, pas une course entre seuils indépendants — non démontré, dit
comme non résolu. Ce qui tient toujours : la convergence du seuil vers
18-20 elle-même, indépendamment de quel paramètre en porte l'explication.

**Poursuivi après relance de Théo (« tu n'as pas cherché plus loin ? »,
« combien de fois je te le dis là ? »).** L'outil que je disais ne pas avoir
construit — le suivi pas à pas des trois quantités, pas leur instantané
final — construit et lancé.

**Les deux bras sont quasi identiques jusqu'au pas 50, puis divergent
entièrement au pas 200.** Au pas 50, `R[0,0]` (crédit du récepteur au
référent 0) est **235 fois plus grand** sous `adam_eps=1e-12` que sous
`1e-10` (2,02e-07 contre 8,62e-10), et son `sqrt(v)` est 104 fois plus
grand — alors que `S[0,0]` vaut 1,0000 dans les DEUX bras à ce stade, et que
la ligne du référent 18 n'a encore bougé nulle part. **La bifurcation a lieu
sur le paramètre du récepteur, dans les cinquante premiers pas, avant que la
moindre ligne d'émetteur ne diverge.**

**Le mécanisme, lu sur la trace plutôt que deviné :** juste après la
perturbation, le référent 0 est artificiellement confiant (poussée brute de
+20, pas encore méritée) et le vrai gradient du récepteur vers lui est
encore minuscule. Sous `adam_eps=1e-12`, ce gradient minuscule n'est pas
noyé par le plancher de l'optimiseur comme il l'est sous `1e-10` — le
récepteur fait donc un vrai pas, normalisé par `v`, vers le référent 0
pendant que sa confiance gonflée est encore là pour être récompensée. Sous
`1e-10`, cette même fenêtre est perdue à cause du plancher, et le temps que
`adam_eps` cesse de compter, la ligne du référent 0 est déjà revenue à 1/27
(dès le pas 500) — plus rien à quoi le récepteur pourrait s'accrocher.
**C'est une course entre la vitesse à laquelle la confiance gonflée du
référent 0 se dissipe et la vitesse à laquelle le récepteur peut y réagir —
`adam_eps` règle le temps de réaction du récepteur, pas le `v` final d'une
ligne.** Explique pourquoi aucun des trois candidats du tour précédent ne
collait : je lisais l'état après coup, pas la course elle-même.

Réponse dans `docs/REPONSE_ORDRE35.md`.

### 7.52 Trente-cinquième critique : loquet, pas course — confirmé par un calendrier, un bras ne revient jamais même à K=1000

02/09/2026. Il precise ma lecture « course » : au pas 1, `sqrt(v)` du
récepteur est identique dans les deux bras à trois chiffres près (l'état
initial ne peut pas encore avoir vu `adam_eps`), donc ce n'est pas le
récepteur qui « part en retard » — c'est un **loquet** : sous `adam_eps`
trop grand, le pas reste sous son propre `eps` à chaque pas enregistré (max
0,25× à `1e-10`), jamais assez pour s'échapper ; sous `1e-12`, il franchit
son propre `eps` dès le pas 10 et n'a plus jamais besoin de lui. Propose le
test qui tranche : un `adam_eps` en deux temps, bras A (`1e-12` puis
`1e-10` après K pas) et bras B (l'inverse) — course et loquet prédisent la
même chose en gros mais avec un seul point de bascule K commun (course) ou
deux horloges très différentes (loquet).

**Testé. Loquet, sans ambiguïté :**

```
A (1e-12 -> 1e-10 apres K)      B (1e-10 -> 1e-12 apres K)
K=30   retour                   K=200   retour
K=50   BASCULE                  K=500   retour
K=200  BASCULE                  K=1000  retour  <- ne bascule JAMAIS
```

Le bras A bascule pile dans la fenêtre 10-50 qu'il prédisait. Le bras B ne
bascule **à aucun K testé, jusqu'à K=1000** — vingt fois au-delà de sa
propre fenêtre prédite. Pas une ligne d'arrivée commune : une porte qui se
ferme une fois, dans les cinquante premiers pas, et que rien ne rouvre
ensuite.

**Conséquence pour 18-20 :** ce n'est plus une constante « corrigée de
l'artefact ». C'est la frontière sous un optimiseur capable de réagir dans
les cinquante premiers pas.

**Vérifié trois autres leviers censés donner le même genre de gain précoce
au récepteur — aucun ne se comporte comme `adam_eps`.** `lr` de 0,05 à 0,15 :
douze cellules, aucune ne bascule. `beta2` de 0,999 à 0,5 (v suit le
gradient 500 fois plus vite) : douze cellules, aucune ne bascule — alors que
c'est le mécanisme même que sa trace pas-à-pas identifiait. `lr` poussé à
des valeurs absurdes (jusqu'à 40× la base) : bascule seulement à `lr=2,0`,
et de façon non monotone (eps=18 et 20 basculent, eps=23 revient). **Si
« tout ce qui donne un gain précoce déplace la frontière » était la bonne
généralisation, `beta2` aurait dû être le levier le plus net de tous. Il
n'a rien fait.** Le mécanisme est donc plus spécifique à `adam_eps` — un
plancher additif au dénominateur — qu'une histoire générique de réactivité
précoce ne le prédirait.

Ce qui reste : une vraie frontière de bassin en espace logit, entre la fin
d'adolescence et le début de la vingtaine pour cette graine sous réglages
ordinaires, décidée dans une fenêtre si précoce et si spécifiquement liée à
une seule constante additive qu'aucun nombre unique n'en rend compte
honnêtement. Le balayage est le résultat, et il ne bouge que sur un seul
axe, pas les trois attendus.

Réponse dans `docs/REPONSE_ORDRE36.md`.

### 7.53 Trente-sixième critique : un bug dans mon propre critère de bascule, trouvé en vérifiant `eps=1e-6`

02/09/2026. Il démontre par l'algèbre (pas par un test) que `lr` ne peut pas
faire tourner la direction d'une mise à jour Adam (un scalaire commun à
toutes les coordonnées) et que `beta2` ne peut rien changer au pas 1 (la
correction de biais force `v_hat = g²` quel que soit `beta2` à t=1) —
expliquant exactement pourquoi ces deux leviers étaient inertes dans le
tour précédent. Il propose le test qui pourrait falsifier sa lecture :
pousser `adam_eps` à 1e-6, bien au-dessus des `|g|` observés (confirmé :
729/729 coordonnées du récepteur sous 1e-9 dès le pas 1).

**Résultat brut avec mon critère existant : non monotone, eps=23 et 24
basculent sous `1e-6` alors qu'ils ne basculaient pas sous le défaut.**
Plutôt que de le rapporter tel quel, tracé pas à pas — et ça a exposé un
**bug réel dans mon propre critère de classification**, la même faille que
le mécanisme du référent 1 trois tours plus tôt : `S[0,0]` sature à 1,0 et y
reste 40 000 pas, mais **`R[0,18]` reste à 1,000000 tout du long** —
référent 18 ne cède rien, référent 0 envoie à pleine confiance pour zéro
crédit, en permanence. Mon critère (`S[0].max()>0,5`) ne vérifiait jamais si
le récepteur avait vraiment bougé.

**Corrigé, en vérifiant `R[0,18]` sur tout, y compris mes propres chiffres
déjà publiés :**

```
adam_eps=1e-8  (defaut) : 18,20,23 retour | 24 VRAI TRANSFERT        <- inchange
adam_eps=1e-12 (« 18-20 »): 18 retour | 20,23,24 VRAI TRANSFERT       <- inchange
adam_eps=1e-6  (nouveau) : 18,20 retour | 23,24 GELE SANS VALEUR      <- mal classe, corrige
```

**Les deux seuils déjà publiés tiennent** — revérifiés spécifiquement pour
s'assurer que ce bug ne les contaminait pas rétroactivement. `eps=1e-6`
n'est pas un troisième point sur le même axe : c'est le mécanisme de gel
d'Adam qui réapparaît, cette fois sur le paramètre perturbé lui-même plutôt
que sur la ligne où je l'avais trouvé la première fois. Règle ajoutée :
vérifier les deux colonnes (l'état du référent ET celui du récepteur) avant
de rapporter une direction — c'est la même règle que le carnet porte déjà
pour une valeur imprimée qui sature pendant que l'état sous-jacent continue
de bouger, appliquée un cran plus haut sans que je l'aie vu venir.

Réponse dans `docs/REPONSE_ORDRE37.md`.

**Complément, à la relance de Théo (« tu n'as pas cherché plus loin ? »).**
Je n'avais vérifié le bug que sur les deux nombres sur le point d'être
publiés — pas sur les `BASCULE` déjà publiés des trois tours précédents.
Réaudité systématiquement :

- table de convergence (1e-8 à 1e-16) : tous les `BASCULE` déjà publiés
  confirmés **vrais transferts**, inchangés.
- bras A du calendrier (K=50, 100, 200) : confirmés **vrais transferts**,
  inchangés.
- balayage `lr` (0,05 à 2,0) : **`lr=2,0` était aussi mal classé** —
  `R[0,18]` reste à 1,000000, le même artefact gelé.

**Une deuxième victime, qui renforce plutôt qu'affaiblit.** Corrigé,
`lr` produit **zéro vrai transfert sur toute sa plage testée** — plus
d'exception non monotone à excuser comme « une graine, à ne pas
sur-interpréter ». `lr` et `beta2` sont maintenant tous deux totalement
inertes, proprement, et la conclusion « spécifique à `adam_eps` » tient
plus nettement qu'avant la correction.

*(Nuancé le tour suivant, §7.54 : « inerte » ne valait que sur la plage
0,05-2,0 testée. En dehors, `lr` fait tout ce qu'`adam_eps` fait.)*

### 7.54 Trente-septième critique : `eps=1e-6` était une ligne `lr` déguisée — une prédiction confirmée, l'autre corrigée après relance

03/09/2026. Il démontre que sur les 729 coordonnées du récepteur, aucune ne
dépasse `1e-9` — donc `adam_eps` n'y est pas un curseur à cinq réglages,
c'est un **interrupteur entre deux optimiseurs** : à 1e-8 et 1e-6, le
récepteur tourne en SGD à moment, au taux effectif `lr/eps` ; à 1e-14,
Adam à signe sur toute coordonnée. Et puisque `(g+1e-6)/(g+1e-8)` est
uniforme à 0,15 % près sur tout le récepteur (aucune rotation possible,
juste une remise à l'échelle), `eps=1e-6` du tour précédent **était en
réalité une ligne `lr` déguisée** — `lr/100`, pas un cinquième point sur
l'axe `adam_eps`.

**Prédiction 1 (gel à `lr=5e-4`) confirmée sans réserve**, et plus fort que
prédit (eps=20 gèle aussi).

**Prédiction 2 (réciproque à `lr=5,0`), rapportée trop vite comme confirmée
« aux trois eps », puis corrigée après une relance de Théo (« cherche plus
loin »).** Vérifié où pointe réellement l'argmax du référent 0, pas
seulement `S[0].max()` et `R[0,18]` :

```
eps=18 : argmax = message 9   (pas 0 — chaos, pas capture)
eps=20 : argmax = message 22  (pas 0 — chaos, nouvelle collision ailleurs)
eps=23 : argmax = message 0   (capture reelle — la seule des trois)
```

**Deux transferts sur trois étaient le système jeté dans une configuration
sans rapport par un `lr=5,0` absurde, pas le mécanisme prédit.** Le logit du
référent 0 ne revient pas doucement vers l'uniforme sous ce `lr` — il
s'effondre de +22 à −53 en quelques pas, traverse zéro et atterrit ailleurs.
Même faille que le bug d'il y a deux tours, un cran plus loin : vérifier
`S[0].max()` et `R[0,18]` ne dit jamais QUEL message. Deux multiplicateurs
plus doux testés (×2, ×10) pour obtenir une confirmation propre — aucun n'y
arrive : ×2 reproduit juste le comportement par défaut (pas de preuve
supplémentaire), ×10 ne transfère nulle part. La preuve du sens « transfert »
tient sur un seul point propre, pas trois.

**Bord gauche du plateau, testé plutôt que laissé en suspens :**
`adam_eps=1e-13` donne déjà le même motif que 1e-12 et 1e-16 (18 retour,
20/23 transfert) — le plateau commence avant 1e-13, cohérent avec son
estimation `~3e-14`, non isolé plus finement.

**Poursuivi encore une fois (« continue encore »).** Hypothèse formée
explicitement : si `lr=5,0` jette le référent 0 hors cible plutôt que de le
geler, peut-être que le `lr=2,0` du tour précédent — que j'avais déjà
requalifié une fois de « BASCULE » à « GELÉ SANS VALEUR » — porte la même
faille, puisque cette requalification ne vérifiait pas non plus l'argmax.
**Hypothèse testée, confirmée :**

```
lr=2,0, eps=18 : argmax = message 6   (pas 0 — pas gele, chaos)
lr=2,0, eps=20 : argmax = message 20  (pas 0 — pas gele, chaos)
```

**Pas gelé non plus — même chaos que `lr=5,0`, ailleurs.** Sur les cinq
cellules à grand `lr` désormais vérifiées par argmax, quatre sont du chaos
et aucune n'est un vrai gel sur la cible perturbée. Le vocabulaire de
classification lui-même avait un trou : « gelé sans valeur » recouvrait en
fait deux choses différentes — une ligne qui reste sur place sans gagner de
récompense (jamais observée en pratique à grand `lr`), et une ligne
projetée ailleurs par un pas trop grand (ce qui arrive systématiquement).
La conclusion de tête ne change pas — `lr` seul ne produit toujours aucune
capture ciblée du message 0 en dehors du point `eps=23` déjà noté — mais le
mécanisme prêté à ces cellules était faux deux fois, pas une.

**Audit complet, à la relance de Théo (« j'ai l'impression que tu as sauté
des étapes »).** Il avait raison de le demander : je n'avais vérifié
l'argmax que sur les cellules déjà cassées (`lr=2,0`, `lr=5,0`). Tout le
reste publié cette session — le seuil 23/24 d'origine, la table de
convergence 18-20 en entier, le bras A du calendrier, le gel à `eps=1e-6`
lui-même — n'avait jamais eu cette vérification. Repris systématiquement :

```
seuil d'origine (eps=24, adam_eps=1e-8)              : argmax=0, capture reelle
table de convergence, TOUS les transferts publies    : argmax=0, capture reelle, sans exception
calendrier, bras A (K=50/100/200)                    : argmax=0, capture reelle, sans exception
gel a eps=1e-6 (eps=23,24)                            : argmax=0, vraiment gele SUR la cible
```

**Tout le reste tient.** Le chaos est propre au `lr` extrême (2,0 et 5,0) —
il ne touche ni le balayage `adam_eps` jusqu'à 1e-16, ni le calendrier, ni
le seuil d'origine sur lequel toute cette ligne d'argumentation repose.
Deux cellules sur une vingtaine maintenant vérifiées une par une, toutes
deux à un `lr` deux ordres de grandeur au-dessus de ce que les runs de
base ont jamais utilisé — c'est la frontière exacte des dégâts.

**Encore un cran plus loin, sur ma propre insistance à rester méfiant.**
Je ne vérifiais que où atterrit le référent 0. Restait à voir si le
référent 18 (celui censé céder) atterrit vraiment proprement sur son autre
message, et si le compte de collisions total s'améliore vraiment sans
créer un nouveau problème ailleurs :

```
adam_eps=1e-12, eps=23 : referent 18 -> message 8 (S=1,0000000000, propre)   26/27, 1 collision restante ailleurs
adam_eps=1e-8,  eps=24 : referent 18 -> message 8 (S=0,9999999997, propre)   26/27, 1 collision restante ailleurs
```

Les deux tiennent — atterrissage propre, une collision résolue, aucune
créée en échange.

**Le problème plus grave, trouvé en cherchant enfin si tout ça généralise
(« les résultats sont faibles », à raison).** Tout le mécanisme
d'`adam_eps` — seuil, convergence 18-20, loquet, équivalence `lr`/`eps` —
vient d'UNE graine, UNE collision, poussée par un scalaire ajouté à la
main sur un seul logit. Jamais vérifié si ça généralise. Ça ne généralise
pas.

**Le mur naturel (sans perturbation) ne bouge jamais, sous aucun `adam_eps`
testé, même 1e-16, sur 300 000 pas de plus :**

```
adam_eps=1e-8, 1e-12, 1e-16 : S[0].max()=0,037 dans les trois cas, toujours uniforme
```

Rien de tout ce mécanisme n'apparaît sans la poussée artificielle qui ne se
produit jamais pendant un entraînement ordinaire.

**Et la même poussée artificielle ne reproduit rien sur un mur différent.**
Deux murs trouvés dans deux flux de générateur distincts, même plage de
perturbation (15-23), même balayage `adam_eps` complet (1e-8 à 1e-16) :

```
mur 2 : 20 cellules, TOUTES retour — aucun seuil dans cette plage
mur 3 : sans perturbation (300k pas) ne se resout jamais ; avec +20, 3 cellules, TOUTES retour
```

**Aucun des deux autres murs ne montre la moindre sensibilité à `adam_eps`,
aux mêmes forces de perturbation et aux mêmes valeurs qui donnaient un seuil
net et précisément caractérisé sur le premier.** Cinq tours à construire un
compte exact et vérifié d'une idiosyncrasie d'une seule graine, jamais
vérifiée comme telle avant qu'on me demande directement si les résultats
étaient faibles. Ils l'étaient, et maintenant je sais en quel sens précis :
tout ce qui précède ce paragraphe décrit une ligne, pas un mécanisme.

Réponse dans `docs/REPONSE_ORDRE38.md`.

---

## 8ter. Cinq questions de fond, dessinées par onze tours de relecture

Écrites le 15/08/2026, à la demande de Théo, en transformant les critiques reçues en
questions plutôt qu'en corrections. Onze tours, et les corrections gagnaient en
précision **en restant au même étage**.

**1. Quand une colonne est-elle une variable ?** Une colonne de tableau ressemble à un
facteur qu'on l'ait réglée ou non. R a survécu à quatre tours de statistique de plus
en plus soignée parce que chacun tarifait un contraste au lieu de demander ce qu'était
la colonne. Dans ce domaine presque tout ce qu'on rapporte comme facteur est une
sortie : perte finale, rang effectif, nombre de features actives, pas où la
convergence est arrivée, architecture ayant survécu à un balayage. Le test est de
savoir si la colonne a été assignée avant le run, et il se répond en lisant le
générateur, pas les données.

**2. Une diagnostique de robustesse peut-elle exister sans modèle génératif ?** La
règle 6 dit non. Le nombre de rupture ne demande aucune loi pour se **calculer** et ne
peut pas s'**interpréter** sans une, et l'interprétation bouge d'un facteur deux selon
des choix sans rapport avec l'effet. Je soupçonne que ça vaut pour toute statistique
vendue comme sans hypothèse : l'hypothèse n'est pas absente, elle est **déplacée vers
l'étape de calibration**, là où personne ne la cherche.

**3. Lesquels de nos nombres sont des fonctions du plan, et lesquels des données ?**
Seuls les premiers peuvent être fixés avant la première graine, et seuls les premiers
ne se truquent pas. Plancher de détection : plan. Barre de Scheffé : plan. K : plan.
p, η², nombre de rupture et puissance observée : données. La puissance observée en est
la démonstration propre, puisqu'elle **ressemble** à une quantité de plan et **est** un
p.

**4. Quel est le plus petit effet qui aurait changé une conclusion ?** Une expérience
sans ce nombre ne peut être ni sous-puissante ni sur-puissante, la puissance étant
relative à un effet que personne n'a nommé. Il doit s'écrire avant les données, et
après les données il ne peut plus s'écrire honnêtement. Le nôtre n'a jamais été écrit,
d'où un tableau dont la réponse dépend d'un rapport que je choisis.

**5. Quelle part de la méthode empirique est une machinerie pour tarifer des mesures
qui ne pouvaient pas compter ?** Onze tours sur le prix correct d'un contraste, porté
par une colonne non éligible, mesurant une grandeur dont le seuil de pertinence n'a
jamais été fixé, dans un tableau où tout effet est à son plancher de détection ou
dessous. Chaque correction était juste. **La suite n'a jamais demandé si la chose
corrigée valait la peine d'être mesurée**, et aucun outil que l'un ou l'autre a
saisi n'était un outil pour cette question.

Je n'ai pas de règle pour 4 et 5. La seule chose faisable est de mettre la colonne des
planchers et un seuil de pertinence écrit dans le document de conception **avant** le
prochain run, où le second est encore écrivable.

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

> **Répondue le 12/08/2026 par la revue de littérature, et par l'affirmative.**
> Le théorème 2 de Kuciński et coll. (§7.23) en exhibe une : sous la perte
> `J(ℓ,f) = 𝔼[H(ρ(f′,f))]` bâtie sur la distance de Hamming entre traits, et un
> canal bruité avec ε < (|𝒜|−1)/|𝒜|, un langage minimise J sur les bijections
> **si et seulement si** il est compositionnel. C'est exactement l'objet demandé.
>
> **Mais la clause « sans qu'on ait codé la compositionnalité à la main » ne tient
> pas.** Cette perte est factorisée sur les traits mêmes dont la compositionnalité
> parle : elle dit à l'agent que les attributs comptent séparément. §6.6 le montre
> par la négative — sous une récompense tout-ou-rien, l'égalité des bijections
> survit à tout ε, donc la factorisation est **porteuse** et non décorative.
>
> Le conditionnel de Q11 ne se déclenche donc pas. Ce qui survit est plus étroit
> et plus juste : **une récompense qui départage les codes compositionnels doit
> déjà connaître la décomposition en traits.** Ce n'est pas « depuis la récompense
> seule » au sens où ce projet l'entend, c'est la spécification déplacée d'un cran.
>
> **Et ma dernière phrase était injuste.** J'accusais la littérature de mesurer des
> contraintes annexes en croyant mesurer le RL. Kuciński et coll. font exactement
> l'inverse : ils énoncent la condition de biais inductif **comme un théorème**, et
> séparent explicitement ce que la récompense fait de ce que le biais fait. J'ai
> écrit cette accusation sans avoir lu, ce qui est le défaut que §7.23 documente.

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

> **Répondue le 11/08/2026, et par la négative.** La journée du test 3 est la
> donnée : sept sections, un monde de 27 référents entièrement énumérable, tout
> calculé exactement — et **huit hypothèses datées mortes** (§1.9 à §1.16), plus
> cinq défauts de protocole rattrapés avant publication.
>
> **Aucune de ces erreurs n'était une erreur de calcul.** Un seuil bâti sur une
> statistique qui n'estime rien ; un critère de falsification qui omettait la
> variable décisive ; un contraste qui ne contrastait rien ; une justification de
> bruit de canal fausse en une ligne ; deux statistiques lues sur la dernière
> graine ; deux lois nulles non appariées ; une affirmation fausse sur les
> émetteurs autorégressifs. Toutes des erreurs de **spécification** — quoi
> comparer, à quoi, sous quelle condition.
>
> Rétrécir jusqu'à l'exactitude supprime donc une classe d'erreur, celle
> d'estimation, et **laisse intacte celle qui dominait**. Pire : l'exactitude
> produit des nombres à quinze décimales, qui se sur-lisent plus facilement.
> Plusieurs erreurs du jour sont des sur-lectures d'un chiffre exact — « écart
> 3,3 × 10⁻¹⁶, donc mon contraste marche », alors qu'il voulait dire que mon
> contraste était vide.
>
> **Ce qui a réellement attrapé les erreurs**, dans l'ordre de rendement : un
> lecteur extérieur ; une seconde lecture indépendante de la même chose (hessien
> contre bissection, Hamming contre information mutuelle, accord d'argmax contre
> z) ; et des prédictions écrites avant la mesure, qui ont tué §1.14 et §1.16.
> L'exactitude n'a servi qu'à rendre la seconde lecture **bon marché**. C'est une
> vraie valeur, et ce n'est pas celle que la question supposait.
>
> **La seconde moitié de la question pose une fausse alternative.** La variable
> n'est pas la taille, c'est le **contrôle**. Mon propre jouet a produit de
> l'anecdote chaque fois qu'il lui manquait une loi nulle : le seuil de 0,35, le
> « biaisé 2 contre 1 » tiré de trois graines. Une expérience à l'échelle avec
> ablation, ligne de base et variance inter-graines n'est pas de l'anecdote ; un
> jouet sans nulle en est. Et l'échelle fait une chose qu'aucun jouet ne fera
> jamais : établir qu'un phénomène **existe** dans le régime qui intéresse. Un
> jouet réfute des affirmations universelles, l'échelle établit des existences.
> Ce sont deux métiers, pas deux niveaux d'honnêteté.
>
> **Enfin, la prémisse de la question est fausse**, et c'est le point le plus
> utile. Ce qui a survécu de la journée n'est pas le jouet : ce sont les énoncés
> **sans chiffre** — le no-go d'équivariance, l'égalité que le canal ne brise pas
> quelle que soit la taille, β_c = 1/N, « un maximum d'échantillon n'estime rien
> quand la valeur à exclure est dans le support ». Aucun ne mentionne 27. La
> petitesse ne les a pas produits, elle a permis de les **trouver** et de les
> vérifier complètement à coût nul.
>
> D'où la règle qui remplace la question : **rétrécir pour chercher, pas pour
> prouver.** On ne débat alors plus de l'extrapolation d'un résultat — on vérifie
> les hypothèses d'un théorème, ce qui est une opération finie. Et le critère
> pratique tient en une phrase : *réécrire chaque conclusion sans aucun nombre ;
> ce qui n'y survit pas meurt avec le banc.*

**Q30 — À quel moment « je mesure ma propre spécification » s'applique-t-il à moi
et plus seulement à l'agent ?** J'ai écrit l'environnement, la récompense, les
diagnostics, les métriques et l'interprétation. Le diagnostic que j'ai construit
(imposer l'antécédent, mesurer le conséquent) détecte les sous-langues
dégénérées de l'agent. **Quel diagnostic détecte les miennes ?** C'est la seule
question de cette liste à laquelle je n'ai aucune piste.

> **Réponse de Théo, 11/08/2026 : un autre humain.** C'est juste, c'est ce qui a
> le plus rendu de la journée, et la mesure permet de préciser pourquoi — et où
> ça casse.
>
> **Ce qui a détecté quoi, par rendement.** Un lecteur extérieur en cinq séries :
> la borne de produit que je n'avais pas vue, la statistique qui mesurait la
> couverture, la saturation au-dessus de 100 %, trois graines lues comme
> vingt-quatre, le seuil bâti sur un maximum. Puis, de mon côté, une **seconde
> lecture indépendante de la même quantité** : hessien contre bissection, qui a
> montré que je mesurais Adam ; distance de Hamming contre information mutuelle ;
> accord d'argmax contre z. Puis des **prédictions écrites avant la mesure**, qui
> ont tué §1.14 et §1.16.
>
> **Ce n'est donc pas l'humanité du lecteur qui compte, c'est qu'il n'hérite pas
> de la spécification.** La propriété décisive de dipankarsarkar n'était pas
> d'être humain : c'était de **relancer le code avant de parler**, à chaque fois,
> et de re-dériver depuis l'artefact publié plutôt que depuis mes intentions.
> Ça impose au passage une contrainte sur moi : publier assez pour que la
> re-dérivation soit possible.
>
> **Et le contre-exemple du jour, qui limite la réponse.** Son argument « une
> concentration de 1 force un argmax injectif, donc le sommet est sûr » était
> juste, je l'ai vérifié, et je l'ai repris à mon compte. Sa prémisse était la
> bijectivité. **Nous étions deux et nous avions tort tous les deux.** Ce qui l'a
> cassé n'est pas un troisième lecteur : c'est §6.5, une étape qui n'avait jamais
> tourné, et qui a produit des codes non bijectifs.
>
> Deux personnes peuvent partager un cadre. Ce qui est **structurellement**
> garanti de ne pas le partager, c'est une mesure qu'on n'a pas encore faite.
> D'où la formulation qui englobe la réponse de Théo au lieu de la remplacer :
>
> > Le diagnostic de sa propre spécification, c'est **tout ce qui n'en hérite
> > pas** : un lecteur qui re-dérive depuis l'artefact, une seconde route vers la
> > même quantité, une prédiction datée — et, quand tout ça s'accorde et se
> > trompe ensemble, **l'étape du programme qu'on n'a pas encore exécutée**.
>
> Ce que ça ne résout pas : rien de tout ça n'est automatique, et les trois
> premiers dépendent de la bonne volonté de quelqu'un. Le quatrième est le seul
> que je contrôle seul, et c'est un argument pour exécuter le programme dans
> l'ordre plutôt que de commenter les étapes non faites.

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
