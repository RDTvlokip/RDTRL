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

### 7.55 Trente-huitième critique : la « non-généralisation » était fausse — je testais la mauvaise fenêtre

04/09/2026. Il démonte la démolition du tour précédent avec précision : mon
mur 2 ne testait `eps_perturb` que de 15 à 23, jamais 24 — le seul point où
le mur 1 transfère au réglage par défaut. Mon mur 3 ne testait que
`eps_perturb=20`, un point où le mur 1 lui-même n'a jamais transféré à
aucun `adam_eps`. **Un négatif dont la fenêtre exclut le seul point positif
connu n'est pas un négatif.**

**Vérifié, et il avait raison sur toute la ligne.**

- Mur 1 sur la grille exacte du mur 2 (`eps_perturb` 15-23) : revient
  presque partout aussi, sauf sous `adam_eps` ≤ 1e-10 — même motif que les
  murs 2 et 3.
- `eps_perturb=24-26` sur les murs 2 et 3, jamais testé avant : **les deux
  transfèrent**, à `eps=26` au lieu de 24. Trois murs sur trois ont
  maintenant un vrai seuil de capture — la variation seuil-à-seuil (23-24
  contre 25-26) est une variation de graine ordinaire, pas une absence de
  mécanisme.

**Je retire la conclusion d'idiosyncrasie du tour précédent.**

**Sa seconde critique, tout aussi grave, tient aussi.** Un seul optimiseur
couvre émetteur ET récepteur (`parametres(e,r)` concatène les deux) — donc
le récepteur ET la ligne du référent 18 voient le même `adam_eps`. Sa table
de ratios recalculée de mon côté, exacte au chiffre près : 1,00000 au
minimum du récepteur, **35,25× à `sqrt(v)=5,29e-7`** (la ligne du référent
18). Le bras B (`eps=1e-6, lr=5,0`) n'est fidèle QUE sur le bloc récepteur
— déjà garanti par le recensement — et jusqu'à 100× plus chaud ailleurs.
**Ma « confirmation réciproque » d'il y a deux tours pourrait donc venir de
l'émetteur suramplifié, pas de l'équivalence du récepteur.** Rouvert,
non tranché : recensement côté émetteur pas encore fait.

**Sur l'idiosyncrasie :** `0,037037 = 1/27` exactement — c'est l'état
générique d'indécision maximale, pas un signe distinctif. Retiré.

Réponse dans `docs/REPONSE_ORDRE39.md`.

**Question posée à moi-même et tranchée : le transfert réciproque venait-il
du récepteur ou de l'émetteur ?** Deux optimiseurs Adam séparés au lieu
d'un seul, pour donner les réglages réciproques (`eps=1e-6, lr=5,0`) à un
seul agent à la fois, l'autre restant aux réglages d'origine.

```
recepteur seul aux reglages reciproques : eps=18,20,23 -> retour, TOUJOURS
emetteur seul aux reglages reciproques  : eps=18,20 chaos, eps=23 -> TRANSFERT
```

**Tranché : c'est l'émetteur, pas le récepteur.** Le récepteur seul ne
produit jamais le transfert, à aucune des trois perturbations. L'émetteur
seul le produit, exactement au même seuil que le test combiné. La
« confirmation réciproque » de deux tours plus tôt créditait le mauvais
agent — c'est le référent 0 qui reçoit une poussée surdimensionnée sous
`lr=5,0`, pas une propriété du récepteur. L'équivalence `lr`/`eps` tient
toujours sur le recensement du récepteur et sur le sens gel ; le sens
transfert n'a jamais eu de rapport avec le récepteur du tout.

**Dernière question posée à moi-même : le récepteur a-t-il seulement
besoin d'APPRENDRE, ou n'est-il qu'un lecteur passif ?** Récepteur
totalement gelé (jamais mis à jour), émetteur seul aux réglages
d'origine :

```
eps=18 a 24 : retour
eps=26      : S[0].max()=1,0000, argmax=0 — ressemblait a un transfert
```

**Vérifié `R[0,0]` avant de le croire — `R[0,0] = 4,4e-10`.** Pas un
transfert : le même engagement gelé et gaspillé que le test du référent 1,
six tours plus tôt. Le référent 0 s'engage pleinement sur le message 0 par
pur emballement auto-renforcé sous Adam, mais le récepteur gelé continue
d'attribuer tout le crédit au référent 18 comme toujours — récompense
nulle. Failli me tromper une troisième fois de la même façon, rattrapé
avant publication. Réponse à ma propre question, à l'inverse de ce que le
premier passage suggérait : l'apprentissage du récepteur n'est pas
accessoire à un vrai transfert, il en est la condition — chaque capture
réelle trouvée dans toute cette enquête impliquait le récepteur lui-même
déplaçant son crédit.

### 7.56 Trente-neuvième critique : le « trois sur trois » retombe à un sur un — et sa prédiction sur l'emprise du titulaire tient

04/09/2026. Il pose la question qui aurait dû être la première : les murs 2
et 3 étaient-ils vraiment scorés sur `R[message, perdant]`, ou seulement sur
`S.max()` et l'argmax — le même bug corrigé deux fois ce tour-ci ?

**Vérifié : scorés sur S/argmax. Et faux.**

```
mur 2, R[10,4]  a eps=25,26,30,35,40,50 : fige a ~6,7e-12, ne franchit jamais 0,5
mur 3, R[10,20] a eps=25,26,30,35,40,50 : fige a ~1,5e-11, meme motif
```

**Chaque « capture » rapportée pour les murs 2 et 3 était le même
engagement gelé et gaspillé que le test du récepteur figé — jamais vérifié
ici.** Le « trois sur trois » retombe à un sur un : seul le mur 1 a une
vraie capture confirmée dans tout cet échange.

**Sa prédiction sur l'emprise du titulaire — confirmée, et elle explique
exactement ce qui vient d'arriver.** Avant toute perturbation :

```
mur 1, message 0  : titulaire R=0,9999999961 (8,41 decades)
mur 2, message 10 : titulaire R=0,9999999995 (9,31 decades)
mur 3, message 10 : titulaire R=0,9999999995 (9,33 decades)
```

Les titulaires du message 10 tiennent ~1 décade de plus que celui du
message 0 — exactement ce qui prédit que les murs 2 et 3 aient besoin de
bien plus que la poussée qui suffit au mur 1. Testé le même levier qui
avait débloqué le mur 1 (`adam_eps` plus petit à eps=26) : **ça empire**,
`R[10,4]` et `R[10,20]` décroissent de façon monotone (1,34e-13 → 3,26e-15
→ 1,77e-16). Sa prédiction est la première de tout cet échange à
**anticiper** un résultat avant le lancement plutôt que d'en expliquer un
après coup.

**Les deux cellules qui portent vraiment la conclusion du gel, confirmées
sur la vraie récompense :** eps=24 et eps=26, non gelé = vraie capture
(`R[0,0]=1,0000`), gelé = engagement gaspillé. Le seuil (23,24] confirmé
aussi par eps=25 : vraie capture.

Réponse dans `docs/REPONSE_ORDRE40.md`.

### 7.57 Quarantième critique : la calibration propre prédit 26-27, l'indépendant dit zéro — le grip compte mais ne suffit pas

05/09/2026. Il propose l'expérience qui règle la pente : au lieu de pousser
les murs 2/3 vers un seuil peut-être inexistant, faire varier le grip sur
UN SEUL défenseur (référent 18, message 0) en le lisant à différents
checkpoints d'entraînement, avant convergence complète.

**Fait. Quatre points, un seul défenseur, aucune confusion de mur :**

```
checkpoint  grip   seuil
 10 000     5,70   (14,17]
 20 000     7,43   (20,22]
 30 000     8,18   (22,24]
 40 000     8,41   (23,24]  (deja etabli)
```

Les deux formes (additive, multiplicative) s'ajustent aussi bien l'une que
l'autre sur cette plage, et **convergent vers la même extrapolation** pour
le grip des murs 2/3 (9,31-9,33) : **26,4 et 27,5 respectivement.**

**Et cette prédiction, vérifiée contre les données déjà en main, échoue.**
Les murs 2 et 3 ne capturent JAMAIS jusqu'à eps=50 (§7.56). Le grip compte
clairement — l'ajustement à un seul défenseur est propre, les résidus
petits — mais il ne suffit pas seul en changeant de défenseur. Quelque
chose distingue le référent 18 des référents 3/14 que le grip seul ne
capture pas. La masse du dauphin, déjà écartée par lui, ne revient pas
(les murs 2/3 diffèrent de 2,3× dessus et se comportent identiquement).

**Sa dernière question, tranchée dans l'autre sens que ce qu'il
soupçonnait :** grip du récepteur GELÉ du mur 1 à eps=26, précision
complète : **8,4107 — exactement la valeur non gelée de base, pas
augmentée.** Geler fixe le grip à sa valeur de checkpoint, il ne l'élève
pas. Le résultat de gel et la loi de grip sont deux découvertes
indépendantes, pas la même mesurée par deux leviers.

Réponse dans `docs/REPONSE_ORDRE41.md`.

### 7.58 Quarante-et-unième critique : le grip ne se superpose pas entre défenseurs — la loi meurt proprement

06/09/2026. Il refait le calcul de faisabilité sur les vrais intervalles
(pas les milieux) et montre qu'aucune courbe cohérente avec mes quatre
points ne prédit un seuil au-delà de 50 pour les murs 2/3 — mon extrapolation
tenait. Mais il montre aussi que grip et log(pas d'entraînement) sont
indiscernables à ma résolution (RSS 0,10 contre 1,79, sous mes propres
demi-largeurs de crénaux) : la calibration à un seul défenseur ne prouve
pas encore que l'axe est le grip plutôt que l'horloge. Propose le test qui
tranche : la même échelle sur un second défenseur, en comparant le
recouvrement grip-contre-seuil et horloge-contre-seuil.

**Deux points de plus sur le référent 18 d'abord (80k, 160k pas) :**
grip=8,7481 puis 8,9763. Réajusté avec les 9 points : les deux formes
convergent maintenant vers une **asymptote sous 9,31** (9,204 et 8,839) —
le référent 18 ne peut jamais atteindre le grip des murs 2/3, quelle que
soit la durée.

**L'échelle complète sur le référent 3 (mur 2), même protocole :**

```
checkpoint  grip     seuil
 10 000     6,1115   > 26 (ne capture jamais)
 20 000     8,0450   > 26
 30 000     8,9951   > 26
 40 000     9,3088   > 26
```

**Les courbes ne se superposent pas, et pas d'un peu.** À grip=6,11 —
proche du plus bas point du référent 18 (5,70, seuil 14-17) — le référent 3
ne capture à AUCUN eps jusqu'à 26, dix unités au-delà d'un point de grip
comparable chez le référent 18. **Le grip décrivait la trajectoire propre
du référent 18, pas un mécanisme portable entre défenseurs.**

**Hypothèse testée immédiatement plutôt que laissée en suggestion : l'écart
de LOGIT brut (pas la probabilité) entre titulaire et dauphin.** Fausse
aussi, et plus nettement que le grip : les écarts de logit sont proches
entre les deux défenseurs aux mêmes checkpoints (13,7 contre 14,6 à 10k ;
21,0 contre 22,9 à 40k) alors que les seuils divergent totalement. Ni la
probabilité ni le logit brut du titulaire seul n'expliquent l'écart —
peut-être le logit de départ du challenger, ou une interaction entre les
deux lignes, non testé.

Réponse dans `docs/REPONSE_ORDRE42.md`.

### 7.59 Quarante-deuxième critique : l'écart de logit n'était jamais un second candidat — c'est le grip fois 2,45

07/09/2026. Il montre que gap/grip reste dans 2,39-2,49 sur les deux
défenseurs et les quatre checkpoints — le logit n'était pas une variable
différente, c'est le grip multiplié par une constante quasi fixe le long
d'une trajectoire d'entraînement. Rejeter l'un rejette l'autre par
construction. **Vérifié, confirmé.**

Il refait aussi le calcul de faisabilité sans milieu : `thr = k·grip` seul
(sans terme additif) satisfait les quatre crénaux du référent 18,
`k ∈ (2,7348 ; 2,8537]`. Mais le référent 3 n'a encore aucun `k` ajusté —
ses quatre lignes sont censurées au même plafond (26), ce qui équivaut à
une seule observation répétée, pas quatre. Demande de pousser le référent
3 à 10k (son seuil le plus bas) au-delà de 26 jusqu'à capture réelle.

**Fait. Le récepteur s'y gèle aussi (même valeur de 30 à 60 sous
`adam_eps` par défaut) — réduit `adam_eps`, comme pour le mur 1 :**

```
eps=30, adam_eps=1e-10/1e-12 : R[10,4]=0,500 — exactement au seuil
```

**k_3 = 30 / 6,1115 = 4,907.** Contre k_18 ≈ 2,79. **Pas proche, et pas
explicable par la variation de gap/grip** — même en comptant les 6 % de
spread trouvés (voir ci-dessous), c'est un ordre de grandeur trop petit
pour expliquer un écart de k de 2,79 à 4,91.

**Sa dernière question — un défenseur sort-il de 2,39-2,49 ?** Vérifié sur
deux de plus : référent 14 (mur 3) = 2,5367, référent 1 (gagnant solo du
tour 33) = 2,5119. **Les deux sortent de la fourchette** — grip et écart de
logit ne sont pas parfaitement la même variable, il y a un vrai spread
(2,39-2,54, ~6 %). Mais ce spread reste bien trop petit pour porter la
différence k_18/k_3. Le troisième facteur doit venir d'ailleurs — du côté
du challenger, comme il le pointait déjà avant ce tour.

Réponse dans `docs/REPONSE_ORDRE43.md`.

### 7.60 Quarante-troisième critique : le plateau à 0,5 n'était pas une capture — c'était une vraie égalité, k_3 est invalide

07/09/2026. Deux vérifications demandées, plus une trouvée en creusant
plus loin sans qu'on me le redemande (consigne du `CLAUDE.md`).

**Le référent 18 ne gèle jamais, nulle part.** Poussé à checkpoint 40k,
`adam_eps` par défaut, jusqu'à eps=100 :

```
eps = 24, 26, 30, 40, 60, 100 : S[0].max()=1,0000, R[0,0]=1,0000 — capture
propre à chaque fois, aucun plateau gelé.
```

**Le levier de référent 3 (`adam_eps` réduit) ne bouge pas k_18.** Testé
sur le même levier que référent 3 (`adam_eps=1e-10`) :

```
checkpoint 40k : eps=18,20 retour, eps=23 capture — même créneau qu'au
  adam_eps par défaut.
checkpoint 10k : eps=10,13,14 retour, eps=15 capture — k=15/5,697=2,633,
  cohérent avec la fourchette 2,79.
```

**Le levier ne fait que dégeler des cellules gelées** ; il ne redéfinit
pas globalement le seuil. Référent 18 n'a jamais été gelé, donc son seuil
ne bouge pas quand on baisse `adam_eps`.

**Et voilà où ça devient grave : le "k_3 = 4,907" du tour précédent ne
mesurait pas une capture du tout.** En imprimant `R[10,4]` et `R[10,3]` en
pleine précision plutôt qu'avec le seuil `>0,5` :

```
eps=24 : R[10,4]=0,500000583504767   R[10,3]=0,499999416424236
eps=26 : R[10,4]=0,499999999984430   R[10,3]=0,499999999944572
eps=28 : R[10,4]=0,499999055744959   R[10,3]=0,500000944184045
eps=29 : R[10,4]=0,500000003840574   R[10,3]=0,499999996088429

et les deux émetteurs : S[4].max()≈0,9999999996  S[3,10]≈0,9999999997
```

**Les référents 3 et 4 sont TOUS LES DEUX pleinement engagés sur le
message 10 en même temps**, et le récepteur partage le crédit presque
exactement moitié-moitié. Ce n'est pas l'engagement gaspillé du mur (un
seul côté gèle, l'autre garde tout) — c'est la vraie égalité déjà
caractérisée bien plus tôt (§7.47-48, paire 23/25), atteinte cette fois
par une route complètement différente. Mon critère `R>0,5` classait ça
« capture » parce que la valeur tombait de quelques millionièmes au-dessus
de 0,5 à deux `eps` sur quatre, et en dessous aux deux autres — ce qui
explique aussi pourquoi ça semblait non monotone : ça ne l'a jamais été
au sens où je vérifiais, ça oscille autour d'une égalité exacte, sans
rapport avec un franchissement de seuil.

**Conséquence : il n'y a pas de k_3 à comparer à k_18.** Ce que 26/09
appelait « k_3 = 4,907 » mesurait le début d'une égalité, pas une capture
propre — un événement d'une autre nature, que mon classeur notait pareil.
Aucun re-calibrage ne le rend comparable à 2,79.

**Ce que ça répond à la vraie question sous-jacente du fil.** Le
« troisième facteur » cherché depuis plusieurs tours n'est peut-être pas
un scalaire de plus du côté du titulaire ou du challenger — c'est une
branche qualitative : la collision se résout-elle en vainqueur/perdant,
ou en égalité ? Les collisions du référent 18, testées partout, ne se
résolvent jamais qu'en vainqueur/perdant. Celle du référent 3, poussée
de cette façon, se résout en égalité. Reste ouvert : qu'est-ce qui décide
de la branche pour une paire donnée — pas encore de réponse sur un seul
exemple.

Réponse dans `docs/REPONSE_ORDRE44.md`.

### 7.60bis Quarante-quatrième critique : le résidu du récepteur reste figé sous une capture propre aussi — la « branche qualitative » ne survit pas

08/09/2026. Il repère que dans mon tableau d'égalité, la masse qui fuit
vers tout ce qui n'est ni le référent 3 ni le référent 4 est immobile à
2e-15 près (7,0997e-11 partout) alors que le partage entre les deux
bouge de 1,9e-6 — cinq ordres de grandeur d'écart sur les quatre mêmes
runs. **eps ne perturbe pas la ligne, il fait tourner la masse À
L'INTÉRIEUR de la paire et conserve tout le reste exactement.** Il
propose le test décisif : imprimer le résidu du référent 18 en pleine
précision, sous une capture propre. Si ça bouge, le résidu figé est une
signature d'égalité. Si c'est figé aussi, sa théorie du scalaire meurt.

**Perdu la reconstruction avant de pouvoir répondre — épisode à
raconter tel quel.** Les étiquettes « référent 18 », « message 0 »
n'étaient jamais les indices bruts du tenseur — de simples labels de
présentation d'un run jamais sauvegardé en fichier. Deux mauvaises
graines testées (`default_rng(5)`, puis `default_rng(31415)`, la
graine du seul script sauvegardé sur disque) donnent de vrais murs qui
capturent réellement — mais aucun ne correspond aux chiffres déjà
publiés (`S[18,0]=0,499479`), et l'un des deux perdants reste gelé à
zéro jusqu'à eps=100, très loin du seuil 23-24 annoncé. **Retrouvée en
grepant le transcript JSONL de la session elle-même** (idée de Théo)
sur `e.p[0][18` : la commande exacte y était mot pour mot, graine
maîtresse `default_rng(999)`, cinq paires sautées. Vérifiée contre le
carnet : `S[18,0]=0,499479`, `S[18,8]=0,500521`, référent 0 au plancher
d'entropie — identique au dixième de pourcent près. Sauvegardée
définitivement dans `replay_idx5.py`.

**Le résultat, sur la vraie paire :**

```
eps= 24 : R[0,0]=0,999999715203  residu=4,648726e-12
eps= 26 : R[0,0]=0,999999715818  residu=4,636735e-12
eps= 30 : R[0,0]=0,999999715881  residu=4,640843e-12
eps= 40 à 100 : identique, residu=4,635403e-12
```

**Figé aussi.** Spread d'environ 0,3 % relatif, contre une capture qui
elle-même ne bouge presque plus sur la même plage. Aucune égalité ici —
un vainqueur net, un perdant net — et le résidu se fige quand même.

**Sa deuxième branche l'emporte : le résidu est structurel, pas une
signature d'égalité.** Ma lecture « branche qualitative » du tour
précédent (§7.60) ne tient pas — je lisais le résidu figé comme un
marqueur distinctif de l'égalité, et il ne l'est manifestement pas
puisqu'il se fige pareil sans égalité en vue.

**Sa comparaison fuite-émetteur/résidu ne se transpose pas telle
quelle.** Sous cette capture, `1 - S[0].max() = 0` (saturation complète,
rien à mesurer) alors que `1 - r[0,0] = 2,84e-7`, quatre ordres
au-dessus du résidu lui-même. Ce qui fige le résidu ici n'est donc pas
simplement « la fuite des émetteurs déguisée en quantité du récepteur »
comme sur sa paire à égalité — sauf si le mécanisme diffère entre un
seul vainqueur plein et deux co-titulaires à égalité, plausible mais
non vérifié.

Réponse dans `docs/REPONSE_ORDRE45.md`.

### 7.60ter Quarante-cinquième critique : ma comparaison fuite/résidu testait le même nombre contre lui-même

08/09/2026. Il repère que `1-R[0,0]=2,841160e-07` et `1-r[0,0]=2,841160e-07`
concordent à sept chiffres significatifs dans mon tableau précédent —
parce qu'à eps=100 le référent 0 est déjà saturé, donc `R` et `r` ne
peuvent plus se distinguer. Ma phrase « quatre ordres au-dessus du
résidu » ne comparait rien d'indépendant. Il prédit ce que devraient
donner `1-r[0,0]` à eps=24 et 26 si `r` est un objet réel : 2,847969e-07
et 2,841825e-07 respectivement, contre autre chose sinon.

**Vérifié, ses valeurs prédites tombent exactement :**

```
eps=24 : 1-R=2,847969e-07  1-r=2,841324e-07  1-s=6,644512e-10
eps=26 : 1-R=2,841825e-07  1-r=2,840549e-07  1-s=1,275220e-10
eps=30 : 1-R=2,841186e-07  1-r=2,841161e-07  1-s=2,431388e-12
eps≥40 : identiques, 1-s=0 (saturation complete)
```

**`r` est un objet distinct, et la mécanique est triviale : `R=s·r`
donc `1-R≈(1-r)+(1-s)` au premier ordre.** Vérifié : `2,841324e-07 +
6,644512e-10 = 2,847969e-07`, exact à eps=24. `R` et `r` ne coïncident
qu'une fois que la fuite du référent 0 (`1-s`) sous-passe la précision
qui les sépare — pas parce que ce sont le même tenseur, mais parce que
l'algèbre force leur accord dès que le sous-cause s'annule. Sur les
lignes où `s` a encore une fuite mesurable (eps=24, 26), la comparaison
« résidu bien plus petit que la fuite du récepteur » tient toujours,
juste sur la bonne ligne cette fois.

**Reste ouvert : une égalité et une capture complète sont les deux bouts
où ce test est le moins parlant** — une égalité n'a pas d'émetteur saturé
qui force `R` et `r` ensemble, une capture complète les y force
mécaniquement au bout d'un moment. Un transfert partiel, où l'émetteur a
encore une fuite mesurable, est le régime qui trancherait vraiment. Pas
encore de paire de ce type sous la main.

Réponse dans `docs/REPONSE_ORDRE46.md`.

### 7.60quater Cherché plus loin sans qu'on me le redemande : le transitoire du transfert, pas seulement sa convergence

08/09/2026, sur ma propre insistance (consigne `CLAUDE.md`). Le régime à
transfert partiel que je disais « pas sous la main » à la fin du tour
précédent était accessible directement : il suffisait d'échantillonner
le nombre de pas post-perturbation au lieu de toujours attendre la
convergence à 20 000.

**Balayage à eps=24, pas de 6000 à 10000 :**

```
pas=6000-7500 : R[0,0]=0        residu≈0,499999...   1-r=1,0        1-s≈2-3e-9
pas=8000      : R[0,0]=0,682857 residu=1,443847e-03  1-r=0,317      1-s=3,3e-9
pas=8500      : R[0,0]=0,999922 residu=5,157076e-10  1-r=7,80e-05
pas=9000-10000: convergence progressive vers residu~1,5e-10
```

**Le résidu n'est PAS figé pendant le transfert réel — il bouge sur
neuf ordres de grandeur**, de 0,5 jusqu'à 1,5e-10. Le résidu figé à
4,6e-12 rapporté au tour précédent n'était que la queue d'un effondrement
déjà terminé, pas un signal sur le mécanisme.

**Et le plus intéressant : de 6000 à 7500 pas, l'émetteur (référent 0)
est déjà saturé (`1-s`≈2e-9) mais le récepteur ne l'a pas encore
rattrapé — la masse du message se scinde presque exactement 50/50 entre
référent 0 et référent 18, quatre points de suite.** Signature identique
à l'égalité référents 3/4 du tour 7.60. Sauf qu'ici ce n'est qu'une
étape transitoire : à 8000 pas c'est déjà rompu (68 % pour le référent
0), à 8500 c'est fini.

**Ça reformule toute la question du résidu.** Un instantané d'égalité et
un instantané de mur capturé peuvent être LA MÊME trajectoire à deux
instants différents, pas forcément deux types de point fixe distincts.
Certaines scissions 50/50 sont des étapes transitoires (comme ici),
d'autres sont de vrais points fixes protégés (référents 3/4, qui
oscillent encore autour de 0,5 après de nombreux pas de plus, sans
signe de résolution). Impossible de distinguer les deux sur un seul
instantané — il a fallu échantillonner across pas pour voir la
différence ici.

Réponse dans `docs/REPONSE_ORDRE47.md`.

### 7.60quinquies Quarante-septième critique : r est R sans exception, et le discriminateur à un seul instantané tombe mort sur la vraie égalité

09/09/2026. Il reconstruit `r[0,0]` à partir de ma propre colonne `1-r`
et montre que ça reproduit `R[0,0]` chiffre pour chiffre sur les neuf
lignes, y compris les quatre où `R[0,0]=0` exactement — aussi loin de la
saturation que cette trajectoire aille jamais. **`r` est `R` sur toute
la ligne, pas seulement après saturation.** Ma comparaison « quatre
ordres au-dessus » n'a jamais reposé sur rien, même à eps=24.

**Puis il pointe le vrai test : que fait `1-s` sur les référents 3 et
4 (la vraie égalité) ?** Si proche de 1e-9 des deux côtés, le test
d'asymétrie est mort et il faut l'axe temporel. Si proche de 0,5, une
seule colonne suffit à distinguer étape transitoire et point fixe sur
un instantané unique, pour toujours.

**Reconstruit la config référents 3/4 (graine 77777, k=3, checkpoint
10k, référent 4 poussé +30 sur le message 10) :**

```
adam_eps=1e-08 : R[10,4]=0,000000  1-s[4,10]=2,46e-12  1-s[3,10]=2,46e-10
adam_eps=1e-10 : R[10,4]=0,500009  1-s[4,10]=2,99e-12  1-s[3,10]=3,34e-10
adam_eps=1e-12 : R[10,4]=0,500001  1-s[4,10]=3,61e-10  1-s[3,10]=3,61e-10
adam_eps=1e-14 : R[10,4]=0,500000  1-s[4,10]=3,61e-10  1-s[3,10]=3,61e-10
```

**Les deux émetteurs sont déjà pleinement saturés (`1-s` entre 1e-10 et
1e-12), loin de 0,5.** Sa première branche, pas la seconde : le test
d'asymétrie est mort ici. Ce qui établit vraiment référents 3/4 comme
un point fixe (et non une étape transitoire anormalement lente) n'est
pas dans cet instantané — ça demande l'axe temporel, exactement comme
sa propre mise en garde le prévoyait.

**Repris avant de publier, plutôt que de me contenter d'invoquer une
vérification d'un tour antérieur.** Théo m'a repris entre-temps : « tu
n'as pas cherché loin et je t'ai vu suivre le standard » puis « ne suis
jamais le standard ». Deux raccourcis corrigés dans la foulée (ajout
`CLAUDE.md` : ne jamais suivre le standard, et le standard porte des
biais qu'il faut interroger explicitement) :

**La dérive longue, refaite sur la vraie paire (pas supposée
transposable depuis un tour antérieur), 400 000 pas cumulés :**

```
pas=40000 à 400000 : R[10,4] = 0,500009 ... 0,499798 ... 0,500080
```

Épinglé à 0,5 sur dix points, aucune dérive systématique — vérifié
cette fois sur la reconstruction actuelle elle-même (graine 77777,
k=3).

**Et le pic non-monotone de `1-s[0,0]` (transitoire idx5, pas=8000),
que j'avais failli classer « noté, pas expliqué » : trois hypothèses
formées et testées plutôt que listées.**

- H1 (artefact d'Adam) : rejoué sous SGD pur — rien ne bouge du tout à
  ce lr sur cette fenêtre (logit figé exactement). Non concluant, mais
  révèle que « montée exacte » dans tout l'échange = Adam, jamais du
  SGD littéral (`monter()` crée toujours un `torch.optim.Adam`).
- H2 (redistribution d'entropie) : **confirmée**. La somme des 26
  logits perdants du référent 0 suit le même pic exactement (53,089 →
  53,337 à pas=8000 → 53,053).
- H3 (rétroaction du récepteur) : **confirmée**. `r[0,0]` bascule de 0
  à 0,683 exactement à pas=8000, le même instant que le pic.

**Ce n'était pas du bruit — c'est la signature visible du système
couplé émetteur/récepteur au moment exact où le décodage du récepteur
bascule**, redistribuant brièvement la masse des logits perdants avant
que l'émetteur ne reprenne sa montée.

**Trois hypothèses de plus, formées et testées avant de clore.**

- H4 (Adam est-il nécessaire au transfert, pas seulement au pic ?) :
  rejoué la fenêtre idx5 sous SGD à lr=5,0 (100× celui d'Adam), jusqu'à
  20 000 pas — `r[0,0]` reste exactement à 0 tout du long, `1-s[0,0]`
  bouge à peine. **Adam n'est pas un raccourci de vitesse ici, c'est
  structurellement nécessaire au transfert observé dans cette fenêtre.**
- H6 (un rival précis ou une redistribution uniforme ?) : les cinq
  meilleurs rivaux (messages 7, 5, 15, 18, 23) montent tous ensemble
  d'environ +0,01 au pic (pas=8000) puis retombent ensemble.
  **Redistribution uniforme, pas un concurrent qui gagne du terrain.**
- H5 (pousser plus fort force-t-il une vraie capture sur référents
  3/4 ?) : poussé à eps=60, 100, 200, 400 (contre 30 pour l'égalité),
  40 000 pas chacun, `adam_eps=1e-10`. **Résultat identique bit pour
  bit sur les quatre eps, à chaque point de contrôle** — l'émetteur
  sature instantanément dès eps=60 (`1-s[4,10]=0`), et au-delà pousser
  plus fort n'a plus rien à mordre. Le récepteur reste épinglé à 0,5
  exactement. **Non : cette égalité résiste à la perturbation côté
  émetteur sur plus d'un facteur six**, une affirmation plus forte et
  plus précise que « stable sur 400 000 pas ».

Réponse dans `docs/REPONSE_ORDRE48.md`.

---

### 7.61 Quarante-huitième critique : le 0,5 est un posterior bayésien fragile, pas un point fixe — mais pas au ratio littéral qu'il prédisait

14/09/2026. Il relit le tableau à eps ajusté (adam_eps=1e-10/1e-12/1e-14)
publié au tour précédent et pointe un fait que j'avais imprimé sans le
lire : `s[4,10]` et `s[3,10]` sont tous deux à 1 moins environ 1e-10 —
**ce n'est pas deux émetteurs proches d'une égalité, c'est un seul
message porté par deux référents.** Le posterior bayésien du récepteur
sur un tel message collisionné est mécaniquement épinglé à
`s4/(s4+s3) = 0,500000000083` — pas un point fixe que le système aurait
trouvé, la réponse à une question qui n'a plus d'information dedans.

**Vérifié chiffre par chiffre : ses trois lignes reproduisent exactement
depuis mon propre script.** Un bémol que j'ajoute plutôt que de le
laisser passer : à adam_eps=1e-12 et 1e-14, l'« écart bayésien » qu'il
cite (2,2e-16, -3,3e-16) est en dessous du plancher de précision de
float64 à cette échelle — les deux émetteurs impriment la même valeur
parce qu'ils LE SONT à seize chiffres, pas parce que le modèle a résolu
une asymétrie aussi fine. Seule la ligne à 1e-10 compare un vrai nombre
à un vrai nombre, et là le résidu du récepteur (9,0e-6) est bien cinq
ordres au-dessus de ce que l'asymétrie des émetteurs peut expliquer
(8,3e-11) — confirmé, ratio 1,1e5.

**Contrôle demandé et vérifié plus loin que demandé** : à eps=1e-8
(récepteur gelé), l'état n'est pas une égalité gelée à 0,5, c'est un
effondrement gelé sur le référent 3 (`r[10,3]=0,9999999997`, entropie
7,3e-9). Entraîné (eps=1e-10), l'entropie de la ligne du récepteur au
message 10 vaut **exactement ln(2) = 0,693147**, avec 100,0000000 % de
la masse sur les seuls référents 3 et 4 — sa prédiction confirmée sur
un chiffre qu'aucun de nous n'avait encore imprimé.

**Le test décisif — casser le prior plutôt que le push — tourné, et il
ne tombe sur AUCUNE des deux prédictions.** Référent 4 pondéré deux
fois plus que le référent 3 dans l'objectif (25 autres référents et les
deux termes d'entropie inchangés), 40 000 pas de plus sous
adam_eps=1e-10 :

```
controle symetrique (repete)      : R[10,4]=0,500000  H=0,693147
asymetrique (referent 4 pese 2x)  : R[10,4]=1,000000  H=0,000000
```

**Pas 0,5 (sa première prédiction, réfutée net — la moindre asymétrie
brise complètement l'égalité). Pas non plus 2/3 (sa seconde prédiction,
réfutée aussi — le système va jusqu'à la spécialisation totale).**

**Cherché plus loin sans qu'on me le redemande** (Théo : « cherche loin,
c'est important, ne t'arrête pas sur le premier résultat ») : dérivé la
condition du premier ordre — `log(r4/r3) = 2·delta/beta` pour
`w4=(1+delta)/N, w3=(1-delta)/N` — et balayé delta plutôt que de
rapporter le seul point testé :

```
delta    R[10,4] observe   sigmoid(2*delta/beta) predit
0,0000      0,500692            0,500000
0,0010      0,524979            0,524979   <- exact au chiffre
0,0020      0,549834            0,549834   <- exact au chiffre
0,0100      0,731486            0,731059
0,0200      1,000000            0,880797   <- la prediction casse ici
0,1000      1,000000            0,999955
```

**La loi molle colle exactement jusqu'à delta=0,01 (quatre décimales
ou mieux), puis un vrai bassin dynamique prend le relais entre 0,01 et
0,02 et le système saute à la spécialisation totale au lieu de suivre
la prédiction analytique (0,88 prédit contre 1,00 obtenu)** — la même
signature que le loquet du tour 34/35 (une porte qui se ferme une fois
un seuil franchi, pas une approche continue de l'optimum). Sa lecture
qualitative gagne entièrement : 0,5 n'est pas un point fixe qui résiste
au repondérage, c'est le point dégénéré d'une vraie loi bayésienne
continue, vérifiée à quatre décimales sur deux ordres de grandeur de
delta. Ce qui ne survit pas est le rapport quantitatif littéral
« 2:1 entre → 2/3 sorti » : le terme d'entropie de cet objectif est
bien trop faible face à un changement de poids d'une unité entière pour
tenir une valeur intermédiaire — ça sature bien avant que « 2× » n'y
arrive.

**Cherché plus loin sans qu'on me le redemande, une deuxième fois**
(Théo : « cherche encore, pose-toi des hypothèses toi-même... 3-5
hypothèses puis des questions ») : pourquoi la loi molle casse-t-elle
net entre delta=0,01 et 0,02 plutôt que de dévier progressivement ?
Cinq hypothèses formées puis testées avant d'écrire une explication :

```
H1 sous-entrainement        H2 plancher adam_eps (meme loquet que les murs)
H3 lr trop grand            H4 vraie bifurcation dynamique
H5 dependance au chemin (imposer l'asymetrie avant la convergence a 0,5)
```

**H1, H2, H3 et H5 toutes réfutées** : 200 000 pas de plus, adam_eps=1e-14,
lr=0,005, ou imposer le déséquilibre dès le checkpoint 10k — même résultat
bit-identique dans les quatre cas (R[10,4]=1,000000 exactement). Ce qui
reste (H4) a été creusé jusqu'à un mécanisme concret plutôt que laissé
comme un « c'est donc ça par élimination » : ma réduction analytique
supposait les deux émetteurs figés près de 1 ; ils ne le sont pas.

```
avant repondération : s[3,10]=0,999999999666   s[4,10]=0,999999999997
apres 40 000 pas    : s[3,10]=0,037064167       s[4,10]=0,999999999998
```

**Le référent 3 (celui dont le poids est réduit) voit sa propre
confiance s'effondrer à 1/27, uniforme — exactement la signature
d'évacuation des murs des tours 20 à 38.** Mon calcul fermé traitait
un problème à deux variables côté récepteur seul ; ce n'en est un que
dans la limite delta→0. Passé un certain déséquilibre, le référent 3
cesse de gagner assez de crédit pour justifier son engagement, son
propre terme d'entropie l'emporte, et l'effondrement du récepteur sur
le référent 4 est en aval de cette évacuation — pas une optimisation
séparée sur la seule ligne du récepteur. Même mécanisme que chaque mur
de ce projet, retrouvé par une porte d'entrée complètement différente.

Scripts : `verifier_prior_asymetrique.py`,
`verifier_prior_asymetrique_balayage.py`, `verifier_bassin_delta002.py`.

**Trouvé au passage, en recroisant d'anciens scripts contre leurs propres
chiffres publiés (pas une nouvelle critique, une vérification de plus)** :
`correction_de_selection.py` (tour 9, §7.26) a un vrai bug d'étiquetage.
Sa fonction `route_1_vectorisee`, sous l'étiquette « sigma connue »/« le
sien » vs « le mien », ré-estimait en réalité TOUJOURS sigma depuis
l'échantillon tiré — ce qui rend les deux lignes mathématiquement
incapables de différer (un contraste normalisé par un sigma ré-estimé
sur le même tirage est invariant à l'échelle des données générées).
Rejoué, le script donnait 1,630/2,456/0,1138, ni la ligne « sigma
connue » publiée (1,619-1,620/2,427/0,1066) ni exactement la ligne
« sigma ré-estimée à 145 ddl » (1,628/2,452/0,1130) — entre les deux.
Corrigé (paramètre `sigma_connue` explicite, les deux variantes
distinctes) : la vraie version à sigma fixe donne maintenant
1,621/2,432/0,1078, et l'écart RELATIF entre connue et ré-estimée
(q90 +0,024, P +0,0060) reproduit quasi exactement ce que la lettre
d'origine affirmait déjà (« q90 by 0.025 and P by 0.006 »). **La
conclusion scientifique publiée tient** — c'était le script de
vérification permanent qui avait un bug, pas le résultat lui-même.

**Journal des hypothèses de ce tour** (règle ajoutée le 14/09/2026,
appliquée ici rétroactivement) :

| # | hypothèse | posée le | statut |
|---|---|---|---|
| H1 | sous-entraînement (40 000 pas insuffisants à delta=0,02) | 14/09 | **réfutée** le 14/09 (200 000 pas de plus : toujours 1,000000) |
| H2 | plancher `adam_eps` (même loquet que les murs) | 14/09 | **réfutée** le 14/09 (adam_eps=1e-14 : toujours 1,000000) |
| H3 | `lr` trop grand, dépassement de l'optimum | 14/09 | **réfutée** le 14/09 (lr=0,005 : toujours 1,000000) |
| H4 | vraie bifurcation dynamique (le point intérieur cesse d'être atteignable) | 14/09 | **retenue** le 14/09, creusée jusqu'au mécanisme (effondrement du référent 3 à 1/27) |
| H5 | dépendance au chemin (imposer l'asymétrie avant la convergence à 0,5) | 14/09 | **réfutée** le 14/09 (même résultat en partant du checkpoint 10k) |

Réponse dans `docs/REPONSE_ORDRE49.md`.

---

### 7.62 Quarante-neuvième critique : les quatre bras étaient tous sur la même frontière saturée — et son correctif d'entropie ne sauve pas la loi molle non plus

14/09/2026. Il relit mes quatre « réfutations » (H1, H2, H3, H5) et pointe
qu'elles sont **toutes mesurées à delta=0,02, déjà saturé** (R=1,0, H=0,0
partout) — cinq bras assis sur la même frontière ne peuvent rien séparer,
le résultat bit-identique n'est pas une preuve de robustesse, c'est ce
qu'une observable saturée rend quoi qu'on varie. Son test : balayer delta
dans la région graduée (où le nombre bouge encore) à plusieurs budgets, et
localiser `delta_c` plutôt que le brancher.

Deuxième point, plus gros selon lui : `entropie_s` est une moyenne NON
pondérée alors que la récompense l'est. Le référent 3 perd du signal de
récompense mais rien ne réduit la pression d'entropie qui le retient
engagé — un ratio récompense/entropie déséquilibré plutôt qu'un vrai
mécanisme d'éviction. Chiffré : sous la famille du balayage à
delta=0,02, le référent 3 ne perd que 2 % de son poids (de 1/N à
0,98/N), et pourtant `s[3,10]` s'effondre de 0,999999999666 à
0,037064167 ≈ 1/27 — rien dans cette fourchette n'évacue une ligne
saturée à neuf neuf ; un régularisateur mal mis à l'échelle, si.
Correctif d'une ligne proposé : pondérer `entropie_s` par `(N*poids[i])`
pour que chaque référent factorise `poids[i]*(récompense_i + beta*H_i)`.

**Ajouté comme demandé : `etat()` retourne maintenant aussi
`s[3,msg]`/`s[4,msg]`.**

**Test 1 — `delta_c` à trois budgets (15 000, 40 000, 200 000 pas), grille
resserrée 0,010-0,020 :**

```
delta=0,010  R[10,4] identique aux trois budgets : 0,731485-0,731486
delta=0,012  identique aux trois : 0,771352
delta=0,014  sature aux trois : R=1,000000, s[3,10]≈0,037
```

**`delta_c` se situe entre 0,012 et 0,014, invariant sur un facteur 13
en durée d'entraînement.** H1 meurt pour de vrai cette fois, testé là où
un déplacement aurait pu apparaître, pas là où tout est déjà saturé — et
ça resserre la fourchette elle-même, de (0,01 ; 0,02) à (0,012 ; 0,014).

**Test 2 — échelle `adam_eps` à delta=0,015 (fixé avant d'avoir la
fourchette resserrée, donc déjà au-delà du bord réel) :** invariant sur
quatre ordres de grandeur (1e-10 à 1e-14, tous à R=1,000000,
s[3,10]≈0,037) — H2 ne ressuscite pas ici non plus, même si le test
mériterait d'être refait pile dans (0,012 ; 0,014).

**Test 3 — son correctif d'entropie, rejoué jusqu'au point 2:1 littéral
(delta=1,0) :** la loi molle ne survit PAS au-delà de la même fourchette
qu'avant correction — `delta=0,01` donne 0,731188 (quasi identique à la
version non pondérée), et l'effondrement à `s[3,10]≈0,037` est déjà
complet à `delta=0,02`. **Le correctif ne déplace pas le bord.** Un piège
signalé plutôt que laissé passer pour une confirmation : à `delta=1,0`
exactement, `poids[3]=0`, donc sous son correctif le terme d'entropie de
la ligne 3 est AUSSI multiplié par zéro — récompense et entropie
s'annulent ensemble, le gradient du référent 3 est nul partout, et il
reste simplement figé où il était avant cette étape (`s[3,10]=1,0`
lu comme si c'était une confirmation, c'est en réalité un artefact de
bord où les deux termes se coupent en même temps).

**Journal des hypothèses de ce tour :**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| (budget) sous-entraînement, testé correctement cette fois | 14/09 (lui) | **réfutée** le 14/09 (invariant sur 15k/40k/200k pas dans la région graduée) |
| (entropie) régularisateur mal mis à l'échelle explique le bord | 14/09 (lui) | **réfutée** le 14/09 (le correctif d'une ligne ne déplace pas `delta_c`) |
| (adam_eps, bissection resserrée) même loquet que les murs, testé pile dans (0,012 ; 0,014) | 14/09 (moi) | **réfutée** le 14/09 (delta=0,013 : 0,794022 identique sur 1e-10/1e-12/1e-14) |
| H6 | vraie bifurcation nœud-col dans le système couplé à 4 variables | 14/09 (moi) | ouverte, favorite |
| H7 | condition du premier ordre côté émetteur (pas récepteur) fixe le seuil | 14/09 (moi) | ouverte |
| H8 | résonance `beta2` d'Adam avec le gradient qui s'amenuise | 14/09 (moi) | ouverte |
| H9 | artefact du softmax à 27 voies vs un jouet à 2 référents seulement | 14/09 (moi) | ouverte |
| H10 | hystérésis : monter delta en continu plutôt que par sauts | 14/09 (moi) | ouverte |

**Bissection resserrée refaite pile dans (0,012 ; 0,014)** (pas à 0,015,
déjà au-delà du bord réel) : à delta=0,013, R[10,4]=0,794022 — dans la
région graduée cette fois (proche de la prédiction molle 0,785835),
**bit-identique sur adam_eps=1e-10/1e-12/1e-14.** Le bord de bassin n'est
définitivement pas le même mécanisme que le loquet des murs.

Scripts : `verifier_invariance_budget.py`, `verifier_adam_eps_ladder_delta.py`
(maintenant à delta=0,013), `verifier_entropie_reponderee.py`. Réponse dans
`docs/REPONSE_ORDRE50.md`.

---

### 7.63 Cinquantième critique : contre H6 (nœud-col) — pas de ralentissement critique, mais la continuation ne survit pas non plus

14/09/2026. Il reprend mes trois points survivants et montre qu'un
nœud-col doit COURBER la branche à l'approche de `delta_c` (le point
stable voyage vers l'instable), alors que la mienne reste analytique
puis s'arrête net. Chiffré : le résidu (mesuré moins prédit par la loi
molle) suit le déficit de l'émetteur `(1-s[3,10])` à un coefficient
~8-10 qui dérive doucement — la signature de H7 (terme du premier ordre
côté émetteur jamais dérivé), pas d'une bifurcation. Extrapolé en
log-linéaire depuis (0,012 ; 0,013) vers 0,014, sa branche prédit
`s[3,10]=0,996906` ; mesuré : `0,037023` — 311 fois l'écart prédit, et
ça atterrit pile sur 1/27 (neuf lignes post-effondrement à
0,037019±3,4e-5). Pas de pôle proche non plus (ajustement en loi de
puissance sur les trois résidus : pôle au-delà de 0,5). Et le test du
tour précédent est une preuve CONTRE H6, pas neutre : un nœud-col
impose un ralentissement critique en `(delta_c-delta)^(-1/2)`, or
15 000 pas donnaient déjà la réponse à 200 000 pas à 10 % du bord.

**Deux expériences proposées, les deux tournées avant d'écrire une
interprétation.**

**Continuation** (partir de l'état déjà convergé à delta=0,013, pas de
l'égalité fixe de départ) :

```
etape 1, converge a 0,013 :        R[10,4]=0,794022  s[3,10]=0,999000
etape 2, continue vers 0,014 :     R[10,4]=1,000000  s[3,10]=0,037049
controle, depart neuf a 0,014 :    R[10,4]=1,000000  s[3,10]=0,037016
```

**Même en partant d'un point déjà sur la branche graduée, ça s'effondre
quand même.** Ni tout à fait sa lecture « la borne s'est déplacée sur un
point de départ fixe », ni un nœud-col classique.

**Ralentissement critique**, `delta_c` bissecté d'abord à
`(0,013422 ; 0,013437)` (fourchette resserrée d'un ordre de grandeur),
puis temps de convergence à 3 distances de ce bord :

```
3,000 % sous delta_c : converge au pas 200
0,300 % sous delta_c : converge au pas 800
0,030 % sous delta_c : converge au pas 800
```

**Plat, pas de divergence.** Deux ordres de grandeur plus près du bord
et rien ne bouge après le premier saut. **H6 meurt exactement comme il
l'a annoncé, sur le test qu'il a lui-même proposé.**

**Ce qui reste : ni H6 ni sa lecture initiale ne collent seules aux
trois faits à la fois** (pas de courbure, pas de ralentissement, la
continuation depuis la branche ne survit pas non plus). Piste retenue :
une **crise de bord** (le bassin de la branche est balayé par une
frontière étrangère plutôt que la branche elle-même perdant sa
stabilité) — explique un saut net sans ralentissement local, et une
continuation qui échoue même depuis un point déjà sur la branche.

**Journal des hypothèses de ce tour :**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| H6 (tour précédent) | vraie bifurcation nœud-col | 14/09 | **réfutée** le 14/09 (pas de courbure, pas de ralentissement critique — les deux tests que lui-même a proposés) |
| (continuation) la borne a juste dépassé le point de départ fixe | 14/09 (lui) | **partiellement réfutée** le 14/09 (la continuation depuis la branche elle-même ne survit pas non plus) |
| H11 | crise de bord (frontière de bassin étrangère qui balaie la branche) | 14/09 (moi) | ouverte, favorite |
| H7 | condition du premier ordre côté émetteur, dérivée proprement | 14/09 | **confirmée** le 14/09 (formule fermée, écart <0,1% aux trois points, contre r3 mesuré) |
| H12 | `exp_avg_sq` de la ligne du référent 3, pas `adam_eps`, comme vraie porte | 14/09 (moi) | ouverte |
| H13 | jouet à 2 référents seulement (le référent 3 vide dans 25 lignes, pas 1) | 14/09 (moi) | ouverte |
| H14 | `delta_c≈0,01343` est un artefact numérologique de N=27, pas dynamique | 14/09 (moi) | ouverte |
| H15 | Adam masque le ralentissement critique (pas de vraie signature, pas H6 blanchie) | 14/09 (moi) | **réfutée** le 14/09 (même plat sous SGD pur, lr=50) |

**Poussé plus loin sans qu'on me le redemande (Théo : « tu comprends
pourquoi ? pourquoi ça bouge ? »), au lieu de laisser H7 en simple
« coefficient qui dérive » :** réduit la ligne du référent 3 à un
softmax à 2 issues (message 10 contre les 26 autres regroupées), résolu
sa condition stationnaire contre `r3` MESURÉ (pas ajusté) à chaque
delta :

```
d3 = 26 * exp(-N * poids[3] * r3 / beta)

delta   r3        d3 predit    d3 mesure    ecart
0,010   0,268514  4,3911e-05   4,3900e-05   +0,02 %
0,012   0,228648  3,2324e-04   3,2310e-04   +0,04 %
0,013   0,205978  1,0008e-03   9,9980e-04   +0,10 %
```

**Formule fermée, pas un ajustement — écart sous 0,1% aux trois points.**
Le coefficient « 8-10 » de dipankar dérive parce que `d3` dépend de `r3`
de façon EXPONENTIELLE, et `r3` baisse avec delta : une pente locale qui
dérive est exactement ce que donne une exponentielle fixe regardée sur
une fenêtre où son argument bouge. Et cette formule n'a elle-même aucune
singularité près de `delta_c` — lisse partout, cohérent avec un
mécanisme couplé `(s3,r3)` plutôt qu'un pôle dans cette approximation à
une variable.

**Puis formé et testé H15** (pourquoi le test de ralentissement critique
n'a rien montré : l'hypothèse qu'Adam, en divisant son pas par la racine
du second moment, efface la signature de ralentissement même si le
mécanisme sous-jacent est un vrai noeud-col) : rejoué le même test à
trois distances sous SGD pur (lr=50) :

```
3,000 % sous delta_c : converge au pas 400
0,300 % sous delta_c : converge au pas 400
0,030 % sous delta_c : converge au pas 400
```

**Plat aussi sous SGD.** H15 réfutée par le test même qui l'aurait
confirmée : l'absence de ralentissement critique n'est pas un artefact
d'Adam, c'est une vraie propriété de la dynamique, indépendante de
l'optimiseur qui la parcourt — un point de plus pour H11 (crise de
bord) contre H6.

**Ablation directe de H13, appliquée tout de suite plutôt que laissée en
remarque** (Théo : « pourquoi si je supprime ou bouge un truc ») :
construit un jouet autonome, deux émetteurs à 2 issues (message 10 contre
UNE seule catégorie « ailleurs » au lieu de 26) et un récepteur à 2 issues
(référent 3 contre référent 4, pas 27), même beta=0,02, même
normalisation /N=27 :

```
delta=0,014  R4=0,802653  s3=0,999940  (encore gradue ici, le vrai systeme sature deja)
delta=0,02   R4=1,000000  s3=0,500081  (effondre, atterrit sur 1/2 pas 1/27)
```

**Les deux prédictions de H13 tombent juste : l'effondrement atterrit
sur 1/2 (pas 1/27), et `delta_c` a bougé** — bissecté dans le jouet à
`(0,018688 ; 0,018711)` contre `(0,013422 ; 0,013437)` dans le vrai
système. **Les 25 autres lignes ne sont pas des passagers : elles
fixent où atterrit l'effondrement ET où se situe le seuil.**

**Compris pourquoi, pas seulement noté que.** Dans la formule fermée de
H7, `d3 = 26·exp(-N·w3·r3/beta)`, le « 26 » est un préfacteur : moins de
destinations possibles (1 au lieu de 26) veut dire moins d'entropie à
gagner en abandonnant le message 10, donc le référent résiste plus
longtemps et il faut pousser plus fort (delta plus grand) avant que
lâcher prise devienne rentable. C'est le mécanisme, pas juste
l'observation que le seuil a bougé.

**H9 se referme du même coup** : c'était exactement la même ablation
proposée sous un autre nom ; atterrir sur 1/2 au lieu de 1/27 EST le
résultat de H9.

**Journal des hypothèses, mise à jour :**

| H13 | jouet à 2 référents seulement | 14/09 (moi) | **confirmée** le 14/09 (atterrit sur 1/2, `delta_c` déplacé à (0,0187 ; 0,0187), mécanisme identifié via le préfacteur de H7) |
| H9 | artefact du softmax à 27 voies | 14/09 (moi) | **close** le 14/09, absorbée par H13 (même test, même résultat) |

Scripts : `verifier_continuation_delta.py`, `verifier_bissection_delta_c.py`,
`verifier_ralentissement_critique.py`, `verifier_ralentissement_sgd.py`,
`verifier_jouet_2_referents.py`. Réponse dans `docs/REPONSE_ORDRE51.md`.

---

### 7.64 Cinquante-et-unième critique : H6 confirmée — la sonde de bassin trouve un seuil net à moins de 1e-4 du jumeau prédit

14/09/2026. Il reprend H7 en le résolvant CONJOINTEMENT (récepteur et
émetteur ensemble, pas récepteur contre un `r3` fixé) et obtient un
système à deux équations couplées :

```
recepteur : logit(R)*beta = 2*delta + (1-delta)*d3
emetteur 3: d3/(26(1-d3)) = exp(-(1-delta)*(1-R)/beta)
```

**Vérifié indépendamment : les deux équations reproduisent mes trois
lignes publiées à 5-6 chiffres.** Le système a trois racines à
delta=0,013 (stable à 9,998e-04, instable à 5,700e-03, effondrée à
0,962963=26/27) — un vrai pli. `delta_c=0,013437210` (mon encadrement :
0,013422-0,013437, tombe pile sur le bord haut). Sur mon propre jouet à
2 référents (H13), sa formule prédit `delta_c=0,018699` sans aucun
réajustement — mon encadrement était (0,018688 ; 0,018711). Il montre
aussi que mon « pas de ralentissement critique » était un artefact de
mesure (fenêtre de bissection plus large que les distances sondées) et
que ma continuation ratée est exactement ce qu'un nœud-col annihilé
prédit.

**Test précommis, tourné en deux temps.** Round 1 : perturber s3 SEUL
(comme sa phrase le suggérait), récepteur laissé à l'égalité de départ
(0,5) — **aucun effondrement, même à s3=0,98.** Compris pourquoi avant
de conclure : le point instable prédit a AUSSI une coordonnée récepteur
(R≈0,829390, dérivée de sa propre équation), et laisser R à 0.5 donne
au référent 3 plus de crédit de récompense que le point instable n'en
prévoit — assez pour le ramener sur la branche quel que soit son
déficit de départ.

**Round 2, corrigé : s3 ET R placés ensemble sur le point instable
prédit (0,994300 ; 0,829390), puis seul s3 varié :**

```
s3=0,99430  -> branche graduee
s3=0,99420  -> EFFONDRE (1/27)
```

**Bascule nette entre ces deux points, à moins de 1e-4 du jumeau
prédit — exactement le critère qu'il a posé.** En dessous du seuil,
tout s'effondre jusqu'à s3=0,90 inclus ; au-dessus, tout remonte.
**H6 confirmée, H11 (crise de bord) n'a plus rien à expliquer que H6
ne couvre pas déjà mieux.**

**Journal des hypothèses :**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| H6/nœud-col résolu conjointement | delta_c est un vrai pli du système couplé | 14/09 (lui) | **confirmée** le 14/09 (seuil net à <1e-4 du jumeau prédit) |
| H11 | crise de bord | 14/09 (moi) | **abandonnée** le 14/09 — plus rien à expliquer que H6 ne couvre |
| (round 1) perturber s3 seul suffit à tester le jumeau | 14/09 | **réfutée** le 14/09 (le récepteur hors variété masque tout) |

**Théo a demandé « il a eu faux sur des choses ? »** Vérifié plusieurs
de ses affirmations indépendamment (recherche de racines numérique sur
son système, pas juste relu) : sa formule du pli et son `delta_c` se
confirment exactement (bifurcation entre 0,0134372 et 0,013438), et sa
prédiction sur le jouet (`delta_c=0,018699`) aussi, sans réajustement.
**Rien de faux trouvé chez lui.** Mais en le vérifiant, une
incohérence réelle est apparue **dans mes propres chiffres publiés** :
la ligne « 3 % sous delta_c » de `verifier_ralentissement_critique.py`
(R=0,792836) ne correspond pas au delta annoncé (0,013026615) selon la
forme fermée (qui prédit 0,794755 à ce delta) — exactement ce qu'il
avait repéré.

**Creusé jusqu'au mécanisme plutôt que corrigé en silence.** Rejoué la
même cellule à 400 000 pas au lieu de 60 000, avec un point tous les
20 000 pas :

```
pas       R[10,4]
 20 000   0,794794
 60 000   0,795968
160 000   0,792799   <- c'est de la que venait le 0,792836 original
380 000   0,794756
```

**Ce n'est pas une convergence lente qui n'avait pas fini — la
trajectoire est déjà sur la branche prédite (0,794756) presque partout,
avec des excursions ponctuelles et brèves vers ~0,7928.** Mon
instantané original (pas=60 000) a capturé une de ces excursions par
hasard, pas un état non convergé. **Conséquence : tout le tableau de
ralentissement critique à trois points doit être reconstruit** — un
instantané unique à un pas rond peut se tromper de 0,002 selon
l'excursion qu'il capture, indépendamment de la question du
ralentissement lui-même.

**Deux hypothèses sur ces excursions, posées avant d'en tester
aucune :**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| (mode lent réel) les excursions sont un mode propre du système linéarisé près du pli, pas du bruit | 14/09 (moi) | ouverte |
| (artefact Adam) `beta2=0,999` laisse une suite de gradients corrélés biaiser brièvement le second moment | 14/09 (moi) | ouverte |

Scripts : `verifier_sonde_bassin.py`. Réponse dans `docs/REPONSE_ORDRE52.md`.

---

### 7.65 Cinquante-deuxième critique : sa forme fermée ne dessine pas la séparatrice, elle a besoin de k — et k n'est pas constant

15/09/2026. Il pointe que tous mes chiffres jusqu'ici (points fixes,
`delta_c`, jumeau, valeur de branche) sont des intersections de
courbes isoclines — ils ne dépendent pas de la VITESSE relative des
deux joueurs. La séparatrice, elle, en dépend. Fournit l'ODE à deux
échelles de temps :

```
dx/dt = x_br(R) - x
dR/dt = k * (R_br(x) - R)
```

et une table de points de bascule à R_init=0,60 pour k=0,5 à 2,0, plus
`k*=0,428762` (en dessous, la séparatrice ne touche plus jamais R=0 —
c'est round 1 rendu littéral).

**Vérifié indépendamment (pas pris pour acquis — règle 5bis) : intégration
numérique de sa propre ODE, sans lire son tableau.** Table de k
reproduite à 5-6 chiffres, `k*=0,428762` retrouvé exactement en
cherchant où R_min franchit zéro. Rien de faux trouvé chez lui, une
deuxième fois.

**Sa question directe — « does your SGD arm show the excursions at
all? » — testée immédiatement.** Rejoué 400 000 pas sous SGD pur
(lr=50) à la même cellule qui montrait des excursions sous Adam :

```
pas=20 000 a 380 000 : R[10,4] = 0,7862825715, plat a la dixieme decimale
```

**Zéro excursion.** Confirme sa lecture (artefact du second moment
d'Adam) contre la mienne (mode propre du système linéarisé). Nouveau
mystère trouvé au passage : SGD converge à 0,786283, pas 0,794756
(valeur de branche prédite par la forme fermée) — un écart stable de
0,0085, pas du bruit. Pas encore expliqué.

**Protocole « épingler avec un essai, falsifier avec deux » exécuté
tel quel.** Point de bascule à R_init=0,60 : s3=0,979619 (bissection à
1e-5). k interpolé ≈1,586. Prédictions pour R_init=0,75 et 0,50 :

```
predit  flip@0,75 = 0,987148    mesure = 0,989740    ecart +0,0026
predit  flip@0,50 = 0,975765    mesure = 0,972652    ecart -0,0031
```

**Les deux prédictions manquent, dans des sens opposés.** k réajusté à
chaque point plutôt que d'ignorer l'écart :

```
R_init=0,75 -> k=2,449
R_init=0,60 -> k=1,586
R_init=0,50 -> k=1,418
```

**k n'est pas une constante — il chute d'environ 42 % entre R_init=0,75
et 0,50.** Le protocole a fait exactement son travail : pas seulement
mesurer k, mais montrer que l'hypothèse centrale du modèle (un seul
rapport de vitesses) ne tient pas sur cette plage — trouvé avec trois
essais, pas une grille. Cohérent avec l'hypothèse d'excursion d'Adam :
si le second moment s'adapte différemment pour l'émetteur et le
récepteur selon leur propre historique de gradient, le k effectif
qu'une trajectoire Adam traverse n'est pas une constante du système,
c'est une quantité dépendante de l'état.

**Grille QUAND/COMMENT/POURQUOI/OÙ/COMBIEN/JUSQU'OÙ/DEPUIS QUAND/SUR
COMBIEN appliquée explicitement** (pas seulement POURQUOI) — résumée
dans la lettre anglaise plutôt que redupliquée ici.

**Journal des hypothèses :**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| (excursions = mode propre) | 14/09 (moi) | **rouverte** le 17/09 (le test SGD qui l'a réfutée avait le référent 3 gelé — voir ci-dessous) |
| (excursions = artefact second moment Adam) | 14/09 (lui) | **rouverte** le 17/09 (idem — plus de preuve nette dans un sens ou l'autre) |
| (k constant) | implicite au tour 51 | **réfutée** le 15/09 (k=1,42 à 2,45 selon R_init) |
| (SGD converge à la vraie valeur de branche) | 15/09 (implicite) | **réfutée** le 15/09 (0,786283 contre 0,794756 prédit — pas encore expliqué) |
| k(R) standard, fonction lisse de l'état seul | 15/09 (moi) | **soutenue** le 15/09 (survit aux deux réfutations ci-dessous) |
| k dépend du chemin (budget d'entraînement), pas de l'état | 15/09 (moi) | **réfutée** le 15/09 (identique à 40k et 200k pas : k=1,4180 pile) |
| k dérive à cause de `beta1` (momentum), pas de `beta2` seul | 15/09 (moi) | **réfutée** le 15/09 (l'écart s'agrandit sans momentum, 1,326 contre 1,031 — mauvais sens) |

**Testées, pas seulement posées** (Théo : « tu n'as pas les hypothèses
j'ai vu »). Rejoué le point R_init=0,50 à 200 000 pas au lieu de 40 000 :
k identique à quatre décimales — pas un artefact de convergence. Rejoué
les trois points de bascule avec `beta1=0` (Adam sans momentum) : les
trois k montent (3,406/2,316/2,080 contre 2,449/1,586/1,418), mais
l'ÉCART entre R=0,75 et R=0,50 grandit (1,326 contre 1,031) au lieu de
se resserrer — le test qui aurait validé l'hypothèse la tue à la place.
**Ce qui reste : `k(R)` est une vraie fonction de l'état, invariante au
budget et indépendante du momentum** — probablement dans le second
moment (`beta2`, non testé directement) ou dans le fait que le modèle
réduit à 2 variables est une projection incomplète du vrai système à
27 référents (les 25 autres lignes, déjà porteuses pour `delta_c` et la
cible d'effondrement au tour 49, pourraient l'être ici aussi).

Scripts : `verifier_pin_k.py`, `verifier_ode_separatrice.py`,
`verifier_excursions_sgd.py`, `verifier_derive_k.py`. Réponse dans
`docs/REPONSE_ORDRE53.md`.

**Repris seul, en son absence** (Théo, 17/09/2026 : « depuis 2 jours il ne
répond pas ... on va continuer à chercher nous-mêmes »). Le mystère laissé
ouvert dans ce même tour — SGD converge à 0,786283, pas 0,794756 prédit —
creusé jusqu'au bout plutôt que laissé en flag :

```
190 pas consecutifs logges un par un : R[10,4]=0,786282571470, inchange a 12 chiffres
norme du gradient a ce point : 3,97e-12
s3 = 0,999999999665   (baseline pre-perturbation : 0,999999999666)
```

**Vrai point fixe (gradient nul), pas une approche lente. Et s3 est resté
COLLE a sa valeur de depart — sous SGD pur, le referent 3 n'a jamais bougé
du tout.** Son gradient brut pres de la saturation est minuscule (meme
mecanisme que toutes les formules de deficit de ce tour) et sans la mise a
l'echelle par parametre d'Adam, lr=50 ne suffit pas a le deplacer en
390 000 pas — alors que le recepteur, parti d'un R=0,5 encore "vivant",
bouge librement. **0,786283 n'est pas un autre point de la vraie branche
couplee : c'est ce que le recepteur seul atteint face a un emetteur que
SGD n'a jamais reellement entraine** — la meme signature "gele, sans
valeur" que ce projet trouve depuis le tour 33 sous les planchers
`adam_eps`, reproduite ici par un mecanisme totalement different (pas de
plancher adaptatif du tout — juste un gradient brut trop faible face a un
pas fixe). **Consequence : le test "zero excursion sous SGD" tient toujours
mais pèse moins que prevu** — un systeme ou un joueur est gele en
permanence ne peut pas montrer d'excursions par construction ; un vrai
test sous SGD demanderait un taux d'apprentissage calibre separement par
parametre, ce qui va a l'encontre de l'idee de tester "SGD pur".

**Croisé avec le carnet avant de crier au nouveau résultat (règle « le
standard a des biais », mais ici appliquée à ma propre nouveauté plutôt
qu'à un seuil reçu) : ce n'est PAS un phénomène nouveau.** Même
signature exactement au tour 47 (`REPONSE_ORDRE48.md`, §7.60quinquies) :
SGD à 100× le taux d'Adam sur le mur idx5 laissait `r[0,0]` à exactement
0,000000 et `1-s[0,0]` presque immobile — même conclusion tirée alors
(« l'échelle adaptative d'Adam ne remplace pas juste un pas plus grand,
elle fait quelque chose que SGD ne peut pas reproduire »). Deux
collisions indépendantes (référents 3/4 ici, référent 0 d'idx5 là-bas),
deux dispositifs de perturbation différents, même mécanisme les deux
fois — ça renforce la lecture au lieu de la répéter : c'est une
propriété de la géométrie de l'objectif près de la saturation, pas un
artefact d'un run particulier.

Script : `verifier_point_fixe_sgd.py`.

**Tentative de correction du test H15, échouée à moitié — rapportée
honnêtement plutôt que cachée.** Objectif : SGD à deux taux
d'apprentissage (émetteur compensé), pour que le référent 3 bouge
vraiment et voir si le ralentissement critique apparaît alors.

Premier essai : calibré sur la norme du tenseur émetteur ENTIER
(729 cases, `e.p[0]`). `|grad_e|=1,146e-07`, `|grad_r|=3,405e-04`, ratio
2970, `lr_e=1,485e5`. **Résultat : `|déplacement s3|≈1,15e-11` — quasiment
rien.** La norme du tenseur entier est dominée par d'autres lignes que
[3,10], pas une calibration valide pour LA case qui compte.

Deuxième essai : gradient précis de la case `[3,10]` seule.
`|grad e[3,10]|=8,65e-14`, `|grad r[10,4]|=2,41e-04`, ratio 2,78e9,
`lr_e≈1,39e11`. **Pas tenté tel quel — trop dangereux** (un `lr` de cet
ordre appliqué à tout le tenseur émetteur risquerait de faire exploser
les autres cases, pas seulement de débloquer [3,10]).

**Diagnostic, pas juste un échec technique** : le gradient réel de
référent 3 est si minuscule (8,65e-14) que même un SGD "corrigé" par un
facteur constant se heurte au même mur que le SGD nu — la solution n'est
probablement pas un `lr` plus grand mais une normalisation ADAPTATIVE
(comme Adam), spécifiquement pour cette ligne. **Piste retenue pour la
suite** : optimiseur hybride, Adam sur l'émetteur seul + SGD pur sur le
récepteur seul — teste si c'est l'adaptativité du récepteur ou celle de
l'émetteur qui porte les excursions. Pas encore construit.

**Journal :**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| SGD à deux taux (norme totale du tenseur) fait bouger le référent 3 | 17/09 (moi) | **réfutée** le 17/09 (déplacement ~1e-11, calibration invalide) |
| SGD à deux taux (gradient précis [3,10]) est faisable sans risque | 17/09 (moi) | **réfutée** le 17/09 (lr requis ~1,4e11, trop dangereux à tester tel quel) |

**Optimiseur hybride construit et tourné** (`verifier_optimiseur_hybride.py`) :
Adam sur l'émetteur seul, SGD pur sur le récepteur seul. Le référent 3
s'entraîne vraiment cette fois (`s3` se stabilise à 0,998963, conforme à
la prédiction fermée de H7 — pas gelé à la valeur de départ) et `R`
converge exactement vers la vraie valeur de branche 0,794756 (pas
l'artefact du référent gelé, 0,786283).

```
pas=60 000    R[10,4]=0,7947553911  s3=0,9989578477   <- excursion
pas=300 000   R[10,4]=0,7947549760  s3=0,9989496174   <- excursion, plus grande
```

**Les excursions sont TOUJOURS LÀ**, plus petites que sous Adam complet
(~1e-5 contre ~2e-3) mais réelles, et au même genre d'endroit (pas=60000
apparaît dans les deux traces). **Ça localise le mécanisme : les
excursions survivent avec un récepteur non-adaptatif (SGD pur), donc
elles ne viennent PAS du récepteur — elles sont portées par le second
moment de l'ÉMETTEUR seul.**

**H15 (artefact du second moment d'Adam) confirmée pour de bon, et plus
précisément qu'avant** — pas "Adam cause des excursions" en général,
mais spécifiquement "la normalisation adaptative de l'ÉMETTEUR près de
sa propre saturation le fait". L'hypothèse concurrente (mode propre du
système linéarisé complet) devient difficile à soutenir : un vrai mode
dynamique du système couplé devrait apparaître quel que soit le joueur
adaptatif, et ce n'est pas le cas ici.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| (excursions = mode propre du système couplé) | 14/09 (moi) | **réfutée** le 17/09 (persistent avec récepteur SGD pur — ne dépend pas du couplage complet) |
| (excursions = artefact second moment Adam, côté émetteur) | 17/09 (moi, affiné de l'hypothèse initiale de lui) | **confirmée** le 17/09 (survit à un récepteur non-adaptatif, localisée côté émetteur) |

Script : `verifier_optimiseur_hybride.py`. Voir `ETAT.md` à la racine du
dépôt pour la reprise de ce fil dans une nouvelle session — mis à jour
en conséquence, ce fil est maintenant clos (H15 tranchée).

---

### Ralentissement critique, refait avec l'optimiseur qui marche

Même protocole que §7.63/§7.64 (trois distances à delta_c, temps de
convergence à 99 %), mais avec l'hybride Adam-émetteur/SGD-récepteur au
lieu du SGD pur qui gelait le référent 3 :

```
3,000 % sous delta_c : R_final=0,794756  converge au pas 800
0,300 % sous delta_c : R_final=0,808282  converge au pas 1000
0,030 % sous delta_c : R_final=0,811395  converge au pas 1000
```

**Les valeurs de R finales collent exactement à la forme fermée et aux
valeurs déjà mesurées sous SGD gelé** (0,808282 contre 0,808275,
0,811395 contre 0,811395 identique) — le récepteur seul atteignait déjà
la bonne valeur même avec un émetteur gelé, la coïncidence des deux
tests précédents n'était donc pas fausse sur CE point-là, seulement sur
la question du gradient/ralentissement.

**Petite hausse (800→1000, +25 %) entre le premier et le deuxième point,
puis plat.** Pas la forme d'une vraie divergence en racine carrée (qui
prédirait une accélération continue à l'approche de `delta_c`), mais pas
non plus parfaitement plat comme le test cassé le laissait croire.
**Verdict : H6 (pas de fort ralentissement critique) tient globalement,
mais la granularité de mesure (pas de 200) est trop grossière pour
trancher entre « un peu de ralentissement réel » et « rien du tout ».**
Laissé ouvert plutôt que forcé à une conclusion nette.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| H6 (pas de ralentissement critique fort) tient avec le bon optimiseur | 17/09 (moi) | **affinée** le 17/09 — voir résolution fine ci-dessous |

**Refait à résolution 10× plus fine** (`CHECK_TOUS=20` au lieu de 200,
même protocole sinon) :

```
760 pas   (3,000 % sous delta_c)
880 pas   (0,300 % sous delta_c)   ratio vs precedent : 1,158
940 pas   (0,030 % sous delta_c)   ratio vs precedent : 1,068
```

**Une vraie hausse monotone apparaît, pas plate comme à résolution
grossière — mais bien plus faible qu'une divergence en racine carrée
classique.** Une vraie loi `(delta_c-delta)^(-1/2)` prédirait un
facteur `sqrt(10)≈3,16` par décade de distance ; on observe 1,16 puis
1,07 — un ralentissement réel, mesurable, mais loin d'être le
ralentissement critique complet d'un nœud-col 1D classique.

**Lecture retenue : ni « H6 pur » ni « H11 pur ».** Le système montre un
vrai signe de ralentissement en approchant `delta_c` (contre l'ancienne
lecture "parfaitement plat" qui était un artefact de mesure), mais
l'effet est nettement plus faible que la loi théorique à une variable.
Cohérent avec un système ou le ralentissement d'un noeud-col existe mais
est partiellement masqué/amorti par le couplage a la dynamique rapide du
recepteur (SGD ici) — une lecture intermediaire, pas tranchee entre les
deux camps, a garder ouverte plutôt que forcee.

Script : `verifier_ralentissement_hybride.py`.

---

### k(R) : la piste beta2 testée, réfutée dans le même sens que beta1

Étape 2 de `ETAT.md` : `H_chemin` et `H_momentum` (beta1) réfutées au
tour 52, restait `beta2` (mémoire du second moment) jamais testée
directement. Trois valeurs de beta2 à `R_init=0,60` fixe, delta=0,013 :

```
beta2=0,99    flip=0,979397  k_fit=1,5676
beta2=0,999   flip=0,979616  k_fit=1,5859   (defaut, coherent avec la baseline)
beta2=0,9999  flip=0,979643  k_fit=1,5883
```

**Effet minuscule** (étendue totale 0,0207 sur deux ordres de grandeur
de `1-beta2`) **comparé à la dérive de k avec R_init** (~1,03 entre
R=0,75 et R=0,50, tour 52). `H_beta2` réfutée, dans le même sens que
`H_momentum` : ni beta1 ni beta2 pris isolément n'expliquent la dérive
de k avec l'état de départ.

**Ce qui reste : `k(R)` est de plus en plus solidement une propriété de
l'ÉTAT lui-même (pas d'un réglage particulier d'Adam), cohérent avec
l'hypothèse des 25 autres lignes du système complet (non testée
directement, toujours ouverte).**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| k dérive à cause de `beta2` (mémoire du second moment) | 17/09 (moi) | **réfutée** le 17/09 (effet minuscule, 0,02 contre ~1,03 de dérive totale) |

Script : `verifier_k_beta2.py`.

---

### Deuxième égalité retrouvée : référents 23/25, message 13 (recette perdue depuis le tour 30-31)

Étape 3 de `ETAT.md` (reproduire sur une autre graine) demandait une
deuxième égalité indépendante. Celle des tours 30-31 (§7.47-48,
référents 23/25, message 13) n'avait jamais été sauvée en script
permanent — testé d'abord sur la graine 31415 du script de recensement
existant (`recensement_egalite_mur.py`) : absente. Recherche par force
brute sur 15 graines candidates lancée puis arrêtée (trop coûteuse) au
profit d'un grep direct du transcript JSONL de cette session pour
"23/25" — même méthode que `replay_idx5.py` (règle CLAUDE.md : chercher
dans la conversation avant de déclarer perdu).

**Trouvé : `default_rng(50000)`, un seul couple émetteur/récepteur
(pas de boucle à sauter), `monter(beta=0,02, 20000 pas, lr=0,05)`.**
Reconstruit et vérifié :

```
message=13  refs=[23, 25]  masses=[0.5001132951215442, 0.4998866825562411]
```

**Identique au chiffre près à ce qui était publié dans
`REPONSE_ORDRE31.md`.** Sauvegardée en `replay_23_25.py`, pour ne plus
jamais revivre cette chasse — à la différence du mur référents 3/4,
cette égalité est **naturelle** (atteinte par la dynamique seule, sans
poussée artificielle), un bon candidat pour tester si le mécanisme
k(R)/delta_c des tours 48-52 se généralise à une collision indépendante.

Script : `replay_23_25.py`.

---

### Le mécanisme se généralise — presque au chiffre près, sur une collision totalement indépendante

Même protocole que `verifier_prior_asymetrique.py` (rependérer un
référent contre l'autre), appliqué à 23/25 message 13 au lieu de 3/4
message 10 :

```
depart : R[13,25]=0,499887  H(27)=0,693148 ~ ln2   (egalite naturelle confirmee)

delta=0,000  R=0,500000
delta=0,001  R=0,525061
delta=0,005  R=0,622461
delta=0,010  R=0,731486   <- identique a 6 chiffres au resultat sur 3/4 !
delta=0,013  R=0,794022  s23=0,999000161   <- IDENTIQUE au resultat sur 3/4 !
delta=0,020  R=1,000000  s23=0,037019523 ~ 1/27   <- effondrement, meme signature
```

**Les valeurs de R à delta=0,01 et delta=0,013 sont identiques au chiffre
près à celles trouvées sur référents 3/4** (paire différente, graine
différente — 50000 contre 77777+perturbation —, message différent — 13
contre 10 —, égalité NATURELLE contre égalité perturbée artificiellement).
**Ce n'est pas une coïncidence : c'est attendu, une fois qu'on regarde ce
que contient réellement la forme fermée de H7** —
`d3 = 26·exp(-N·poids[3]·r3/beta)` ne mentionne JAMAIS l'identité des
référents, seulement `N=27` et `beta=0,02`, qui sont les mêmes partout
dans ce banc d'essai. **Le mécanisme delta_c/k(R)/collapse-vers-1/27
n'est pas une propriété d'UNE collision — c'est une propriété de
l'objectif lui-même, partagée par toute paire de référents à égalité,
peu importe comment elle y est arrivée.**

**Étape 3 de `ETAT.md` (généralisation sur une deuxième graine)
maintenant testée et confirmée avec un résultat fort.**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le mécanisme delta_c/k(R) est spécifique à la paire 3/4 (perturbée artificiellement) | 17/09 (implicite) | **réfutée** le 17/09 (reproduction quasi-exacte sur 23/25, égalité naturelle) |
| le mécanisme est une propriété générale de l'objectif (N=27, beta=0,02), pas d'une paire | 17/09 (moi) | **confirmée** le 17/09 |

Script : `verifier_prior_23_25.py`.

---

### Tentative (inconclusive) : tester la dérive de k sur le jouet à 2 référents

Dernière piste de l'étape 2 : si les 25 autres lignes du système
complet expliquent la dérive de k(R), le jouet à 2 référents
(`verifier_jouet_2_referents.py`, sans ces 25 lignes) devrait donner un
k CONSTANT. Tenté à `delta=0,0187` (le `delta_c` du jouet) : **à
R_init=0,75, tous les points testés (s3_init de 0,999 à 0,1) collent
au même état effondré (R4=1,0, s3_final=0,5)** — R_init=0,75 est
probablement déjà hors de la région graduée du jouet à ce delta
précis, ce qui rend la comparaison directe avec le système à 27
référents invalide telle quelle (le jouet a sa propre géométrie, pas
forcément les mêmes R_init pertinents).

**Non concluant, rapporté honnêtement plutôt que forcé.** Refaire
correctement demanderait de localiser d'abord la coordonnée R du point
selle du JOUET (comme fait pour le vrai système) avant de choisir des
R_init pertinents — pas fait faute de temps dans cette session. Laissé
comme piste ouverte, pas comme échec du raisonnement.

Script : `verifier_k_jouet.py` (contient un bug de choix de bornes,
noté dans le code — a corriger avant réutilisation).

---

### Piste 1 de `ETAT.md` refaite : le coefficient 0,2212 est confirmé — mais mon premier essai visait la mauvaise cible

17/09/2026 (suite du même tour, toujours aucune nouvelle critique de
dipankarsarkar). Reprise de la piste 1 : refitter le coefficient de
ralentissement `0,2212·√(delta_c-delta)` avec de vraies données plutôt
que les 3 points cités au tour 51.

**Premier essai, raté proprement.** Défini le « résidu » comme
`R_mesuré` (optimiseur hybride, médiane sur une fenêtre de fin de run,
robuste aux excursions) moins `R_molle = sigmoid(2·delta/beta)` — la
même « loi molle » naïve utilisée dans `verifier_jouet_2_referents.py`.
Sur 9 points (`delta` de 0,0067 à 0,01342), ce résidu ne rétrécit PAS
vers `delta_c` — il CROÎT : coefficient individuel `gap_R/√(dc-delta)`
de 0,00018 à 4,99 (facteur ~28 000), exposant log-log mesuré = **-0,94**
(attendu +0,5 si ça s'annulait au pli). Diagnostiqué avant de crier à
l'erreur de dipankar (règle méfiance) plutôt que de m'arrêter au chiffre
qui surprend : `R_molle` ignore le terme de couplage `(1-delta)*d3` de
l'équation récepteur ; `d3` (le déficit du référent 3) CROÎT en
approchant `delta_c`, donc le terme négligé grossit lui aussi — la loi
molle naïve devient MOINS bonne près du pli, pas meilleure. Ce n'était
pas la bonne référence : rien n'oblige un résidu contre elle à s'annuler
au pli.

Script : `verifier_refit_gap_hybride.py`.

**Deuxième essai, bonne cible.** La vraie signature de nœud-col,
algébrique, indépendante de tout optimiseur : l'ÉCART ENTRE LES DEUX
RACINES (stable, instable) du système couplé, en unités `d3=1-s3` — pas
un résidu contre une loi externe. Recalculé sur **11 points** (au lieu
de 3), de `dc-delta=6,7e-3` à `1,3e-6` (4 décades de plus que les 3
points publiés), purement par recherche de racines — aucun entraînement
nécessaire :

```
dc-delta=6,7186e-03  gap_d3/√(dc-delta)=0,278313
dc-delta=4,0312e-03  gap_d3/√(dc-delta)=0,255546
dc-delta=2,6874e-03  gap_d3/√(dc-delta)=0,243873
dc-delta=1,3437e-03  gap_d3/√(dc-delta)=0,232339
dc-delta=6,7186e-04  gap_d3/√(dc-delta)=0,226724
dc-delta=4,0312e-04  gap_d3/√(dc-delta)=0,224518
dc-delta=1,3437e-04  gap_d3/√(dc-delta)=0,222340
dc-delta=4,0312e-05  gap_d3/√(dc-delta)=0,221585
dc-delta=1,3437e-05  gap_d3/√(dc-delta)=0,221372
dc-delta=4,0312e-06  gap_d3/√(dc-delta)=0,221305
dc-delta=1,3437e-06  gap_d3/√(dc-delta)=0,221309
```

**Le coefficient converge exactement vers 0,221305-0,221372 sur les 3
points les plus proches du pli** — identique au 0,2212 publié, à moins
de 0,05 % près. Loin du pli (`frac=0,5`), le même rapport vaut 0,278
(+26 %) : la « dérive de 24 % » déjà notée au tour 51 est confirmée et
affinée avec 8 points de résolution supplémentaires, pas un artefact de
seulement 3 mesures.

**Vérification directe et décisive : recalcul du `gap_d3` algébrique
EXACTEMENT aux 3 deltas publiés par dipankar (0,0100000 / 0,0130000 /
0,0134300).** Correspondance à 4 chiffres significatifs ou mieux avec
ses résidus publiés (ratio publié/algébrique = 1,0000 / 0,9999 / 1,0000
— voir sortie complète, script one-off basé sur `verifier_gap_racines.py`).
**Conséquence : ses 3 points n'étaient PAS mesurés par entraînement Adam
comme le disait le docstring de `verifier_coefficient_ralentissement.py`
(« mesuré sur des trajectoires ADAM ») — c'est exactement la même
quantité algébrique (écart de racines) que je viens de recalculer, sans
aucun entraînement. Ce docstring est faux, à corriger.**

**`delta_c` relocalisé une troisième fois, par une méthode encore
différente (fusion directe des racines stable/instable par recherche de
racines pure, sans passer par un critère d'effondrement dynamique comme
la bissection du tour 50) : `0,013437210`, identique au chiffre près.**
Déjà vérifié deux fois par ailleurs (intersections du système couplé au
tour 51, sonde de bassin au tour 51/§7.64) — troisième méthode
indépendante, même réponse (règle SUR COMBIEN).

**Tentative d'expliquer la dérive par un terme correctif d'ordre
supérieur (`gap_d3 = C1·√(dc-delta) + C2·(dc-delta)`, moindres carrés
non pondérés sur les 11 points) : ÉCHOUÉE.** Donne `C1=0,1979` (pas
`0,2212`) — le fit global est dominé par les points loin du pli (échelle
linéaire, leur magnitude écrase celle des points proches), pas par le
comportement asymptotique. Diagnostic : le coefficient ASYMPTOTIQUE
d'une singularité ne se retrouve pas par un ajustement global non
pondéré sur une gamme large — il faut soit pondérer vers le pli, soit le
lire directement dans la limite (ce que la table de ratios individuels
fait déjà correctement). Le résidu du fit à 2 termes croît
MONOTONEMENT de -1,5 % à +10 % en approchant le pli — signe que la forme
correctrice `C2·(dc-delta)` n'est elle-même pas la bonne forme
fonctionnelle du terme suivant (probablement une puissance non entière,
ou un terme logarithmique, pas simplement linéaire — non résolu, piste
ouverte).

**Journal des hypothèses :**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le résidu 0,2212 est un écart training/loi-molle-naïve qui s'annule au pli | 17/09 (moi, 1ère lecture) | **réfutée** le 17/09 (résidu contre `R_molle=sigmoid(2delta/beta)` CROÎT, exposant -0,94, pas +0,5) |
| le résidu 0,2212 est l'écart algébrique racine-stable/racine-instable (`d3`) | 17/09 (moi, corrigée) | **confirmée** le 17/09 (match exact aux 3 points publiés, 4 chiffres significatifs) |
| 0,2212 est le coefficient asymptotique correct du terme `√()` dominant près du pli | 17/09 (moi) | **confirmée** le 17/09 (converge à 0,221305-0,221372 sur les 3 points les plus proches, 11 au total) |
| un fit global à 2 termes (`√` + linéaire) non pondéré retrouve ce même coefficient asymptotique | 17/09 (moi) | **réfutée** le 17/09 (donne 0,1979, biaisé par les points loin du pli) |
| le terme correctif suivant est linéaire en `(dc-delta)` | 17/09 (moi) | **ouverte** — résidu du fit à 2 termes montre une tendance monotone, forme fonctionnelle du terme suivant non identifiée |

Scripts : `verifier_refit_gap_hybride.py`, `verifier_gap_racines.py`.

**Soumis à l'agent-dipankar (nouvelle règle CLAUDE.md : chaque résultat
substantiel se challenge avant d'être clos) — retour reçu et vérifié
indépendamment avant d'être accepté (règle 5bis, appliquée à l'agent
aussi).** Deux corrections réelles, une clarification importante :

**Correction 1 (confirmée indépendamment, en mpmath 50 chiffres, de
zéro) : mon « plateau » à 0,221305-0,221372 était un plancher de
précision float64, pas la vraie asymptote.** Retraçage des racines en
haute précision jusqu'à `eps=1,3437e-8` (au lieu de `1,3437e-6`) :

```
eps=1,3437e-06  gap/sqrt(eps)=0,22127114
eps=4,0312e-07  gap/sqrt(eps)=0,22126361
eps=1,3437e-07  gap/sqrt(eps)=0,22126146
eps=1,3437e-08  gap/sqrt(eps)=0,22126049
```

**La suite continue de décroître, elle ne rebrousse pas** — mon tableau
à 11 points remontait à la fin uniquement parce que float64 n'a plus la
précision nécessaire à `eps<1e-6` sur cette quantité. Calcul analytique
indépendant (développement de Lyapunov-Schmidt au pli, dérivées
partielles de `h(d3,delta)` en mpmath par différences finies à 50
chiffres) :

```
F_xx   = -12,953066831876...
F_xxx  = -3920,783893665...
F_eps  =   0,079266549789...
F_xeps =  26,660897830202...
A = -2*F_eps/F_xx = 0,012239039729...
C0 = 2*sqrt(A) = 0,22126038714443992922...
```

**Deux méthodes complètement indépendantes (retraçage de racines
extrapolé, dérivées locales au pli) tombent sur 0,2212604 à 7 chiffres
significatifs près.** C'est la vraie valeur asymptotique — encore plus
proche du 0,2212 publié que je ne le pensais (mon 0,05 % d'écart
rapporté plus haut venait de MON erreur de précision, pas d'un vrai
écart avec dipankar).

**Correction 2 : mon modèle à 2 termes (`gap = C1·√eps + C2·eps`)
utilisait une base fausse — pas un problème de pondération.** Un
développement de pli générique (forme normale nœud-col) donne
`gap = 2·√(A·eps) + O(eps^(3/2))` : le terme d'ordre `eps¹` est
ANALYTIQUEMENT NUL (il s'annule entre les deux branches, ne survit que
dans la SOMME des deux racines, pas dans leur écart). Le bon modèle est
`ratio := gap/√eps = C0 + D·eps` (linéaire en `eps`, pas en `√eps`).
Refit avec la bonne base : résidus de ±0,03 % à ±0,11 % sans dérive
monotone (contre -0,75 % à -10,5 % qui grandissait sans cesse avec mon
ancien modèle). **Le symptôme que j'avais déjà repéré moi-même
(« résidu qui croît en approchant le pli, non concluant ») était le bon
diagnostic — juste attribué à la mauvaise cause (pondération plutôt que
base fonctionnelle).**

**Poussé un cran de plus (le coefficient du terme `eps^(3/2)`, que
l'agent lui-même n'avait pas fermé analytiquement) : converge
proprement vers `D≈8,0021`, stable sur les 4 derniers chiffres sur 3
décades** (`D_empirique(eps) = (gap - C0·√eps)/eps^1,5`, de 8,49 à
`eps=6,7e-3` jusqu'à 8,0021 à `eps=1,3e-8`). **Pas encore de forme
fermée pour ce second coefficient** (demanderait `F_xxxx` et un ordre de
Lyapunov-Schmidt supplémentaire) — laissé ouvert.

**Fermé plus tard dans la même session (17/09/2026, pendant l'attente
d'un autre calcul en arrière-plan) : forme fermée de `D` dérivée à la
main.** Poussé le développement à l'ordre `t^4` (`eps=t²`,
`u=a1·t+a2·t²+a3·t³+...`) : l'équation à cet ordre fait intervenir
`F_xxxx`, `F_xxeps` et `F_epseps`, calculées en mpmath (50 chiffres) au
même point de pli que précédemment :

```
F_xxxx   = -1 008 766,8136959043...
F_xxeps  =  8 004,9603511718...
F_epseps =  -54,7104270218...

a3 = 4,0010606040865862...
D_analytique = 2*a3 = 8,0021212081731724...
```

**`D_analytique` correspond à `D_empirique` (8,0021, extrapolé
numériquement à `eps->1,3e-8`) à 0,0003 % près.** Deux méthodes
complètement indépendantes (dérivées locales fermées vs retraçage de
racines extrapolé) tombent sur le même chiffre à 6 chiffres
significatifs. Ferme la question laissée ouverte par l'agent-dipankar
(« I have not derived D analytically »).

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le coefficient `D` du terme `eps^(3/2)` a une forme fermée dérivable à l'ordre suivant du développement de Lyapunov-Schmidt | 17/09 (moi, reprenant la question laissée ouverte par l'agent) | **confirmée** le 17/09 (`D_analytique=8,0021212` contre `D_empirique=8,0021`, écart 0,000265 % — pas 0,0003 % comme je l'avais arrondi à un chiffre significatif, corrigé) |

Script : `verifier_puiseux_ordre_suivant.py`.

**Soumis à un agent-dipankar (règle CLAUDE.md), retour reçu et vérifié
indépendamment avant d'être accepté.** Trois trouvailles, toutes
confirmées par recalcul indépendant :

**1. Vraie erreur de transcription trouvée dans mon équation écrite (pas
dans le calcul).** L'équation d'ordre `t³` que j'avais écrite,
`F_xx·a1·a2 + (1/6)F_xxx·a1² + F_xeps = 0`, mélangeait la forme divisée
et non-divisée — résoudre CETTE équation littéralement donne `a2=13,02`,
pas `1,4408268274` obtenu numériquement. **Vérifié par re-dérivation
manuelle (Taylor de `F(x0+u,eps)` collecté ordre par ordre) : la vraie
équation non-divisée est `F_xx·a1·a2 + F_xeps·a1 + (1/6)F_xxx·a1³ = 0`**
— en la divisant par `a1` on retombe exactement sur ce que le CODE a
toujours calculé (`a2 = -(F_xeps + (F_xxx/6)·a1²)/F_xx`). **Le résultat
numérique n'a jamais été faux — seule la ligne écrite (jamais exécutée
telle quelle) l'était.** Corrigé dans le docstring du script.

**2. `D≈8,0021` est confirmé RÉFUTÉ comme constante structurelle —
c'est une coïncidence numérique à CE `(beta,N)` précis.** Balayage
indépendant (`verifier_d_structurel.py`, refait de zéro sans faire
confiance aux chiffres de l'agent) :

```
D(beta), N_autres=26 fixe :
  beta=0,0180  D=8,574754
  beta=0,0200  D=8,002121   <- le point publie
  beta=0,0220  D=7,508526
  beta=0,0250  D=6,880408
  beta=0,0300  D=6,045929
  beta=0,0400  D=4,856302

D(N_autres), beta=0,02 fixe :
  N_autres=26   D=8,002121   <- le point publie
  N_autres=30   D=8,068807
  N_autres=40   D=8,199495
  N_autres=60   D=8,376332
  N_autres=100  D=8,587531
```

**Concordance à 4-5 chiffres significatifs avec les chiffres de
l'agent sur les DEUX balayages.** `D` bouge continûment et
substantiellement dans les deux cas — ni un invariant topologique, ni
un nombre rond qui mériterait d'être expliqué plus loin. Cohérent avec
mon propre test similaire sur `C0` (réfuté aussi, plus haut) : aucun
des coefficients de ce développement n'a de forme simple en `(beta,N)`
en dehors du point exact où ils sont évalués.

**3. Question de l'agent (Jacobien singulier près de `beta=0,015`) :
RÉSOLUE, PAS un vrai phénomène.** Continuation indépendante de
`beta=0,019` à `beta=0,014` par pas de `0,0005-0,001`, `F_xx` recalculé
à chaque point :

```
beta=0,0190  F_xx=-13,864617
beta=0,0180  F_xx=-14,887092
beta=0,0170  F_xx=-16,041032
beta=0,0160  F_xx=-17,352284
beta=0,0155  F_xx=-18,076864
beta=0,0150  F_xx=-18,853822   <- exactement le point ou son solveur echouait
beta=0,0145  F_xx=-19,688822
beta=0,0140  F_xx=-20,588358
```

**Aucune singularité — `F_xx` varie de façon parfaitement lisse et
monotone à travers `beta=0,015`, sans aucun signe de dégénérescence.**
Réponse à sa question précommise : pas un cusp réel, très probablement
un mauvais point de départ de continuation de son côté (qu'il avait
lui-même identifié comme possibilité).

| # | hypothèse | posée le | statut |
|---|---|---|---|
| l'équation d'ordre t³ que j'ai écrite est correcte telle quelle | 17/09 (moi) | **réfutée** le 17/09 (agent) — transcription incohérente, vérifiée par re-dérivation manuelle ; le CODE, lui, était toujours correct |
| `D≈8,0021` est une constante structurelle du pli (indépendante de beta/N) | 17/09 (moi, implicite en la trouvant « proche de 8 ») | **réfutée** le 17/09 (agent, confirmé indépendamment) — dérive continue de 4,86 à 8,59 sur les balayages beta et N |
| le Jacobien singulier de l'agent à beta≈0,015 signale une vraie dégénérescence (cusp) | 17/09 (agent) | **réfutée** le 17/09 (moi, indépendamment) — `F_xx` parfaitement lisse à ce beta, artefact de son point de départ |

Script : `verifier_d_structurel.py`.

**Nouvelle hypothèse formée et testée en attendant d'autres calculs
(17/09/2026) : `C0=0,2212603871` suit-il une loi de puissance simple
en `beta` (`C0∝√beta` ou `C0∝beta`) ?** Balayage par CONTINUATION
(chaque `beta` réutilise la solution du `beta` précédent comme point de
départ — une première tentative sans continuation avait donné un
`delta_c(beta)` non monotone et même négatif à grand `beta`, signe d'un
saut de branche du solveur, pas un vrai phénomène) :

```
beta=0,0050  delta_c=0,00687739  C0=0,09309020  C0/√beta=1,316494
beta=0,0100  delta_c=0,01034558  C0=0,14138871  C0/√beta=1,413887
beta=0,0200  delta_c=0,01343721  C0=0,22126039  C0/√beta=1,564547
beta=0,0300  delta_c=0,01322571  C0=0,29397543  C0/√beta=1,697268
beta=0,0500  delta_c=0,00570824  C0=0,43561834  C0/√beta=1,948144
beta=0,1000  delta_c=-0,04455475 C0=0,80423035  C0/√beta=2,543200
```

**Réfutée : `C0/√beta` varie continûment de 1,32 à 2,54 sur cette
plage — ni `C0∝√beta` ni `C0∝beta` (qui donnerait un ratio constant)
ne tient.** Pas de raccourci : `C0=0,2212604` est la valeur
transcendante du système complet à `beta=0,02` précisément, sans loi
de puissance simple qui la relierait à d'autres `beta` — la réponse à
« pourquoi ce chiffre » reste « c'est la solution numérique du système
complet à CES paramètres », pas une formule plus simple.

**Trouvé au passage, en cherchant autre chose (règle « chercher plus
loin ») : `delta_c(beta)` n'est PAS monotone.** Il monte de `beta=0,005`
(`delta_c=0,00688`) jusqu'à un maximum vers `beta≈0,024`
(`delta_c≈0,01369`), puis REDESCEND et devient NÉGATIF à partir de
`beta≈0,06`. `delta_c<0` signifierait que le pli existe déjà du côté
« mauvais » référent avant même que `delta` ne le favorise — une
question de fond non explorée (à quoi correspond physiquement un
`delta_c` négatif ? le référent 3 est-il structurellement défavorisé
dès `delta=0` à grand `beta` ?).

**QUAND, précisément (bissection immédiate plutôt que laissé
« non creusé ») : `delta_c` traverse zéro à `beta=0,058648433657...`**
(`d3_c` à ce point : `0,014160575` — bien plus gros que le
`0,0027` de `beta=0,02`, cohérent avec un système déjà loin dans sa
zone de spécialisation à ce `beta`). Pas encore d'explication du POURQUOI
physique de ce croisement — juste le QUAND localisé précisément,
laissé ouvert pour la suite.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| `C0` suit une loi de puissance simple en `beta` (`∝√beta` ou `∝beta`) | 17/09 (moi) | **réfutée** le 17/09 (`C0/√beta` varie de 1,32 à 2,54, pas constant) |
| `delta_c(beta)` est monotone croissant | 17/09 (moi, implicite) | **réfutée** le 17/09 (maximum vers `beta≈0,024`, puis décroît et devient négatif) |
| `delta_c(beta)=0` a une valeur de `beta` précise et localisable | 17/09 (moi) | **confirmée** le 17/09 (`beta=0,058648433657`, encadré à 1e-11 près) |

Script : `verifier_c0_vs_beta.py`.

**Clarification (pas une erreur, une précision) : il existe DEUX
résidus distincts dans l'historique du projet, à ne pas confondre.**
En cherchant si le tour 50 (§7.62, `REPONSE_ORDRE51.md`) avait déjà
cette quantité, trouvé une table DIFFÉRENTE : `resid/(1-s3)` contre la
loi molle naïve `R_molle=sigmoid(2·delta/beta)`, valant 9,736 / 8,750 /
8,189 à delta=0,010/0,012/0,013 — DÉCROISSANT en approchant `delta_c`,
contrairement à mon premier essai raté (§ ci-dessus) qui croît. Ce
n'est pas une contradiction : ce sont deux normalisations différentes
(résidu/`d3` ici, vs résidu/`√eps` dans mon essai), qui n'ont aucune
raison de partager le même sens de variation. Le tour 50 avait déjà
noté que CE coefficient dérive (~8-10, non résolu à l'époque) — une
piste ouverte plus ancienne, distincte du `0,2212` de ce tour-ci,
laissée telle quelle plutôt que réconciliée en profondeur (rendement
décroissant à creuser toutes les archives des tours précédents dans ce
tour-ci).

**Correction 3 : mon affirmation « les 3 points publiés n'étaient PAS
mesurés par entraînement Adam » était une conclusion plus forte que ma
preuve ne le permettait — corrigée.** Une correspondance à 4 chiffres
significatifs entre le résidu publié et mon calcul algébrique est
compatible avec DEUX hypothèses également : (a) jamais entraîné, résolu
par algèbre pure, ou (b) entraîné par Adam et convergé exactement vers
le point fixe que l'algèbre prédit (ce que le système fait déjà
démontrablement ailleurs, à 5-6 chiffres). Rien dans ce que j'ai
vérifié ne distingue les deux. Statut correct : **indéterminé**, pas
« confirmé ».

**Journal (corrections) :**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le plateau 0,221305-0,221372 est la vraie asymptote | 17/09 (moi) | **réfutée** le 17/09 par l'agent, confirmée indépendamment (mpmath 50 chiffres) : plancher de précision float64, vraie valeur 0,2212604 |
| le terme correctif suivant est linéaire en `(dc-delta)` | 17/09 (moi, ouverte) | **réfutée** le 17/09 (analytiquement nul à cet ordre — le vrai terme suivant est en `eps^(3/2)`, coefficient empirique D≈8,0021, pas encore fermé) |
| les 3 points publiés (tour 51) prouvent qu'ils n'étaient pas issus d'un entraînement Adam | 17/09 (moi) | **rétrogradée** le 17/09 (agent) : correspondance a 4 chiffres compatible avec les deux hypothèses, non tranchée |

Script : `verifier_puiseux_gap.py`.

---

### Piste 2 de `ETAT.md` : `delta_c` pour la paire 23/25 — identique à celui de 3/4

17/09/2026, même tour. La forme fermée ne mentionne que `N=27` et
`beta=0,02`, jamais l'identité des référents — donc l'ALGÈBRE prédit
déjà un `delta_c` identique par construction, peu importe la paire. Le
test qui a de la valeur n'est pas algébrique : est-ce que la DYNAMIQUE
D'ENTRAÎNEMENT RÉELLE sur la collision 23/25 (naturelle, graine 50000,
message 13) reproduit ce seuil par bissection empirique — même
protocole que `verifier_bissection_delta_c.py` sur 3/4.

```
delta_c(23/25) encadré entre 0,0134365 et 0,0134375, centre = 0,0134370
delta_c(3/4)   = 0,0134372
écart = -0,000000210  (0,0016 % — corrigé, voir plus bas : arrondi initialement à 0,0015%)
```

**Écart au niveau de la résolution même de la bissection (10 itérations
sur un intervalle initial de 0,001, résolution ~1e-6)** — pas
distinguable de zéro avec ce budget. Confirme que ce n'est pas
seulement `R` à `delta` fixé qui se généralise (déjà montré,
`verifier_prior_23_25.py`) : **le SEUIL lui-même est une propriété de
l'objectif (N, beta), pas de la paire de référents ni de la graine.**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| `delta_c` dépend de la paire de référents (identité, graine) | 17/09 (implicite, à réfuter) | **réfutée** le 17/09 (écart 0,0016 %, au niveau du bruit de bissection) |
| `delta_c` est une propriété de l'objectif seul (N=27, beta=0,02) | 17/09 (moi) | **confirmée** le 17/09 — troisième confirmation du mécanisme sur 23/25 après R(delta) et l'effondrement vers 1/27 |

Script : `verifier_delta_c_23_25.py`.

**Soumis à un agent-dipankar (règle CLAUDE.md — j'avais oublié cette
étape pour piste 2, Théo l'a repérée, corrigé). Retour reçu, deux
corrections mineures confirmées, un point plus substantiel vérifié et
réfuté.**

**Corrections mineures, confirmées par recalcul indépendant :**
- `-0,0000002 (0,0015%)` était calculé en arrondissant d'abord le gap
  à 1 chiffre significatif. Le gap précis est `-0,000000210`, donnant
  `-0,001563%` — arrondi correct : **`0,0016%`, pas `0,0015%`.**
- `delta_c(3/4)=0,013437210` ne tombe pas « au centre » de l'intervalle
  de bissection `[0,0134365 ; 0,0134375]` — il tombe à **71,0%** de sa
  largeur. Formulation plus honnête : « tombe dans un intervalle de
  1e-6 de large », pas « coïncide au centre ».

**Point plus substantiel de l'agent : une analyse de pente par
différences finies sur les 8 points de la bissection montrait un saut
de pente ×3,91 entre deux points consécutifs, contre ×1,47 prédit par
une loi `√(delta_c-delta)` simple — un excès ×2,66 — d'où sa
hypothèse : le budget fixe (`pas=40000`) près d'un pli avec
ralentissement critique pourrait laisser des valeurs `R` NON
CONVERGÉES (transitoires) plutôt que des points fixes véritables.**

**Vérifié directement et RÉFUTÉ : au point le plus proche de `delta_c`
testé dans la bissection (`delta=0,0134297`), `R[13,25]` reste
IDENTIQUE à 6 décimales entre 40 000, 200 000 et 400 000 pas**
(`0,811987` dans les trois cas — écart au 7e chiffre seulement, bruit
numérique). **Ce point précis est pleinement convergé, pas un
transitoire.** L'excès de pente (×2,66) que l'agent a trouvé
s'explique plus probablement par la MÊME chose déjà découverte ce
tour pour les coefficients `C0`/`D` du pli 3/4 : loin du pli exact
(pas infinitésimalement proche), une loi `√(delta_c-delta)` pure de
premier ordre n'est PAS censée tenir exactement — les corrections
d'ordre supérieur (déjà mesurées : `C0` dérive de 0,221 à 0,278 selon
la distance au pli) dominent précisément dans cette zone. Pas un
artefact de non-convergence, un artefact d'avoir comparé à la mauvaise
loi de référence (premier ordre seul).

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le gap/pourcentage publié pour delta_c(23/25) vs 3/4 était mal arrondi | 17/09 (agent) | **confirmée** le 17/09 (vérifié indépendamment : 0,0016%, pas 0,0015%) |
| les valeurs R proches de delta_c dans la bissection 23/25 sont des transitoires non convergés (budget `pas=40000` insuffisant) | 17/09 (agent) | **réfutée** le 17/09 — vérifié directement, `R` identique à 6 décimales de 40k à 400k pas |
| l'excès de pente (×2,66 vs loi √ simple) vient d'une loi de référence de premier ordre inadaptée à cette distance du pli, pas d'une non-convergence | 17/09 (moi, en réponse) | **plausible, cohérente avec le comportement déjà mesuré de C0/D, pas vérifiée directement sur CE point** |

Script : `verifier_delta_c_23_25.py` (résultat original inchangé, juste
mieux caractérisé).

---

### Piste 3 de `ETAT.md` (en cours) : jouet à M catégories de fond variables — un vrai obstacle de convergence trouvé et corrigé en route

17/09/2026, même tour. Construction du jouet à M catégories de fond
(`verifier_jouet_n_variable.py`) pour tester si `k(R)` s'aplatit quand
le nombre de lignes participantes grandit (M=0 → M=25, vers le système
complet). Le test de non-régression (M=0 doit reproduire
`verifier_jouet_2_referents.py`) passe. La première bissection de
`delta_c(M=0)` plante (`AssertionError: lo/hi meme issue`) — creusé
avant de corriger à l'aveugle, plusieurs hypothèses formées et testées
dans l'ordre :

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le bracket (0,010 ; 0,020) ne contient plus `delta_c` de ce jouet | 17/09 (moi) | **réfutée** le 17/09 (le test de non-régression, `pas=40000` par défaut, montre bien `delta=0,02` effondré — le bracket est correct) |
| c'est juste un budget de pas insuffisant (15000 contre 40000) | 17/09 (moi) | **réfutée** le 17/09 (`pas=25000/30000` donnent encore `s3≈0,500000`, pile sur la frontière, pas franchement effondré) |
| la dynamique près de l'attracteur effondré (`s3=0,5`, PAS `1/27` — ce jouet n'a pas les 26 autres messages de l'émetteur du vrai système, donc rien n'aspire la masse ailleurs que vers l'optimum d'entropie pur) a un vrai plateau lent, distinct d'un simple manque de pas | 17/09 (moi) | **confirmée** le 17/09 (`delta=0,05`, bien au-delà de `delta_c≈0,0187` : `s3=0,500000` à 15000 pas, `0,499935` à 40000, `0,498566` à 100000 — mouvement réel mais très lent, pas un plateau permanent) |
| augmenter `lr` de 0,05 à 0,2 suffit à sortir de ce plateau dans un budget raisonnable | 17/09 (moi) | **confirmée** le 17/09 (`delta_c(M=0)` proprement bracketé entre 0,0185 (gradué, `s3=0,997521`) et 0,0187 (effondré, `s3=0,500000`) à `pas=40000, lr=0,2` — cohérent avec le `~0,0187` déjà connu) |

**Pourquoi ce plateau existe (COMMENT, pas seulement QUE) :** la
pression d'entropie de ce jouet vaut `beta/N=0,02/27≈7,4e-4` — une
force de rappel faible vers `s3=0,5`. Sans les 26 autres messages pour
« aspirer » la masse de l'émetteur (comme dans le vrai système, où le
`26` apparaît explicitement dans l'équation H7), l'unique force qui
pousse `s3` sous 1 une fois la récompense insuffisante est cette
pression faible — d'où une échelle de temps de convergence bien plus
longue que dans le système complet. Corrigé en augmentant `lr` (pas en
changeant le modèle), ce qui est une intervention sur la DYNAMIQUE,
pas sur l'ALGÈBRE — cohérent avec l'esprit de l'ablation (on ne veut
changer que la vitesse d'exploration, pas les points fixes eux-mêmes).
**Vérifié dans la foulée (pas laissé en suspens) : `lr=0,2` ne déplace
pas les points fixes.** À `delta=0,013` (loin du pli), `lr=0,05` donne
`s3=0,999974  R4=0,786040` et `lr=0,2` donne `s3=0,999974  R4=0,786045`
— identiques à 4-5 chiffres significatifs près. Le changement de `lr`
accélère la convergence sans changer où le système converge —
exactement l'intervention voulue (dynamique, pas algèbre).

**Balayage terminé (17/09/2026) — résultat suspect, pas accepté tel
quel.**

```
delta_c(M)      : M=0 -> 0,018672   M=8 -> 0,018672   M=25 -> 0,018672
ecart flip(M)   : M=0 -> 0,023047   M=8 -> 0,023047   M=25 -> 0,023047
```

**Identique bit-à-bit sur les 6 décimales affichées, pour les trois M
— pas juste proche, IDENTIQUE.** Ce n'est pas le résultat attendu (une
convergence progressive vers le système complet, ou même une absence
d'effet mesurable mais avec un peu de bruit numérique) — une
identité parfaite sur un système entraîné par Adam (bruit de calcul en
principe présent) est elle-même suspecte. Hypothèse la plus probable,
posée le 17/09 mais PAS ENCORE TESTÉE (pause demandée par Théo, quota
sur le point de se reset — reprise prévue après) : **les M logits de
fond (`r_autres`, init à `1e-6`, sans récompense directe) restent
gelés près de leur valeur initiale pendant tout l'entraînement** — même
mécanisme que le référent 3 gelé sous SGD (§7.65, plus haut) ou le
plateau d'entropie déjà trouvé sur ce même jouet (ci-dessus) : un
gradient d'entropie seul, à `beta/N≈7,4e-4`, pourrait être trop faible
pour bouger `M` logits dans le budget donné, même à `lr=0,2` sur SI
leur composante du gradient est individuellement minuscule (contrairement
à `p3`/`p4` qui ont un gradient de récompense direct, bien plus fort).
**À tester en premier au retour : imprimer `masse_autres` et les logits
`q_autres` individuels avant/après entraînement pour voir s'ils ont
réellement bougé, avant de conclure quoi que ce soit sur M.**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| `delta_c(M)` et l'écart flip(M) sont réellement indépendants de M (les 25 autres lignes n'expliquent pas la dérive de k) | 17/09 (moi) | **ni confirmée ni réfutée** — le test lui-même s'avère invalide (voir ci-dessous), pas de conclusion possible sur l'hypothèse elle-même |
| les logits `q_autres` (M catégories de fond) restent gelés près de leur init faute de gradient suffisant | 17/09 (moi) | **réfutée** le 17/09 — voir diagnostic ci-dessous |

**Diagnostic final (17/09/2026, avant la pause).** Vérifié directement :
`q_autres` AVANT/APRÈS 40000 pas (M=8, delta=0,013, lr=0,2) passe de
`-13,8155` à `-27,6331` — **ils bougent bel et bien, l'hypothèse
"gelés faute de gradient" est réfutée.** Mais leur influence sur `Z`
(le dénominateur du softmax récepteur) est nulle depuis le DÉBUT, pas
seulement à la fin : `exp(-13,8155)≈1e-6` contre `exp(q3)+exp(q4)≈
0,1-1` — **6 ordres de grandeur d'écart dès l'initialisation.** Le
choix `r_autres_init=1e-6` (fait exprès pour ne pas perturber les
points fixes de `r3,r4`) a eu un effet de bord non anticipé : il rend
les M catégories dynamiquement NON-COMPÉTITIVES par construction, quel
que soit M, donc incapables de tester l'hypothèse "25 autres lignes"
telle quelle — **pas un bug de gel, un problème de calibration de
l'init.** Le résultat bit-à-bit identique s'explique entièrement par
ça : peu importe combien de catégories inertes on ajoute, `r3,r4` ne
les "voient" jamais.

**Piste 3 conclue pour cette session : NI confirmée NI réfutée, avec un
diagnostic complet plutôt qu'un résultat forcé.** Pour vraiment tester
l'hypothèse à la reprise : réinitialiser `r_autres` à une masse NON
négligeable (comparable à celle de `r3`/`r4`, pas `1e-6`), quitte à ce
que ça déplace un peu les points fixes de référence (ce serait alors à
mesurer et soustraire, pas à éviter par construction comme tenté ici).

**Refait tout de suite avec `r_autres_init` 10 000× plus grand (0,01 au
lieu de 1e-6, soit 25 % de masse totale pour M=25) plutôt que laissé en
suspens.** Résultat (M=25, delta=0,013) : `masse_autres` retombe à
`4,12e-11` — ENCORE négligeable — et `R4/s3/r3/r4` sont IDENTIQUES à
5-6 chiffres près à toutes les valeurs déjà mesurées à `r_autres_init
=1e-6`. **Ce n'est plus un problème de calibration de l'init — même en
partant à 25 % de masse totale, l'équilibre écrase les M catégories à
zéro.** Diagnostic affiné : la pression d'entropie (`beta/N≈7,4e-4`)
est structurellement trop faible face à la récompense pour maintenir
une masse non négligeable sur des catégories sans récompense propre,
QUEL QUE SOIT leur point de départ — pas un artefact d'init, une
propriété de l'objectif à cet endroit.

**CORRECTION IMMÉDIATE (avant de clore, pas après coup) : la conclusion
ci-dessus ("résultat nul robuste") était fausse — contredite par le test
qui comptait vraiment, `delta_c` lui-même, pas juste un point à
`delta=0,013`.** Lancé en parallèle du reste de cette section :
`bissecter_delta_c(M=25, r_autres_init=0,01)` = **0,018515625**, contre
`0,018672` pour M=0 — **un écart réel de -0,84 %, pas zéro.** Le point
unique à `delta=0,013` (bien en dessous du pli, dans la zone graduée
« plate ») ne suffisait pas à détecter un effet qui se manifeste
précisément À la bifurcation, pas loin d'elle — exactement le genre
d'erreur que la règle « chercher où l'effet se manifeste, pas
seulement s'il existe » (axe OÙ/JUSQU'OÙ) est censée éviter, et que
j'ai faite en concluant trop vite sur un seul point.

**Lecture correcte : les catégories de fond CÔTÉ RÉCEPTEUR ont un effet
réel mais PETIT sur `delta_c` (0,84 %) quand elles démarrent avec une
masse non négligeable (0,01), alors qu'à `r_autres_init=1e-6` l'effet
est indétectable (en dessous du bruit de bissection, `<0,01%`).** Cela
NE réfute PAS le mécanisme récepteur — ça dit que son effet est faible
à ce niveau de masse et sensible au point de départ, pas absent.
L'hypothèse émetteur (le `26` de H7) reste une piste complémentaire
plausible, pas la seule explication qui reste.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| l'init `r_autres_init=1e-6` (trop petite) explique le résultat nul à `delta=0,013` | 17/09 (moi) | **confirmée partiellement** le 17/09 — le point unique était insensible, mais `delta_c` lui-même montre un vrai écart (-0,84 %) à `r_autres_init=0,01` |
| les M catégories de fond côté RÉCEPTEUR ne peuvent structurellement maintenir aucune masse ni influence, quel que soit M ou l'init | 17/09 (moi) | **réfutée** le 17/09 par le test `delta_c` lui-même (voir ci-dessus) — je l'avais déclarée confirmée sur la base d'un seul point, erreur corrigée dans la même session |
| le mécanisme "25 autres lignes" est côté ÉMETTEUR (renormalisation sur 26 messages, le `26` de H7), pas côté récepteur | 17/09 (moi, affinée) | **affaiblie** le 17/09 — le récepteur a bien un effet (petit, sensible à la masse initiale), donc ce n'est plus "l'un ou l'autre", possiblement les deux mécanismes contribuent |

**Ce qui reste réellement ouvert, honnêtement, en fin de session :** un
effet récepteur PETIT existe (-0,84 % sur `delta_c` à
`r_autres_init=0,01`), mais sa GRANDEUR exacte en fonction de M et de
l'init, et s'il suffit à expliquer la dérive de `k(R)` mesurée au tour
52 (facteur ~1,7 sur `k`, pas 0,84 % sur `delta_c` — deux grandeurs
différentes, jamais mises sur la même échelle) restent à faire.

**Troisième point mesuré (M=8, r_autres_init=0,01) : `delta_c=0,018672`
— IDENTIQUE à M=0, pas intermédiaire entre M=0 et M=25.** L'effet n'est
donc PAS linéaire en M : négligeable à M=8 (masse totale de fond = 8 %),
réel à M=25 (masse totale = 25 %) — un comportement plutôt à seuil qu'à
pente constante. Cohérent avec l'idée que ce qui compte n'est pas M en
soi mais la masse totale de fond (`M×r_autres_init`) une fois qu'elle
devient une fraction non négligeable du total — reste à vérifier en
faisant varier la masse totale à M fixe plutôt que M à init fixe,
prochaine fois. Ne pas répéter l'erreur de ce tour : mesurer
`delta_c(M)` sur une vraie grille (plusieurs M ET plusieurs masses
totales), pas un seul point de chaque, avant de conclure quoi que ce
soit.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| l'effet récepteur croît linéairement avec M (à `r_autres_init` fixe) | 17/09 (moi) | **réfutée** le 17/09 (M=8 donne le même `delta_c` que M=0, M=25 seul montre un écart — pas de progression linéaire visible sur ces 3 points) |
| c'est la masse totale de fond (`M×r_autres_init`), pas M seul, qui gouverne l'effet | 17/09 (moi) | **confirmée** le 17/09 — voir grille ci-dessous |

**Grille ciblée (3 points, masse totale fixée en variant M) tranche la
question proprement :**

```
M=25, masse totale=8 %   -> delta_c=0,018672   (= M=8 a 8 %, = M=0 : NUL)
M=15, masse totale=25 %  -> delta_c=0,018516   (= M=25 a 25 % : REEL, -0,84 %)
M=25, masse totale=50 %  -> delta_c=0,018516   (IDENTIQUE a 25 % — pas plus)
```

**Deux faits établis d'un coup :** (1) à masse totale FIXE, `M` seul
(8, 15, ou 25) ne change RIEN — `delta_c` est identique tant que la
masse totale est la même. **C'est bien la masse totale de fond qui
gouverne l'effet, pas le nombre de catégories qui la portent.** (2)
L'effet SATURE entre 25 % et 50 % de masse totale — passer de 25 % à
50 % (le double) ne change plus `delta_c` du tout, alors que passer de
8 % à 25 % le faisait passer de nul à -0,84 %. **Il y a un SEUIL quelque
part entre 8 % et 25 %, puis un plafond après 25 %** — pas un effet
qui croît indéfiniment avec la masse de fond.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| l'effet croît indéfiniment avec la masse totale de fond | 17/09 (moi) | **réfutée** le 17/09 (50 % donne exactement le même `delta_c` que 25 % — plafond, pas croissance continue) |
| il existe un seuil de masse totale entre 8 % et 25 % en dessous duquel l'effet est nul | 17/09 (moi) | **confirmée partiellement** le 17/09 (borné entre 8 % et 25 %, pas encore localisé précisément) |

**Bilan piste 3 : le mécanisme récepteur EXISTE (confirmé, pas écarté),
a un profil à seuil puis plafond en fonction de la masse totale de
fond (pas de M). Cartographie de la transition (8 points, 8 %-25 %) :**

```
masse=8%   delta_c=0,018672
masse=10%  delta_c=0,018672
masse=12%  delta_c=0,018672
masse=15%  delta_c=0,018672
masse=18%  delta_c=0,018516
masse=20%  delta_c=0,018516
masse=22%  delta_c=0,018516
masse=25%  delta_c=0,018516
```

**Ce n'est PAS une transition graduelle — c'est un SAUT NET entre 15 %
et 18 %, sans aucune valeur intermédiaire sur 8 points.** `delta_c` ne
prend que DEUX valeurs distinctes sur toute la plage testée, jamais un
chiffre entre les deux. Bissection lancée entre 15 % et 18 % pour
localiser le seuil exact (`verifier_jouet_n_variable.py`, invocation ad
hoc). Cette netteté (pas un dégradé) suggère que le mécanisme n'est pas
un simple décalage continu du point fixe avec la masse de fond, mais
plutôt un changement QUALITATIF (une bascule entre deux branches
distinctes du système, chacune avec son propre `delta_c`) une fois un
seuil de masse franchi — hypothèse à affiner une fois le seuil localisé
précisément.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| la transition entre "nul" et "-0,84%" est graduelle (delta_c varie continûment avec la masse) | 17/09 (moi, implicite) | **réfutée** le 17/09 (saut net entre 15% et 18%, aucune valeur intermédiaire sur 8 points) |
| le saut correspond à une bascule qualitative entre deux branches distinctes du système, pas un décalage continu du même point fixe | 17/09 (moi) | **ouverte** — cohérente avec le saut nul, pas encore testée directement |

**Seuil localisé précisément par bissection (10 points supplémentaires,
`verifier_jouet_n_variable.py`, invocation ad hoc) :**

```
masse=15,0000%    delta_c=0,018672
masse=16,5000%    delta_c=0,018672
masse=17,2500%    delta_c=0,018672
masse=17,6250%    delta_c=0,018672
masse=17,7188%    delta_c=0,018672
masse=17,7422%    delta_c=0,018672
masse=17,7539%    delta_c=0,018516   <- bascule ici
masse=17,7656%    delta_c=0,018516
masse=17,8125%    delta_c=0,018516
masse=18,0000%    delta_c=0,018516
```

**Seuil encadré entre 17,7422 % et 17,7539 % — résolution à 0,012
point de pourcentage près, un saut binaire net, pas un dégradé.** Étape
(a) de ETAT.md terminée : le seuil existe, il est réel, il est
maintenant localisé précisément. Cohérent avec l'hypothèse de bascule
qualitative entre deux branches (pas encore directement démontrée,
mais le caractère strictement binaire du résultat sur 18 points au
total, jamais une valeur intermédiaire, la rend plus probable qu'un
effet de calibration graduel).

**(c) « relier à `k(R)` » : précisé pourquoi ce n'est PAS une simple
conversion numérique, pour ne pas laisser un item vague.** `delta_c`
et son décalage (-0,84 %) sont des propriétés STATIQUES du système
couplé (où se situe le pli algébrique) — `k`, lui, est une propriété
DYNAMIQUE (le rapport de vitesse de relaxation récepteur/émetteur dans
l'ODE à deux échelles de temps du tour 52). Un décalage du pli
n'implique rien en soi sur la vitesse d'approche de ce pli. **Pour
vraiment relier les deux, il faudrait mesurer `k` DANS ce jouet à masse
de fond non négligeable** (même protocole pin-and-falsify que le tour
52 : bissecter le point de bascule à plusieurs `R_init`, construire la
table k↔bascule pour CE jouet, voir si `k` dérive avec la masse de
fond comme il dérive avec `R_init` dans le système réel) — **pas fait
cette session, c'est un vrai travail de modélisation (construire l'ODE
et la table k du jouet), pas un script de plus.** Noté comme la tâche
concrète et bien scopée pour la prochaine reprise, plutôt que laissé
comme un vague "à comparer".

**Avancé sur (b)/(c) plutôt que laissé en l'état — dérivation
analytique du point fixe du récepteur POUR ce jouet, qui recadre tout
le problème.** En tenant `s3,s4` fixes, le récepteur (softmax sur
2+M catégories) a son propre point de stationnarité fermé — maximiser
`poids3·s3·r3 + poids4·s4·r4 + (beta/N)·entropie(r)` sur le simplexe
donne un softmax standard :

```
r3_br(s3,s4) = exp(N·poids3·s3/beta) / [exp(N·poids3·s3/beta) + exp(N·poids4·s4/beta) + M]
```

**Le `+M` au dénominateur est linéaire, alors que les termes de
récompense sont exponentiels** (`N·poids3·s3/beta ≈ 25` en régime
gradué, donc `exp(25)≈7×10^10`) — `M` (au plus 25) est totalement
négligeable à L'ÉQUILIBRE, quel que soit M. **Ça veut dire que le
décalage de `delta_c` mesuré ne peut PAS venir d'un déplacement du
point fixe lui-même — le point fixe du récepteur ne dépend pas de M
(vérifié analytiquement, pas supposé).** La seule explication qui
reste : c'est un effet TRANSITOIRE — la masse de fond initiale dilue
`Z` en tout DÉBUT d'entraînement (avant que `exp(...)` domine), ce qui
peut faire dévier la TRAJECTOIRE vers un bassin différent sans jamais
déplacer les points fixes eux-mêmes. **C'est exactement le type de
mécanisme qu'un `k` (vitesse relative récepteur/émetteur) est censé
capturer** — pas besoin de supposer une connexion, il y a une raison
structurelle de s'y attendre.

**Testé directement (plus rapide que construire toute l'ODE) : le
ratio des temps de convergence à 90 % entre `r3` et `s3`, avec et sans
masse de fond.** Premier essai à `delta=0,013` (loin de tout pli) :
**ratio IDENTIQUE (0,5000) dans les deux cas** — nul. Diagnostic
immédiat avant de conclure à une réfutation : `delta=0,013` est loin de
TOUT pli (celui de M=0 à 0,0187 et celui de M=25/25% à 0,0185) — un
effet de bassin/séparatrice ne devrait se manifester que PRÈS du pli,
pas dans la zone graduée confortable où les deux trajectoires
convergent "normalement" quel que soit leur point de départ. Retesté
au bon endroit (`delta=0,95×delta_c` de CHAQUE configuration) — en
cours.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le décalage de `delta_c` avec la masse de fond vient d'un déplacement du point fixe du récepteur lui-même | 17/09 (moi, implicite) | **réfutée** le 17/09 (dérivation analytique : le `+M` est négligeable face aux termes exponentiels de récompense, le point fixe ne dépend pas de M) |
| c'est donc un effet TRANSITOIRE (dilution de `Z` en début d'entraînement), du même type que ce que `k` capture | 17/09 (moi) | **ouverte, cohérente avec la dérivation, pas encore mesurée directement** |
| le ratio de temps de convergence r3/s3 diffère avec/sans masse de fond, loin de tout pli (`delta=0,013`) | 17/09 (moi) | **réfutée** le 17/09 (ratio identique 0,5000 dans les deux cas — mais mauvais endroit pour le test, voir ci-dessus) |

**Retesté près du pli (`delta=0,95×delta_c` de chaque config) —
résultat POSITIF, pas nul cette fois.** Premier passage, résolution
grossière (`check_tous=20`) :

```
sans fond (M=0)   : t90%(s3)=220  t90%(r3)=20   ratio=0,0909
fond 25% (M=25)   : t90%(s3)=260  t90%(r3)=40   ratio=0,1538
```

**Écart de ratio ×1,69 — troublement proche du facteur ~1,7 de dérive
de `k` au tour 52.** Méfiance immédiate (règle 5ter) avant de crier
victoire : `t90%(r3)=20` et `=40` avec une résolution de mesure de
`check_tous=20` pas veut dire une incertitude de ±20 sur des valeurs de
20-40 — potentiellement rien qu'un artefact de quantification. **Refait
à résolution fine (`check_tous=1`, chaque pas individuellement) avant
d'accepter le chiffre :**

```
sans fond (M=0)   : t90%(s3)=208  t90%(r3)=4    ratio=0,01923
fond 25% (M=25)   : t90%(s3)=230  t90%(r3)=32   ratio=0,13913
```

**Le signal est RÉEL et même PLUS FORT à résolution fine, pas un
artefact de quantification — l'écart de ratio grandit à ×7,2 (pas
×1,69) une fois mesuré précisément.** Décomposition claire : `s3`
(émetteur) converge à peu près à la même vitesse dans les deux cas
(208 contre 230 pas, ×1,1 seulement) — **c'est `r3` (récepteur) qui
ralentit spécifiquement et fortement (4 contre 32 pas, ×8)** avec la
masse de fond. Ceci confirme directement l'hypothèse transitoire posée
plus haut : la masse de fond dilue `Z` en tout début d'entraînement, ce
qui retarde spécifiquement la convergence du RÉCEPTEUR — exactement
l'asymétrie qu'un paramètre `k` (vitesse relative récepteur/émetteur)
est censé capturer, mesurée ici directement plutôt que supposée.

**Ce que ce chiffre n'est PAS encore : une valeur de `k` comparable au
facteur ~1,7 du tour 52.** Le `×8` mesuré ici est un ratio brut de
temps de convergence (`t90%`), pas le paramètre `k` ajusté dans l'ODE à
deux échelles de temps (qui est calibré différemment, via le point de
bascule de la séparatrice, pas une mesure directe de vitesse). Les deux
quantités pointent dans la MÊME direction qualitative (le récepteur
ralentit relativement à l'émetteur quand un paramètre externe change),
mais ne sont pas sur la même échelle sans le travail de calibration
ODE encore à faire.

**Soumis à un agent-dipankar (règle CLAUDE.md) pour challenger ce
résultat avant de le considérer clos.**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le décalage de `delta_c` avec la masse de fond vient d'un déplacement du point fixe du récepteur lui-même | 17/09 (moi, implicite) | **réfutée** le 17/09 (dérivation analytique : le `+M` est négligeable face aux termes exponentiels de récompense, le point fixe ne dépend pas de M) |
| c'est donc un effet TRANSITOIRE (dilution de `Z` en début d'entraînement), du même type que ce que `k` capture | 17/09 (moi) | **confirmée** le 17/09, mais reformulée (voir correction ci-dessous) |
| le ratio de temps de convergence r3/s3 diffère avec/sans masse de fond, loin de tout pli (`delta=0,013`) | 17/09 (moi) | **réfutée** le 17/09 (ratio identique 0,5000 dans les deux cas — mais mauvais endroit pour le test) |
| le ratio de temps de convergence r3/s3 diffère avec/sans masse de fond, PRÈS du pli | 17/09 (moi) | **confirmée** le 17/09, à résolution fine — mais le chiffre `×7,2` était lui-même artefacté, voir correction |

**CORRECTION IMPORTANTE (agent-dipankar, vérifiée indépendamment
avant d'être acceptée) : le `×8`/`×7,2` publié plus haut était
contaminé par deux problèmes réels, pas un vrai taux de relaxation.**

**1. Bug d'indexage confirmé en relisant mon propre code (le test
original était un `python -c` jamais sauvé — faute en soi, corrigée
maintenant).** Le test verifiait `if i % check_tous == 0` APRÈS
`opt.step()` — donc le "`t=0`" enregistré était déjà après UN pas
d'Adam complet. Avec `lr=0,2`, ce premier pas est quasi un pas plein
en espace de logit (`m1/(√v1+eps)≈sign(gradient)` dès que le
gradient dépasse `eps`) — pas cosmétique, c'est le plus gros
déplacement de toute la trajectoire.

**2. La trajectoire de `r3` n'est PAS monotone.** Vérifié directement
(`verifier_evacuation_fond.py`, script permanent construit après
coup) : `r3` (M=0) descend jusqu'à un MINIMUM de `0,0527` au pas 14
(62 % SOUS la valeur finale `0,1384`) avant de remonter. `t90%`
(premier croisement du seuil) mesure donc le FRONT d'un dépassement,
pas une convergence — l'instrument ne voit pas ce qu'il croit voir.

**3. Mécanisme identifié et nommé par l'agent, vérifié indépendamment :
« évacuation de la masse de fond ».** La masse des M catégories de
fond s'évacue GÉOMÉTRIQUEMENT vite (ratio mesuré `~0,71-0,74`/pas,
demi-vie ~2 pas — vérifié : `0,25→0,183→0,131→0,093→...→0,0025` en 20
pas). **PENDANT cette évacuation, `r3` ET `r4` montent ENSEMBLE**
(tous deux face au fond à récompense nulle, pas encore l'un contre
l'autre) — ce n'est qu'une fois le fond évacué que l'asymétrie
`poids3<poids4` prend le relais et tire `r3` vers le bas. M=0 n'a pas
cette phase (il commence directement en « phase B »).

**Décomposition du `×7,2` brut, vérifiée indépendamment à 3 chiffres
près :**

```
M=0,  fenetre [1,9)   : pente = -0,286460  (log|r3-r3_final| vs pas)
M=25, fenetre [16,34) : pente = -0,129821
ratio des pentes (M=0/M=25) = 2,2066
```

**Ce `2,2066` — le VRAI taux de relaxation post-évacuation du
récepteur — tombe DANS la fourchette `k∈[1,42 ; 2,45]` du tour 52.**
Le `×7,2` brut se décompose en `~×3,3` (délai d'apparition, le temps
que le fond s'évacue) `×~2,2` (vrai ralentissement de taux, une fois
le fond parti). **C'est une confirmation plus fine et mieux posée que
le `×8` original, pas une réfutation — mais ce n'est PAS le chiffre
que j'avais publié.**

**Ablations supplémentaires de l'agent, vérifiées cohérentes avec le
reste (pas re-testées individuellement, mais aucune ne contredit rien
d'établi) :** l'échange de `delta` entre les deux configs ne change
rien (`delta` n'est pas un facteur de confusion) ; égaliser le point
de départ géométrique (`r_tie=0,375` sur M=0) ne reproduit PAS le
ralentissement (`t90(r3)=4`, comme M=0 normal, pas `34`) — la
géométrie seule n'explique rien, c'est bien la masse qui compte ;
`r_autres_init=1e-9` (masse négligeable, M=25 catégories présentes
mais vides) donne le même résultat que M=0 — confirme absolument que
c'est la MASSE, pas le nombre de catégories (cohérent avec la grille
seuil/plafond déjà établie plus haut).

**Point laissé ouvert par l'agent, tranché maintenant (17/09/2026,
suite du même tour) : la dérive de `k` avec `R_init` seul (round 52,
sans masse de fond) est-elle « le même bouton » que la masse de fond ?**
L'agent avait mesuré le balayage `r_tie` avec l'ancien `t90%` contaminé
(moins de ×2 sur toute la plage) — refait avec la MÊME pente
post-transitoire propre (`taux_post_evacuation`, fenêtre `[1,9)`,
`M=0` donc pas d'évacuation à proprement parler mais la même mesure de
pente reste valide) :

```
r_tie=0,50  pente[1,9)=-0,286460  min(r3)=0,0527 au pas 14
r_tie=0,60  pente[1,9)=-0,453440  min(r3)=0,0449 au pas 17
r_tie=0,70  pente[1,9)=-0,541894  min(r3)=0,0387 au pas 20
r_tie=0,75  pente[1,9)=-0,610464  min(r3)=0,0357 au pas 21   <- extremum
r_tie=0,80  pente[1,9)=-0,438835  min(r3)=0,0329 au pas 22
r_tie=0,90  pente[1,9)=-0,234749  min(r3)=0,0269 au pas 26
```

**La pente elle-même varie fortement avec `r_tie` — NON monotone,
maximum (relaxation la plus rapide) vers `r_tie≈0,75`, puis
redescend.** Ratio entre les deux extrêmes mesurés (`0,75` contre
`0,90`) : `0,610464/0,234749 = 2,60`. **Ce facteur `×2,6` est du MÊME
ORDRE DE GRANDEUR que le `×2,2` trouvé pour la masse de fond, et que
le `k∈[1,42 ; 2,45]` du tour 52 — sans AUCUNE masse de fond.**
Conséquence : le mécanisme "R_init seul" et le mécanisme "masse de
fond" produisent des variations de taux COMPARABLES en grandeur, ce
qui appuie (sans le prouver formellement) l'idée qu'ils touchent au
même phénomène sous-jacent (la façon dont la trajectoire doit
traverser la même dynamique compétitive en S, que ce soit parce
qu'elle démarre loin de son point fixe ou parce qu'elle doit d'abord
évacuer une masse parasite). **Différence notable : la dépendance en
`r_tie` est NON monotone (pic à 0,75), alors que la dépendance en masse
de fond était un plafond monotone** — les deux mécanismes ne sont donc
probablement pas IDENTIQUES (un seul et même paramètre sous-jacent),
mais du même ordre de grandeur et de la même famille phénoménologique
(vitesse relative de traversée d'une dynamique en S).

**CORRECTION IMMÉDIATE (agent-dipankar, vérifiée indépendamment,
RÉFUTE ce qui précède) : le "pic à 0,75" est un artefact de fenêtre,
pas un vrai phénomène physique.** L'agent a repéré, avant même de
recalculer quoi que ce soit : `min(r3)` (0,0527→0,0269) et son pas
d'apparition (14→26) sont tous les deux MONOTONES sur les 6 points —
seule ma pente calculée sur une fenêtre FIXE `[1,9)` ne l'est pas. Un
signal brut monotone + une statistique dérivée non-monotone est la
signature d'un artefact de la fenêtre, pas d'un nouveau phénomène.
Confirmé par un modèle nul construit par l'agent (taux vraiment
CONSTANT + une bosse parasite qui se déplace, calée sur mes propres
`argmin`) : appliquer EXACTEMENT le même calcul de pente `[1,9)` à ce
modèle sans aucun mécanisme réel produit AUSSI un pic à `r_tie=0,75`.

**Vérifié directement (fenêtre TARDIVE, bien après tout dépassement,
`[200,400)`, sur les mêmes 6 trajectoires) :**

```
r_tie=0,50  pente[1,9)=-0,286460  pente[200,400)=-0,050615
r_tie=0,60  pente[1,9)=-0,453440  pente[200,400)=-0,052669
r_tie=0,70  pente[1,9)=-0,541894  pente[200,400)=-0,054030
r_tie=0,75  pente[1,9)=-0,610464  pente[200,400)=-0,052617   <- plus de pic
r_tie=0,80  pente[1,9)=-0,438835  pente[200,400)=-0,053246
r_tie=0,90  pente[1,9)=-0,234749  pente[200,400)=-0,053716
```

**La pente tardive est QUASI CONSTANTE (-0,0506 à -0,0540, ~6 % de
variation) sur toute la plage de `r_tie`, pas `×2,60`.** Le pic à
`0,75` disparaît complètement une fois mesuré loin de la fenêtre
précoce contaminée par le passage du minimum mobile. **La conclusion
« R_init seul produit une variation de taux comparable à `k` »
publiée plus haut est FAUSSE — réfutée, pas juste nuancée.** Le vrai
taux asymptotique de `r3` ne dépend quasiment pas de `r_tie` dans ce
jouet.

**Conséquence logique immédiate, à vérifier avant de continuer :** le
`×2,2` de la masse de fond a été mesuré de la MÊME façon (fenêtre
précoce, `[1,9)` pour M=0 contre `[16,34)` pour M=25 — pas une fenêtre
tardive) — est-il LUI AUSSI un artefact du même type ? Vérification
lancée immédiatement (pas laissé en suspens).

**Résultat de la vérification, plus compliqué que prévu — un DEUXIÈME
problème méthodologique trouvé, distinct du premier.** Fenêtre tardive
`[200,400)` : pente(M=0)=`-0,050615`, pente(M=25)=`-0,001039` — ratio
`×5,07`, PAS nul (contrairement au cas `r_tie`) mais aussi PAS `×2,2`.
Poussé plus loin (`pas=3000`, plusieurs fenêtres) pour vérifier la
stabilité :

```
M=0   : pente[200,400)=-0,050610  pente[500,1000)=-0,006512  pente[1000,2000)=-0,000751  pente[1500,2500)=-0,000997
M=25  : pente[200,400)=-0,001039  pente[500,1000)=-0,002241  pente[1000,2000)=-0,001608  pente[1500,2500)=-0,001745
```

**La pente NE SE STABILISE PAS — elle continue de rétrécir avec la
fenêtre pour M=0 (`-0,0506→-0,0007`), sans converger vers une valeur
fixe.** Diagnostic : `r3_final` (la dernière valeur d'un run FINI,
utilisée comme référence pour `|r3(t)-r3_final|`) est elle-même une
CIBLE MOBILE près du pli — vérifié : `r3_final` à `pas=500` valait
`0,139522` (run précédent), à `pas=3000` il vaut `0,140897` (ce
run-ci) — pas identiques. **C'est le même ralentissement critique déjà
documenté ailleurs dans ce même tour (§ ralentissement critique,
tour 50-52) qui contamine ici la mesure : tant que le "point final" de
référence n'est pas le VRAI point fixe (mais la fin d'un run fini plus
proche du pli que je ne le pensais), toute pente calculée contre lui
est biaisée, et le biais rétrécit avec la fenêtre sans jamais se
stabiliser sur un run fini.**

**Conclusion honnête : NI le `×2,2` (masse de fond) NI le `×2,6`
(r_tie) ne sont des mesures fiables du "vrai" taux asymptotique —
tous les deux souffraient d'une référence `r3_final` non convergée,
juste de façons différentes (fenêtre trop précoce pour l'un, cible
mobile pour l'autre). Le "`×2,2` tombe dans `k∈[1,42;2,45]`" n'est
PAS confirmé de façon fiable — ni infirmé, simplement pas mesuré
correctement.** Pour trancher proprement, il faudrait soit (a) des
runs BEAUCOUP plus longs pour que `r3_final` soit vraiment stable
avant de calculer quoi que ce soit, soit (b) utiliser la valeur fixe
ANALYTIQUE du point de branche (`r3_br(s3,s4)` dérivé plus haut ce
même tour) comme référence au lieu de la dernière valeur d'un run
fini — cette deuxième option est probablement la bonne, puisque la
forme fermée existe déjà et évite tout le problème de convergence.
**Pas fait cette session — noté comme la vraie prochaine étape,
distincte de celle notée plus haut.**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le ratio `×2,2` (masse de fond) est, comme le `×2,6` de r_tie, un artefact de fenêtre qui disparaît en fenêtre tardive | 17/09 (moi) | **réfutée en partie** le 17/09 — le ratio ne disparaît PAS (`×5,07` en fenêtre tardive, pas nul), mais un second problème (référence `r3_final` non convergée) invalide la mesure elle-même dans les deux cas |
| la pente se stabilise à une valeur fixe en fenêtre suffisamment tardive | 17/09 (moi) | **réfutée** le 17/09 — continue de rétrécir de `pas=200` à `pas=2500` sans se stabiliser, signature d'une référence `r3_final` elle-même non convergée |

**Suivi de l'agent (message séparé) : sa bissection indépendante de
`delta_c` a fini après son rapport initial — `delta_c(M=0)=0,018711`,
`delta_c(M=25)=0,018555` (tolérance de bissection plus large que la
mienne, d'où l'écart absolu de 0,2 %), mais l'écart RELATIF mesuré
`-0,835%` reproduit quasiment exactement mon `-0,84%`.** Referme le
seul point qu'il avait annoncé ne pas avoir vérifié.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le `t90%` mesure une convergence monotone | 17/09 (moi, implicite) | **réfutée** le 17/09 (agent, vérifié indépendamment) — dépassement réel, `r3` passe 62% sous sa valeur finale avant de remonter |
| le ralentissement du récepteur est un effet homogène de taux (`k` pur) | 17/09 (moi) | **réfutée** le 17/09 (agent) — se décompose en délai d'évacuation (`×3,3`) et taux post-évacuation (`×2,2`), deux mécanismes distincts |
| le taux post-évacuation (`×2,2`) est comparable au `k∈[1,42;2,45]` du tour 52 | 17/09 (agent) | **rétrogradée** le 17/09 (moi, en creusant plus loin) — le `×2,2` lui-même s'est révélé instable selon la fenêtre (`×5,07` en fenêtre tardive), la référence `r3_final` n'étant pas convergée ; ni confirmée ni infirmée, mesure à refaire proprement |
| `delta_c(M=0)`/`delta_c(M=25)` se reproduisent indépendamment | 17/09 (agent, suivi) | **confirmée** le 17/09 (écart relatif -0,835% contre -0,84% publié) — CE résultat n'est pas affecté par le problème de fenêtre ci-dessus (delta_c est un seuil discret, pas une pente) |
| `R_init` (`r_tie`) seul, sans masse de fond, produit une variation de taux du même ordre de grandeur (`×2-2,6`) que la masse de fond | 17/09 (moi, tranchant le point laissé ouvert) | **réfutée** le 17/09 — le pic à 0,75 était un artefact de fenêtre précoce (agent + vérification directe), le vrai taux tardif est quasi constant (~6% de variation) |
| le mécanisme "R_init" et le mécanisme "masse de fond" sont IDENTIQUES (un seul et même paramètre sous-jacent) | 17/09 (moi) | **sans objet** le 17/09 — l'effet R_init lui-même est réfuté (ci-dessus), rien à comparer |

**Bilan final piste 3, honnête : ce qui reste solide, ce qui ne l'est
plus.** Solide, vérifié plusieurs fois indépendamment : (1) le
mécanisme récepteur/masse de fond sur `delta_c` est réel (seuil à
17,75%, plafond à -0,84%, confirmé par l'agent lui-même en bissection
séparée) ; (2) le "`×8`" original était un artefact (bug d'indexage +
non-monotonie), confirmé. **Ce qui s'est effondré en creusant plus
loin, dans la MÊME session, et c'est très bien ainsi (règle méfiance —
le résultat qui vient de confirmer une hypothèse se vérifie PLUS, pas
moins) :** le chiffre précis `×2,2` censé tomber dans
`k∈[1,42;2,45]` n'est PAS fiable tel que mesuré — sa méthode
(pente contre la dernière valeur d'un run fini) souffre du même
ralentissement critique déjà documenté ailleurs dans ce tour, rendant
`r3_final` lui-même une cible mobile. Le lien QUANTITATIF entre le
mécanisme récepteur et `k(R)` du tour 52 reste donc À FAIRE
correctement — soit avec des runs beaucoup plus longs, soit (mieux) en
utilisant `r3_br(s3,s4)` (la forme fermée déjà dérivée ce même tour)
comme référence fixe au lieu d'un run fini. **Pas de chiffre à retenir
pour l'instant — juste la certitude qu'un mécanisme réel existe, sans
que sa taille précise soit encore mesurable proprement.**

Scripts : `verifier_jouet_n_variable.py` (corrigé pour la lenteur de
convergence), `verifier_evacuation_fond.py` (nouveau, corrige le bug
d'indexage et décompose évacuation/taux, mais dont la méthode de
mesure de pente reste elle-même à corriger — voir ci-dessus).

**Repris le 18/09/2026 : point fixe algébrique résolu (pas de
gradient, pas de ralentissement critique de descente) comme référence
FIXE au lieu de la dernière valeur d'un run fini.** Système à 4
équations couplées (`s3=sigmoid(N·poids3·r3/beta)`,
`r3=exp(N·poids3·s3/beta)/Z`, etc.), itération de point fixe pure —
converge en 25-26 itérations à precision machine, sans aucun des
problèmes de cible mobile du gradient.

```
M=0  : point fixe r3*=0,1384022335 — apres 3000 pas d'entrainement, r3=0,1384022336 (ecart 2,18e-11)
M=25 : point fixe r3*=0,1409666240 — apres 3000 pas d'entrainement, r3=0,1408969731 (ecart 6,97e-05)
```

**Déjà révélateur avant même de calculer une pente : à 3000 pas, M=0
est à `2e-11` de son point fixe, M=25 est encore à `7e-5` — 6 ordres
de grandeur d'écart, la lenteur du récepteur avec masse de fond est
massive, pas juste `×2` ou `×8`.**

**Fenêtre adaptative (du dernier extremum local jusqu'au plancher de
bruit `1e-6`, calculée automatiquement, pas choisie à l'œil) :**

```
M=0  : fenetre [190,322)   pente=-0,050454
M=25 : fenetre [143,3001)  pente=-0,001277   (n'atteint PAS le plancher 1e-6 en 3000 pas — fenetre = tout le budget)
```

**Ratio brut `39,5` — mais PAS un chiffre fiable tel quel** : la
fenêtre de M=25 ne représente pas un régime exponentiel unique propre
(elle couvre tout le budget disponible faute d'avoir atteint le
plancher de bruit), contrairement à celle de M=0 qui EST propre (132
pas, bien après tout extremum, bien avant le plancher). **Comparer une
fenêtre propre à une fenêtre qui ne l'est pas donne un nombre, pas une
mesure.** Relancé avec un budget beaucoup plus long (`pas=20000`) pour
M=25, pour trouver SA propre fenêtre adaptative propre.

**Trouvé la VRAIE raison pour laquelle aucune fenêtre n'est jamais
propre pour M=25 — pas un problème de budget, un phénomène déjà connu
dans ce projet.** À `pas=20000`, l'écart au point fixe ne décroît PAS
monotonement :

```
pas=1000   ecart=5,1300e-04
pas=3000   ecart=6,9651e-05
pas=5000   ecart=1,9891e-05
pas=10000  ecart=3,7972e-05
pas=15000  ecart=4,6416e-04   <- REBOND de presque 2 ordres de grandeur
pas=19999  ecart=1,1902e-07
```

**C'est le même phénomène d'EXCURSIONS déjà caractérisé en détail plus
haut dans ce même tour (H15 : artefact du second moment d'Adam,
`beta2=0,999`, confirmé via l'optimiseur hybride sur le VRAI système à
27 référents — pas une coïncidence, le même mécanisme, retrouvé
indépendamment ici sur le jouet réduit).** Ce n'est donc pas un
problème de fenêtre de mesure à affiner encore — c'est une limite
FONDAMENTALE de l'approche "lire une pente sur une trajectoire" près
d'un pli sous Adam : la trajectoire elle-même n'est jamais un simple
exponentiel unique, elle est ponctuée d'excursions imprévisibles.
**Exactement la raison pour laquelle le tour 52 avait dû abandonner la
mesure directe de taux au profit du protocole "épingler-falsifier"
(bissecter le point de bascule, ajuster `k` sur l'ODE à deux
échelles de temps) plutôt que de lire `k` sur une trajectoire brute.**

**Conclusion méthodologique pour cette piste : la bonne façon de
chiffrer le lien entre la masse de fond et `k(R)` n'est PAS de mesurer
une pente de trajectoire (quelle que soit la fenêtre, la référence, ou
le budget) — c'est de construire l'ODE à deux échelles de temps ET le
protocole pin-and-falsify POUR CE JOUET**, en utilisant les fonctions
de branche maintenant disponibles (`r3_br(s3,s4)`, le point fixe
algébrique déjà résolu plus haut), exactement le travail de
modélisation identifié comme la vraie prochaine étape dès le début de
cette piste — confirmé, pas contourné, après plusieurs tentatives de
raccourci toutes tombées sur la même limite fondamentale.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| une fenêtre adaptative bien choisie donne une pente stable et fiable pour M=25 | 17-18/09 (moi) | **réfutée** le 18/09 — même à 20000 pas, l'écart au point fixe n'est pas monotone, rebond de ~2 ordres de grandeur à pas=15000 |
| l'échec des mesures de pente vient des mêmes excursions (H15, second moment d'Adam) déjà trouvées sur le système complet | 18/09 (moi) | **confirmée** le 18/09 — signature identique (rebond imprévisible, pas de décroissance monotone) retrouvée indépendamment sur le jouet réduit |
| la mesure directe de taux sur trajectoire ne peut pas donner de `k` fiable près d'un pli sous Adam, quelle que soit la méthode de fenêtrage | 18/09 (moi) | **confirmée** le 18/09 — cohérent avec le choix méthodologique du tour 52 (pin-and-falsify sur l'ODE, pas de lecture directe de pente) |

**Soumis à un agent-dipankar (règle CLAUDE.md) pour challenger cette
conclusion méthodologique. Retour très substantiel, vérifié pour
l'essentiel avant d'être accepté.**

**1. Vérification de cohérence qu'il propose (`r4*=0,75-r3*` si
« masse totale=25% » persiste à l'équilibre) : REFAIT, et son
hypothèse était fausse, la mienne confirmée.** `r3*+r4*` calculé =
`1,0000000000` EXACTEMENT (pas `0,75`) — la masse de fond est bien
nulle à l'équilibre (dérivé analytiquement plus haut ce même tour),
« masse totale=25% » décrit la condition INITIALE d'entraînement, pas
une part qui persiste. Son test a raté parce qu'il n'avait pas cette
dérivation (accès repo refusé) — mais le test lui-même était le bon
réflexe (méfiance), et il l'a signalé honnêtement comme non vérifiable
de son côté plutôt que de trancher à l'aveugle.

**2. Correction mineure confirmée : `6,97e-05/2,18e-11 = 3 197 248`,
`log10=6,505`, pas `6`.** Écart de 6,5 ordres de grandeur, pas 6 —
corrigé.

**3. Trouvaille plus importante, pas encore explorée : M=0 et M=25
sont à quasiment la MÊME distance absolue de leur pli respectif
(`9,336e-4` contre `9,258e-4`, écart 0,84%), mais convergent avec un
écart de 6,5 DÉCADES.** Si la distance au pli seule fixait l'échelle
de temps, les deux devraient être comparablement lents. **Ça veut dire
que `M` ne fait pas que déplacer `delta_c` — il change la GÉOMÉTRIE
LOCALE du pli** (courbure, ou une direction propre lente propre à M,
pas encore isolée). Un vrai « pourquoi CE nombre » non résolu, noté
pour une reprise future.

**4. Apport théorique décisif, vérifié en dérivant moi-même la forme
normale d'un pli standard : à `mu=0` EXACTEMENT (pile sur le pli),
`dx/dt=-x²` donne `x(t)=x0/(1+k·x0·t)` — une décroissance en PUISSANCE
(1/t), PAS exponentielle, à AUCUNE fenêtre, purement déterministe
(sans Adam, sans bruit).** Ça explique, indépendamment de H15, pourquoi
une pente en fenêtre précoce ne peut jamais se stabiliser près d'un
pli : il n'existe pas de taux exponentiel unique tant que `t` n'est
pas très supérieur à `1/(2√mu)`. **Les deux mécanismes coexistent :
un effet de "fantôme" déterministe du pli (proche de `mu=0`) ET les
excursions d'Adam (H15) — pas l'un OU l'autre.**

**5. Test précommis par l'agent pour distinguer les deux, exécuté
directement : faire varier `beta2` et voir si la récurrence des
rebonds suit sa mémoire.** Trajectoire complète (`pas=0` à `19000`,
tous les 1000 pas) à `beta2=0.999` (mémoire 1000 pas) contre `beta2=0.9`
(mémoire 10 pas) :

```
beta2=0,999 : descend a ~1e-6 vers pas=9000, PUIS rebonds ponctuels
              (pas=10000: 3,1e-5 ; pas=15000-16000: jusqu'a 2,0e-3 ;
              pas=18000: 4,2e-4) -- periodes de stabilite longues,
              interrompues par des sauts occasionnels.
beta2=0,9   : NE SE STABILISE JAMAIS -- oscille en continu entre
              2,5e-4 et 2,1e-3 sur TOUTE la duree (19000 pas), aucune
              periode de stabilite comparable a beta2=0,999.
```

**Signature qualitative nette : réduire la mémoire d'Adam (`beta2`
plus petit) rend les oscillations PLUS fréquentes, pas plus rapides à
disparaître — exactement ce que prédit H15 (le second moment d'Adam
cause l'instabilité), et incompatible avec un pur artefact déterministe
du pli (qui ne dépend pas de `beta2`, qui n'existe même pas dans une
ODE pure).** **H15 confirmé comme un vrai contributeur, pas juste une
hypothèse par défaut — vérifié par la MÊME méthode qui l'avait établi
sur le système complet (variation de beta2), retrouvée indépendamment
sur ce jouet réduit.**

**6. Sur l'immunité du protocole ODE+pin-and-falsify (question posée
à l'agent) : PAS totalement immunisé, argument accepté sans
contre-test (dérivation théorique propre, pas de calcul à vérifier).**
Bissecter à budget fini `T` près du pli hérite du même goulot
(distinguer "converge" de "encore en train de relaxer" demande
`t~O(1/√mu)`) — bande de classification irréductible de largeur
`~C/T²` autour de `delta_c`, pas un point exact. Correction à
appliquer la prochaine fois : rapporter un INTERVALLE pour `delta_c`,
pas un chiffre unique, et doubler le budget adaptativement quand une
étape de bissection tombe dans la bande d'incertitude courante.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le rebond vient de H15 (second moment d'Adam) | 18/09 (moi) | **confirmée** le 18/09 par test direct (beta2=0,9 oscille en continu, beta2=0,999 a de longues périodes stables — signature H15, pas un artefact déterministe indépendant de beta2) |
| le rebond pourrait être un pur artefact déterministe du pli (fantôme/bottleneck), sans rapport avec Adam | 18/09 (agent) | **réfutée en tant qu'EXPLICATION UNIQUE** le 18/09 — le phénomène du fantôme existe bien (dérivation théorique confirmée) mais n'explique pas la dépendance en `beta2` observée ; les deux mécanismes coexistent |
| M change seulement la POSITION du pli (`delta_c`), pas sa géométrie locale | 18/09 (moi, implicite) | **réfutée** le 18/09 (agent) — deux configs à distance-au-pli quasi identique convergent avec 6,5 décades d'écart, la géométrie locale doit différer |
| le protocole ODE+pin-and-falsify est totalement immunisé contre le problème de fenêtre/budget près du pli | 18/09 (moi, implicite) | **réfutée** le 18/09 (agent, argument théorique accepté) — bande de classification irréductible `~1/T²`, correction : rapporter un intervalle, budget adaptatif |

**Bilan définitif de ce fil (piste 3, lien k(R)) : le mécanisme
récepteur/masse de fond est réel et son existence est solidement
établie (via `delta_c`, un seuil discret non affecté par les
excursions) — ET sa lenteur près du pli a maintenant DEUX explications
identifiées et distinguées (fantôme déterministe du pli + excursions
Adam H15, confirmées coexister par un test direct). Sa taille précise,
comparée à `k∈[1,42;2,45]`, reste À CHIFFRER via le protocole
ODE+pin-and-falsify — désormais avec la mise en garde qu'il faudra
rapporter un intervalle et non un point, et avec une nouvelle question
ouverte (pourquoi M change la géométrie locale du pli, pas juste sa
position) à explorer en même temps. C'est la tâche pour la prochaine
session, pas un raccourci qui reste à trouver — mais on comprend
maintenant BEAUCOUP mieux pourquoi les raccourcis ont tous échoué.**

Scripts : `verifier_point_fixe_jouet_m.py` (point fixe + fenêtre
adaptative), tests `beta2` ad hoc (à sauver en script permanent si
cette piste est reprise).

---

### Piste 3c : test direct sur le VRAI système — le bassin s'inverse sous masse de fond

18/09/2026, suite du même tour (Théo : « on n'arrête pas »). Reprise du
protocole pin-and-falsify du tour 52 DIRECTEMENT sur le système complet
à 27 référents, plutôt que de construire une ODE pour le jouet
(`verifier_masse_fond_systeme_reel.py`).

**Baseline (sans masse de fond), reproduit avec succès :** bissection
du point de bascule à `R_init=0,60`, `delta=0,013` — `flip=0,979616`,
`k` lu sur la table de dipankar (tour 52) par interpolation :
`k=1,613`. Dans la fourchette `[1,42 ; 2,45]` déjà connue.

**Avec 25 % de masse de fond (10 référents parmi les 25 autres, message
10) : le script a PLANTÉ (assertion `lo/hi même issue`), pas un signe
d'échec — un signe qu'il fallait regarder de plus près plutôt que
d'élargir le bracket à l'aveugle.** Diagnostic : `s3_init=0,999` →
GRADUÉ (`R=0,794022`, exactement la branche connue) ; `s3_init=0,9999`
→ EFFONDRÉ (`s3=0,037037=1/27`, la signature de collapse connue).
**`s3_init` plus ÉLEVÉ donne l'effondrement, pas la branche graduée —
l'INVERSE du cas non perturbé.**

**Cartographié sur 9 points (0,90 à 0,99999) : le renversement est net
et monotone dans les DEUX régimes, pas un artefact isolé.**

```
s3_init=0,90000  ->  GRADUEE  (R=0,794022)
s3_init=0,95000  ->  GRADUEE  (R=0,795609)
s3_init=0,97000  ->  GRADUEE  (R=0,794022)
s3_init=0,99000  ->  GRADUEE  (R=0,794022)
s3_init=0,99500  ->  GRADUEE  (R=0,794022)
s3_init=0,99900  ->  GRADUEE  (R=0,794022)
s3_init=0,99950  ->  EFFONDRE (s3=0,037035, R=1,000000)
s3_init=0,99990  ->  EFFONDRE (s3=0,037037, R=1,000000)
s3_init=0,99999  ->  EFFONDRE (s3=0,037032, R=1,000000)
```

**Cycle complet appliqué avant la bissection précise du nouveau seuil
(Théo a rappelé le cycle répondre/expérimenter/hypothèses/analyser,
les axes QUAND/COMMENT/POURQUOI, le quota 3-5 hypothèses) :**

**Hypothèses (posées le 18/09, AVANT de connaître le seuil précis) :**

| # | hypothèse | type | posée le | statut |
|---|---|---|---|---|
| H-ordre : pendant l'évacuation du fond, r3 ET r4 montent ensemble contre le fond ; à s3_init très élevé, l'émetteur n'a "nulle part où monter" (déjà saturé) et ne peut pas compenser l'avantage structurel de poids4>poids3 pendant cette phase, alors qu'à s3_init modéré l'émetteur bouge encore et peut compenser | standard (gradient/saturation) | 18/09 | ouverte |
| H-plancher : le gradient de l'émetteur est quasi nul très près de la saturation (déjà documenté ailleurs dans ce projet, planchers `adam_eps`) — à s3_init=0,9999+ l'émetteur est effectivement gelé pendant toute la phase critique, contrairement à s3_init=0,90-0,999 où il a encore un vrai gradient | standard (plancher numérique) | 18/09 | ouverte |
| H-course : le "vainqueur" de la compétition r3-vs-r4 après évacuation dépend d'un avantage marginal accumulé PENDANT les premiers pas (pas de la préférence de départ) — le seuil de bascule devrait coïncider avec là où le gradient de l'émetteur devient négligeable, pas un chiffre arbitraire | non standard | 18/09 | ouverte |
| H-artefact-optimiseur : `continuer_sous_prior` ne réinitialiserait pas l'état Adam entre les appels de bissection, contaminant les résultats successifs | non standard (mais vérifiable directement dans le code) | 18/09 | **réfutée le 18/09** — vérifié : `continuer_sous_prior` appelle `torch.optim.Adam(...)` à chaque appel, un optimiseur frais à chaque fois, pas d'état résiduel |
| H-bug-fixer : `fixer_masse_fond` interagit mal avec `fixer_s3` (ordre d'appel, cases qui se chevauchent) | non standard (vérification de code) | 18/09 | **réfutée le 18/09** — les deux fonctions touchent des tenseurs différents (`e.p` pour l'émetteur, `r.p` pour le récepteur), aucun chevauchement possible |

**QUAND :** le seuil de bascule se situe entre `s3_init=0,999` (gradué)
et `0,9995` (effondré) — pas encore localisé plus précisément,
bissection en cours.

**COMMENT (mécanisme, étape par étape) :** la masse de fond (25 %,
répartie sur 10 référents) s'évacue rapidement (par analogie avec le
jouet, demi-vie de quelques pas) ; PENDANT cette évacuation, `r3` et
`r4` montent tous les deux contre le fond (pas encore l'un contre
l'autre) ; une fois le fond épuisé, la compétition `r3`-vs-`r4`
commence avec un avantage relatif hérité de cette phase — et cet
avantage semble dépendre de l'état de l'ÉMETTEUR au moment où la
compétition démarre (mobile vs déjà figé), pas de sa valeur de départ
en tant que telle.

**POURQUOI (à confirmer) :** si H-plancher/H-course tiennent, c'est
parce qu'un émetteur encore mobile (`s3_init` pas trop proche de 1)
peut continuer à répondre au signal de récompense PENDANT la phase
critique d'évacuation, alors qu'un émetteur déjà saturé (`s3_init`
quasi 1) ne peut plus bouger et rate cette fenêtre — inversant quel
référent « gagne » la course, sans rapport avec la préférence de
poids `delta`.

**Croisement (quand du comment) :** est-ce que ce mécanisme
d'inversion existe UNIQUEMENT très près de la saturation du
checkpoint d'origine (`s3` déjà quasi figé avant même la
perturbation), ou apparaît-il aussi à des deltas plus éloignés du pli,
ou avec un `s4_init` différent ? Pas testé.

**Seuil précis localisé : entre `s3_init=0,9994521` et `0,9994526`**
(net, résolution ~5e-7, pas de flou — un vrai bord, pas un bassin
fractal). Comparé au seuil non perturbé (`s3=0,994300`, 3 décades plus
loin de la saturation) : le seuil sous masse de fond est ~10× plus
proche de `s3=1` que le seuil d'origine.

**H-plancher testée directement et RÉFUTÉE dans sa forme simple.**
Gradient brut de l'émetteur (`e.p[0].grad[3,10]`, un seul pas avant/
arrière, pas d'entraînement) à travers la zone du seuil :

```
s3_init=0,99      grad=-5,144832e-05
s3_init=0,999     grad=-3,480993e-06
s3_init=0,9994    grad=-1,862357e-06
s3_init=0,99945   grad=-1,671796e-06   <- juste sous le seuil
s3_init=0,9995    grad=-1,484590e-06   <- juste au-dessus
s3_init=0,9999    grad=-1,778014e-07
```

**Le gradient décroît de façon parfaitement LISSE et continue à
travers le seuil — aucune discontinuité, aucun plancher net à cet
endroit précis.** Ça réfute l'explication la plus simple (« le
gradient tombe à zéro pile là »). Le seuil est un vrai effet
DYNAMIQUE de compétition/course pendant la phase d'évacuation
(H-course), pas un artefact statique de gradient — cohérent avec le
caractère NET (pas graduel) du seuil lui-même : une course a un
gagnant net, pas une transition graduelle, même si le gradient qui
l'alimente varie continûment.

**Conséquence pour la question initiale (« combien vaut le décalage de
`k` ») : réponse honnête, PAS le chiffre espéré, mais une réponse
quand même (Théo : « il nous faut des réponses même négatives ou
positives »).** Ce renversement de bassin est QUALITATIF, pas
QUANTITATIF — il ne s'agit plus de lire un `k` différent sur la MÊME
table de dipankar (qui suppose une orientation de bassin fixe), mais
d'une réorganisation complète de la structure de bassin sous masse de
fond significative. **Le résultat décisif est donc : « oui, la masse
de fond affecte massivement la dynamique près du pli — au point de
retourner l'orientation du bassin, pas seulement de déplacer un
seuil » — plus fort que ce qui était cherché, mais pas directement
comparable au chiffre `k∈[1,42;2,45]` sans reconstruire la table pour
CETTE orientation de bassin.**

| # | hypothèse | type | posée le | statut |
|---|---|---|---|---|
| H-plancher : un plancher net de gradient coïncide avec le seuil précis | standard | 18/09 | **réfutée** le 18/09 — gradient parfaitement lisse à travers le seuil, aucune discontinuité |
| le seuil renversé peut se lire directement sur la table k de dipankar (même convention) | 18/09 (moi, implicite) | 18/09 | **réfutée** le 18/09 — la table suppose une orientation de bassin fixe, un bassin RENVERSÉ n'est pas le même objet, pas de lecture directe possible |
| la masse de fond a un effet qualitatif (pas seulement quantitatif) sur la dynamique près du pli | 18/09 (moi) | 18/09 | **confirmée** le 18/09 — renversement complet de l'orientation du bassin, pas un simple décalage |

**Bilan piste 3c : résultat DÉCISIF et inattendu — la masse de fond ne
se contente pas de ralentir le récepteur (déjà établi), elle peut
RENVERSER l'orientation du bassin de bascule près du pli, sur le vrai
système à 27 référents.** Ni confirmation simple ni réfutation simple
de « k se déplace comme prévu » — un résultat plus riche que la
question posée, à soumettre à un agent-dipankar avant de le considérer
clos (règle CLAUDE.md), puis à explorer plus loin (est-ce spécifique à
`R_init=0,60`, à `delta=0,013`, à ce jeu précis de 10 référents ?).

**Testé tout de suite (pas laissé en suspens) : le renversement est-il
universel sur tous les `R_init`, ou spécifique à `0,60` ?** À
`R_init=0,75` (plus proche de la vraie valeur de branche `≈0,794` que
`0,60`), les deux extrêmes testés restent GRADUÉS :

```
R_init=0,75  s3_init=0,9000  ->  GRADUEE (R=0,794112)
R_init=0,75  s3_init=0,9999  ->  GRADUEE (R=0,794022)
```

**Pas de renversement à `R_init=0,75` — le phénomène n'est PAS
universel, il est spécifique (au moins) à `R_init=0,60`.** Cohérent
avec une lecture géométrique simple : `R_init=0,75` est déjà proche de
la vraie branche (`0,794`), donc robustement dans le bassin gradué
quel que soit `s3_init` et la perturbation ; `R_init=0,60` est plus
loin de la branche, plus proche de la zone sensible où la séparatrice
d'origine (au seuil `s3=0,9943`) passait déjà — c'est là que la
perturbation a le plus de prise pour renverser l'issue.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le renversement de bassin sous masse de fond est universel, indépendant de `R_init` | 18/09 (moi, implicite) | **réfutée** le 18/09 — absent à `R_init=0,75`, présent à `R_init=0,60` |
| le renversement se manifeste préférentiellement près de la zone de sensibilité déjà connue de la séparatrice d'origine, pas partout | 18/09 (moi) | **cohérente avec les 2 points testés, pas encore un test direct** |

**Soumis à un agent-dipankar (règle CLAUDE.md). Critique dense —
plusieurs failles méthodologiques réelles trouvées :** (1) une
bissection à `tol=1e-7` termine TOUJOURS avec un bracket de cette
largeur, que la transition soit une vraie discontinuité ou un
croisement continu raide — « net » décrit mon critère d'arrêt, pas
forcément le système ; (2) ma grille de gradient (espacement `~5e-5`)
est 100× plus grossière que le bracket du seuil (`5e-7`) — un
« lisse » mesuré à cette échelle ne peut pas exclure un coude DANS un
intervalle entre deux points testés ; (3) surtout : réfuter `g3=0`
exactement au seuil ne confirme PAS la course — il faudrait observer
`g3` et `g4` CROISER (changer de signe relatif), pas juste vérifier
que `g3` seul est non-nul. **Question précise posée par l'agent :
`grad[4,10]` (émetteur du référent 4) aux mêmes 6 points, `g3-g4`
change-t-il de signe ?**

**Testé directement, réponse NÉGATIVE et décisive (compte comme
résultat, pas un échec — Théo, 18/09 : « il nous faut des réponses
même négatives ») :**

```
cible_s3=0,99000    grad3=-5,144832e-05  grad4=2,615269e-14  grad3-grad4=-5,14e-05
cible_s3=0,99900    grad3=-3,480993e-06  grad4=2,615269e-14  grad3-grad4=-3,48e-06
cible_s3=0,99940    grad3=-1,862357e-06  grad4=2,615269e-14  grad3-grad4=-1,86e-06
cible_s3=0,99944    grad3=-1,709650e-06  grad4=2,615269e-14  grad3-grad4=-1,71e-06
cible_s3=0,99945    grad3=-1,671796e-06  grad4=2,615269e-14  grad3-grad4=-1,67e-06
cible_s3=0,999452   grad3=-1,664241e-06  grad4=2,615269e-14  grad3-grad4=-1,66e-06
cible_s3=0,999453   grad3=-1,660466e-06  grad4=2,615269e-14  grad3-grad4=-1,66e-06
cible_s3=0,99946    grad3=-1,634077e-06  grad4=2,615269e-14  grad3-grad4=-1,63e-06
cible_s3=0,99950    grad3=-1,484590e-06  grad4=2,615269e-14  grad3-grad4=-1,48e-06
cible_s3=0,99990    grad3=-1,778014e-07  grad4=2,615269e-14  grad3-grad4=-1,78e-07
```

**`grad4` est CONSTANT et négligeable (2,6e-14, bruit numérique)
partout — référent 4 est déjà saturé depuis le checkpoint d'origine,
son émetteur ne bouge jamais. `grad3-grad4` ne change JAMAIS de signe
sur toute la plage, y compris à résolution fine autour du seuil exact
(`0,999452`/`0,999453`, à l'intérieur même de l'ancien intervalle
grossier).** H-course, dans sa formulation littérale (« une course
entre les deux gradients d'émetteur à `t=0` »), est **RÉFUTÉE** : il
n'y a pas de croisement à observer, un seul gradient bouge, l'autre
est figé depuis le départ. Le mécanisme n'est PAS une compétition
entre les deux ÉMETTEURS — il doit se jouer ailleurs (côté RÉCEPTEUR,
comme l'agent le suggère en alternative, ou dans l'évolution du
gradient de l'émetteur 3 PENDANT l'entraînement plutôt qu'à `t=0`,
que je n'ai pas non plus testée).

| # | hypothèse | posée le | statut |
|---|---|---|---|
| H-course littérale : `g3` et `g4` (gradients des deux émetteurs) se croisent près du seuil, déterminant qui « gagne » | 18/09 (moi) | **réfutée** le 18/09 — `g4` constant et négligeable partout, aucun croisement possible |
| le mécanisme du renversement est une compétition entre les deux ÉMETTEURS | 18/09 (moi, implicite) | **réfutée** le 18/09 — un seul gradient d'émetteur bouge (référent 3), l'autre est figé ; la compétition doit être ailleurs |

Script : `verifier_masse_fond_systeme_reel.py`. Piste ouverte pour la
reprise : tester le gradient du RÉCEPTEUR (pas de l'émetteur) aux
mêmes points, et/ou suivre `g3(t)` sur toute la trajectoire (pas
seulement `t=0`) pour voir s'il croise un plancher `adam_eps` en cours
de route plutôt qu'au départ.

**Fait tout de suite (pas laissé en suspens) : gradient du RÉCEPTEUR
aux mêmes points.**

```
cible_s3=0,99000   grad_r3=-2,437375e-03  grad_r4=-4,118794e-03  diff=1,681418e-03
cible_s3=0,99900   grad_r3=-2,506465e-03  grad_r4=-4,074379e-03  diff=1,567913e-03
cible_s3=0,99944   grad_r3=-2,509843e-03  grad_r4=-4,072207e-03  diff=1,562364e-03
cible_s3=0,99945   grad_r3=-2,509920e-03  grad_r4=-4,072158e-03  diff=1,562238e-03
cible_s3=0,999452  grad_r3=-2,509935e-03  grad_r4=-4,072148e-03  diff=1,562213e-03
cible_s3=0,999453  grad_r3=-2,509943e-03  grad_r4=-4,072143e-03  diff=1,562200e-03
cible_s3=0,99946   grad_r3=-2,509996e-03  grad_r4=-4,072108e-03  diff=1,562112e-03
cible_s3=0,99950   grad_r3=-2,510303e-03  grad_r4=-4,071911e-03  diff=1,561608e-03
cible_s3=0,99990   grad_r3=-2,513374e-03  grad_r4=-4,069937e-03  diff=1,556563e-03
```

**Même verdict que côté émetteur : `grad_r3-grad_r4` reste POSITIF et
quasi constant (~1,562e-3) sur toute la plage, aucun croisement, aucune
discontinuité au seuil.** Ni le gradient de l'émetteur, ni celui du
récepteur, pris à `t=0`, n'expliquent le seuil précis. **Conclusion
robuste : le mécanisme du renversement de bassin n'est PAS visible
dans un instantané statique (quel que soit le côté regardé) — c'est
un phénomène authentiquement DYNAMIQUE/TEMPOREL, qui se joue au fil
des ~40000 pas d'entraînement, pas dans l'état de départ.** Cohérent
avec l'intuition de dilution/évacuation (une COURSE dans le temps, pas
une différence de point de départ), mais reste à observer directement
(suivre la trajectoire complète, pas un instantané) — pas fait cette
session, tâche bien scopée pour la suite.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le gradient du récepteur (pas de l'émetteur) a la vraie discontinuité au seuil | 18/09 (agent, alternative proposée) | **réfutée** le 18/09 — `grad_r3-grad_r4` aussi lisse et sans croisement que côté émetteur |
| le mécanisme du renversement est authentiquement dynamique (visible seulement en suivant la trajectoire, pas un instantané) | 18/09 (moi) | **confirmée par élimination** le 18/09 — aucune explication statique (émetteur OU récepteur, à t=0) ne tient |

Script : `verifier_trajectoire_renversement.py` (permanent, remplace
les invocations ad hoc de gradient).

**TROUVÉ (18/09/2026, suite immédiate) : le moment exact du
renversement, en suivant la trajectoire complète plutôt qu'un
instantané.** Les deux configurations (`cible_s3=0,999450` vs
`0,999455`, écart de `5e-6` seulement) sont QUASI IDENTIQUES pendant
les 600 premiers pas :

```
pas=0    : s3=0,999450 vs 0,999455 (les deux)
pas=300  : s3=0,997367 vs 0,997357 (quasi identiques)
pas=600  : s3=0,995611 vs 0,995291 (encore tres proches)
```

**Puis elles divergent brutalement entre pas=600 et pas=800 :**

```
config SOUS le seuil (0,999450) :
  pas=600  s3=0,995611  g_e3=8,1540e-12   <- gradient quasi NUL, point selle en TEMPS
  pas=700  s3=0,996039  <- REMONTE (le signe du gradient a bascule)
  pas=800  s3=0,997898  <- continue de remonter, va converger vers la branche graduee

config AU-DESSUS du seuil (0,999455) :
  pas=600  s3=0,995291  g_e3=2,8931e-09
  pas=700  s3=0,993272  <- continue de DESCENDRE
  pas=800  s3=0,038651  <- EFFONDREMENT CATASTROPHIQUE entre pas=700 et pas=800
```

**Interprétation, avec les axes QUAND/COMMENT/POURQUOI :**

**QUAND :** le point de bifurcation temporelle se situe autour de
`pas≈600-700` — PAS à `t=0`. C'est exactement pourquoi aucun
instantané statique (gradient émetteur OU récepteur, à `t=0`) ne
pouvait révéler le mécanisme : il n'existe simplement pas encore à ce
moment-là.

**COMMENT :** la trajectoire, après avoir évacué la masse de fond
(rapide, `pas<200`) et traversé une phase de réajustement (`pas
200-600`), ARRIVE À UN POINT DE PRESSION QUASI NULLE (`g_e3≈8e-12` à
`pas=600` pour la config qui va survivre) — un vrai point selle
rencontré EN COURS DE ROUTE, pas au départ. La minuscule différence
initiale (`5e-6` sur `s3_init`), amplifiée par ~600 pas de dynamique
lente, suffit à déterminer de quel côté du point selle la trajectoire
arrive à ce moment précis — et donc si elle rebondit (branche graduée)
ou s'effondre.

**POURQUOI :** c'est le même phénomène de « fantôme »/bottleneck déjà
théorisé (par un agent) pour un pli en dynamique lente — sauf qu'ici,
contrairement au cas sans masse de fond (où le point selle est
rencontré directement à `t=0`, via le choix de `s3_init`), la masse de
fond retarde et DÉPLACE le moment de rencontre avec ce point selle à
`pas≈600-700`. La masse de fond ne crée pas un nouveau mécanisme —
elle change QUAND le système rencontre le même type de point selle
qui existait déjà (H6, confirmé au tour 51), le repoussant dans le
temps plutôt que dans l'espace des conditions initiales.

**Zoom fin fait (pas=550-850, tous les 5 pas) : le moment exact est
localisé, et `g_e3` change bien littéralement de signe.**

```
config SOUS le seuil (survit) :
  pas=600  g_e3=+8,1540e-12   (derniere valeur positive)
  pas=605  g_e3=-1,3590e-10   <- CHANGEMENT DE SIGNE ICI, entre pas=600 et 605
  pas=610  g_e3=-2,8982e-10   (de plus en plus negatif, s3 remonte franchement ensuite)

config AU-DESSUS du seuil (s'effondre) :
  pas=550 a 845 : g_e3 reste POSITIF tout du long, sans jamais changer
  de signe -- au contraire, sa MAGNITUDE croit (2,2e-9 a pas=550, jusqu'a
  2,1e-6 a pas=760) -- puis effondrement catastrophique :
  pas=745  s3=0,976822
  pas=750  s3=0,963591
  pas=755  s3=0,927345
  pas=760  s3=0,783917
  pas=765  s3=0,316994
  pas=770  s3=0,088940   <- l'essentiel de la chute en 25 pas
```

**Le mécanisme est maintenant entièrement clair : les deux
trajectoires, qui ne diffèrent que de `5e-6` sur `s3_init`, restent
quasi confondues pendant ~600 pas (approche lente commune d'un point
selle DYNAMIQUE, pas juste algébrique) ; à `pas≈600-605`, le gradient
de la config « SOUS » change de signe (rebondit) alors que celui de
« AU-DESSUS » ne change jamais de signe (continue d'éroder, de plus en
plus vite) — un cas manuel, texbook, de sensibilité extrême aux
conditions initiales près d'un point selle, où 600 pas de dynamique
lente amplifient un écart de `5e-6` en une divergence totale.** C'est
exactement la signature attendue d'une séparatrice traversée EN TEMPS
plutôt qu'en ESPACE DES CONDITIONS INITIALES — cohérent à 100% avec le
POURQUOI déjà écrit (même point selle H6, déplacé dans le temps par la
masse de fond).

| # | hypothèse | posée le | statut |
|---|---|---|---|
| `g_e3` change littéralement de signe au moment du basculement (pas juste "quasi nul") | 18/09 (moi) | **confirmée** le 18/09 — changement de signe net entre pas=600 et pas=605 pour la trajectoire qui survit ; la trajectoire qui s'effondre ne change jamais de signe |

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le renversement de bassin se joue à un moment précis PENDANT l'entraînement (pas à t=0), visible en suivant la trajectoire | 18/09 (moi) | **confirmée** le 18/09 — divergence nette entre pas=600 et pas=800, gradient quasi nul à pas=600 pour la trajectoire qui survit |
| la masse de fond crée un NOUVEAU mécanisme de bifurcation, distinct du point selle H6 déjà connu | 18/09 (moi, implicite) | **réfutée** le 18/09 — c'est le MÊME type de point selle (gradient quasi nul, extrême sensibilité), juste rencontré à un autre moment, pas un mécanisme différent |

**Soumis à un agent-dipankar une 3e fois sur ce fil. Retour riche —
chiffres affinés, un préfacteur qui confirme joliment la lecture
« fantôme de nœud-col », et sa question finale réglée par simple
lecture de code, pas par expérience.**

**Précisions numériques (recalculées indépendamment par l'agent, sans
prendre mes valeurs pour acquises) :**
- Traversée de zéro de `g_e3` (config SURVIT) : pas exactement
  `600,283` par interpolation linéaire entre pas=600 et 605 — pas «
  pas=600-605 », une valeur précise.
- Facteur de croissance du gradient (config EFFONDRE, pas 550→760) :
  `954,5×`, pas `~1000` (j'avais arrondi 5% trop haut).
- `λ_grad` (taux local, pas 550-760) = `0,0327`/pas (e-folding 30,6
  pas) ; `λ_state` (taux sur toute la fenêtre 0-770) = `0,0157`/pas
  (e-folding 63,6 pas). **Ratio `2,08` entre les deux — les deux NE
  DEVRAIENT PAS concorder si la croissance n'est pas uniformément
  exponentielle, et elles ne concordent pas : ~550 des 770 pas sont
  passés dans un régime de passage LENT (bien en dessous du taux
  asymptotique), puis ~210 pas au taux rapide.** Exactement la forme
  attendue d'un « fantôme » de nœud-col (forme normale de Strogatz
  `dx/dt=μ+x²`, temps de passage `τ=C/√μ`).

**Préfacteur `C` calculé (test que je n'avais pas fait) — confirmation
propre :**

```
mu_survit = |0,999450-0,99945235| = 2,35e-6  ->  C = 600/mu^(-1/2) = 0,920
mu_effondre = |0,999455-0,99945235| = 2,65e-6 ->  C = 600/mu^(-1/2) = 0,977
```

**`C` d'ordre 1 dans les deux configs, à 6% près l'une de l'autre —
cohérent avec un vrai fantôme de nœud-col.** Ne règle PAS à lui seul
« même point selle que H6 » (un point selle DIFFÉRENT créé par la
masse résiduelle donnerait aussi un `C` d'ordre 1) — juste une
confirmation supplémentaire de la STRUCTURE (fantôme), pas de
l'IDENTITÉ (même point selle vs nouveau).

**Correction sur « masse de fond négligeable à pas=600 » : vraie
mais mauvais dénominateur.** `masse_fond≈5e-4` à pas=600 est bien
`0,2%` de la masse initiale injectée (25%) — MAIS elle est **100×
plus grande** que l'écart initial `s3` (`5e-6`) qu'on essaie de
suivre. Pas assez petite pour être écartée sans vérification — reste
une piste ouverte (protocole 3 de l'agent : suivre la DIFFÉRENTIELLE
`masse_fond_effondre(t) - masse_fond_survit(t)`, pas encore fait).

**Comparaison `λ` à `k∈[1,42;2,45]`/`D≈8,0021`/`C0≈0,2212604` :
refusée à raison, unités incommensurables (même mise en garde que la
fois précédente sur une comparaison similaire) — `λ` est un taux par
pas d'Adam discret, `k`/`D`/`C0` sont des constantes d'ajustement
d'équations différentes, paramétrées par `delta` ou `s3_init`, pas par
le nombre de pas. Pas de ratio à publier.

**Question finale de l'agent, réglée par lecture de code (pas besoin
d'expérience) : `fixer_s3` réinitialise-t-il l'état Adam `(m,v)` de la
case `[3,10]` ?** Vérifié directement dans `verifier_prior_asymetrique.py`
: `continuer_sous_prior` crée `opt = torch.optim.Adam(...)` **APRÈS**
que `fixer_s3`/`fixer_r4`/`fixer_masse_fond` aient modifié les
paramètres — un optimiseur TOUJOURS neuf (déjà vérifié dans ce même
fil pour une question connexe). **Réponse : oui, l'état est
implicitement "réinitialisé" — en réalité il n'existe simplement pas
encore, `exp_avg`/`exp_avg_sq` s'initialisent à zéro pour TOUTES les
cases (y compris `[3,10]`) au premier `step()`, aucun état antérieur
ne peut être "resté" puisque l'optimiseur vient d'être construit.**
Le risque d'artefact d'optimiseur (protocole 4-5 de l'agent) est donc
**réfuté par construction**, pas seulement par une expérience — pas
d'état résiduel possible, point.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| la croissance de `g_e3` est uniformément exponentielle sur toute la fenêtre 0-770 | 18/09 (agent, implicite pour tester) | **réfutée** le 18/09 — `λ_grad` et `λ_state` diffèrent d'un facteur 2,08, signe d'un régime de passage lent suivi d'un régime rapide, pas une seule exponentielle |
| le préfacteur `C` du temps de passage est d'ordre 1, cohérent avec un fantôme de nœud-col | 18/09 (agent) | **confirmée** le 18/09 — `C=0,920` et `0,977`, à 6% près |
| `fixer_s3` pourrait laisser un état Adam résiduel qui fausse la lecture du point selle | 18/09 (agent) | **réfutée** le 18/09, par lecture de code — l'optimiseur est toujours construit APRÈS les `fixer_*`, aucun état préexistant possible |
| la masse de fond résiduelle à pas=600 (`5e-4`) est négligeable face à l'écart initial suivi (`5e-6`) | 18/09 (moi, implicite) | **réfutée** le 18/09 (agent) — elle est 100× PLUS GRANDE, mauvais dénominateur utilisé, piste toujours ouverte |

**Bilan à ce stade : le mécanisme est confirmé être un fantôme de
nœud-col authentique (structure confirmée par le préfacteur `C`), le
risque d'artefact d'optimiseur est définitivement écarté, mais
l'identité exacte du point selle (même que H6, ou nouveau créé par la
masse résiduelle) reste ouverte — protocole précis proposé par l'agent
(overlay temporel avec la trajectoire H6 d'origine, continuation
paramétrique sur la fraction de masse) pour la prochaine reprise.**

**Protocole 3 de l'agent exécuté (18/09, suite immédiate, Théo :
« on continue ») : la DIFFÉRENTIELLE de masse de fond entre les deux
configs, pas juste sa valeur absolue.**

```
pas    diff_s3       diff_masse_fond
  0    -5,000e-06    +0,000e+00
200    +2,029e-06    +3,116e-09
450    +3,144e-05    +1,111e-07
550    +1,337e-04    +3,822e-07
600    +3,207e-04    +8,217e-07
650    +8,571e-04    +1,952e-06
700    +2,767e-03    +5,350e-06
750    +3,328e-02    +2,697e-05
775    +9,497e-01    +2,436e-04
```

**`diff_s3` est SYSTÉMATIQUEMENT 2 à 3 ordres de grandeur plus grand
que `diff_masse_fond`, à chaque pas, pendant toute la phase de
divergence** (ratio `~390` à pas=600, `~518` à pas=700, `~1234` à
pas=750). Les deux différentielles CROISSENT ENSEMBLE, à partir
d'environ pas=450-500 (avant `t_cross=600,283`, pas après) — donc la
masse de fond n'est pas totalement inerte, elle réagit bien avant la
bifurcation elle-même. **Mais elle ne DOMINE jamais : `s3` porte et
amplifie l'essentiel du signal, la masse de fond suit avec un facteur
constant de retard, plutôt que de le devancer ou de le piloter.**

**Lecture du critère précommis par l'agent (protocole 3) : le
différentiel de masse DÉPART bien de ~0 aux alentours de la fenêtre
critique (`±100` pas de `600,283`) — mais son AMPLEUR reste toujours
trop petite pour être le canal causal principal.** Résultat mitigé
mais penchant plutôt vers « même famille de point selle que H6, la
masse résiduelle est un paramètre qui l'influence sans être elle-même
la variable dynamique qui pilote la bifurcation » — cohérent avec (pas
une preuve définitive de) la lecture « point selle déplacé dans le
temps », pas « mécanisme entièrement nouveau ».

| # | hypothèse | posée le | statut |
|---|---|---|---|
| la masse de fond résiduelle est le canal causal principal qui porte le signal (`5e-6`) jusqu'à la bifurcation | 18/09 (agent, protocole 3) | **réfutée** le 18/09 — `diff_masse` toujours 2-3 ordres de grandeur plus petit que `diff_s3`, corrélé mais pas dominant |
| `diff_s3` et `diff_masse` divergent ensemble à partir d'environ pas=450-500, avant le croisement à `600,283` | 18/09 (moi) | **confirmée** le 18/09 |

**Bilan affiné : penche vers « même famille de point selle que H6,
influencée mais pas remplacée par la masse résiduelle » — pas une
preuve complète (le test 1 de l'agent, superposition temporelle avec
la trajectoire H6 d'origine, reste le test le plus direct, pas encore
fait), mais un indice de plus dans ce sens plutôt que dans celui d'un
mécanisme entièrement nouveau.**

**Test 1 de l'agent exécuté (superposition temporelle avec H6
d'origine) — résultat qui COMPLIQUE la lecture précédente, pas qui la
confirme.** Rejoué le test H6 original (`verifier_sonde_bassin.py`
round 2 : `s3` ET `R` placés directement sur le point instable prédit,
`0,994300`/`0,829390`), avec deux points à `±5e-6` du seuil connu
(`0,994295` et `0,994305`), tracés à résolution fine (`check_tous=1`)
dès `pas=0` :

```
H6 direct (cible_s3=0,994295, s'effondre vraiment) :
  pas=0-20  : g_e3 oscille pres de zero (bruit ~1e-6)
  pas=20-59 : g_e3 croit clairement, s3 chute de 0,994 a 0,588 en 59 pas
  facteur de croissance g_e3 (pas 30->50) = 57,85x en 20 pas
  lambda_local = ln(57,85)/20 = 0,2029/pas  (e-folding = 4,93 pas)

H6 direct (cible_s3=0,994305, reste gradue) :
  g_e3 oscille pres de zero tout du long, s3 remonte tranquillement vers 0,999
```

**Même structure qualitative à deux phases (lente puis rapide) que le
cas retardé — MAIS une échelle de temps locale apparemment
différente.** Comparé au taux local du cas retardé (`λ_grad=0,0327`/pas,
e-folding 30,6 pas, calculé par l'agent) :

```
ratio des e-folding (retarde / H6 direct) = 30,58/4,93 = 6,20x plus lent

Test plus dur (meme prefacteur C, meme mu) :
  mu_H6 = 5e-6 (identique en ordre de grandeur au mu du cas retarde, 2,5e-6)
  duree predite pour H6 direct SI meme C=0,95 que le cas retarde :
    tau = C/sqrt(mu) = 0,95/sqrt(5e-6) = 424,9 pas
  duree REELLEMENT observee pour H6 direct : ~20-30 pas
  ECART : facteur 17x
```

**CORRECTION IMMÉDIATE (agent-dipankar, vérifiée indépendamment) : ce
`×17` était en grande partie un ARTEFACT DE DÉFINITION, pas une preuve
de géométrie différente.** Le `C=0,95` avait été calibré avec
`τ=600` — le nombre de pas ÉCOULÉS DEPUIS `t=0` — alors que la fenêtre
où `λ_grad` a réellement été mesurée est `pas=550-760`, soit **210**
pas, pas 600. Ce sont deux définitions différentes de « durée du
goulot » (temps total écoulé vs temps réellement passé dans le régime
mesuré). Recalculé avec la définition COHÉRENTE (`τ=210`) :

```
C' = 210 * sqrt(2,5e-6) = 0,3320
tau_predit_H6_direct = C' / sqrt(5e-6) = 148,49 pas
ratio a la duree observee (25 pas) = 5,94
```

**`5,94`, à comparer au ratio e-folding indépendant `6,20` — les deux
s'accordent à 4,3% près.** Une simple correction de définition fait
passer l'écart de `×17` à `×5,94-6,20` (cohérent entre deux méthodes
de calcul indépendantes). **Vérifié indépendamment (recalcul complet en
Python), tous les chiffres de l'agent confirmés au chiffre près.**

**Deuxième correction, plus profonde : `λ_local=0,2029` (H6-direct)
N'EST PAS une mesure de la linéarisation près du point selle — c'est
`45,4×` AU-DESSUS du plancher théorique.** Forme normale canonique du
fantôme (`dx/dt=μ+a·x²`, `a=1` implicitement supposé) : taux minimal
théorique `λ_min=2√μ`. Pour `mu_H6=5e-6` : `λ_min=0,00447`. Le taux
MESURÉ (`0,2029`) est `45,4×` plus grand — **ma fenêtre de mesure
(pas 30-50) n'était déjà plus dans le régime "près du point selle",
elle mesurait la queue non-linéaire de l'envolée, loin du plancher.**
En sens inverse : le `mu` qui rendrait `0,2029` cohérent avec le taux
MINIMAL théorique serait `0,0103` — **2058× plus grand que
`mu_H6=5e-6`** (la distance statique `s3-0,994300`). Autrement dit,
la distance statique en `s3` seul n'est probablement PAS la bonne
coordonnée de forme normale (le coefficient `a` n'est sans doute pas
`1`, ou `s3` n'est pas la bonne combinaison linéaire des variables du
système).

**Table de sensibilité de l'agent : le "×17" que j'avais publié était
le coin le PLUS EXTRÊME d'une fourchette de `×2,52` à `×16,97`, selon
seulement des choix de comptabilité (quelle définition de `τ`, quelle
fenêtre de durée pour H6-direct) — aucun de ces choix n'est de la
physique.** Publier le pire cas sans le signaler était une erreur de
ma part (règle méfiance : je n'ai pas assez cherché ce qui pourrait
RENDRE MON RÉSULTAT FAUX avant de le publier).

**Conséquence sur le verdict "même famille, point différent" : à
RÉOUVRIR, pas à confirmer.** Les deux lectures (même point avec de
mauvaises unités, vs point réellement différent) restent toutes les
deux possibles avec les données actuelles — **la question n'est PAS
tranchée**, contrairement à ce que j'avais écrit. Protocole de l'agent
pour trancher vraiment (aucun fait cette session) : (1) redéfinir la
fenêtre de `mu_delayed` avec le MÊME critère opérationnel que
H6-direct ; (2) ajuster le coefficient quadratique `a` directement sur
les données brutes `(s3, g_e3)` de chaque run séparément, sans
supposer `a=1` ; (3) recalculer `mu_eff=a·mu_brut` avant toute
comparaison de `C` ; (4) vérifier si `R` dérive AUSSI loin de
`0,829390` pendant les 59 pas de H6-direct (la prémisse « R exactement
sur cible » ne vaudrait alors qu'à `t=0`) ; (5) vérifier le biais de
correction Adam spécifiquement pour une paire "run frais vs run
réchauffé" (jamais testé — mon refus de l'artefact d'optimiseur
concernait une paire différente de trajectoires).

**Synthèse finale RÉVISÉE, toutes preuves pesées (réponse honnête à
« c'est le H6 ? ») — la question N'EST PAS tranchée, contrairement à ce
que j'avais écrit avant la correction ci-dessus :**
- **Même TYPE de mécanisme** (fantôme de nœud-col, structure à deux
  phases) — confirmé, solide.
- **`s3` domine le signal sur la masse de fond résiduelle** (protocole
  3) — cohérent avec « pas un canal causal séparé », plutôt en faveur
  d'un lien avec H6.
- **L'écart de taux/durée (`×17` publié) était en grande partie un
  artefact de définition de `τ` — corrigé à `×5,94`, cohérent à 4,3%
  près avec le ratio e-folding indépendant (`×6,20`).** Ce qui reste
  (`×6` environ) pourrait être un vrai signal de géométrie différente,
  OU un résidu d'une comparaison encore mal posée (le `mu` statique en
  `s3` seul n'est probablement pas la bonne coordonnée — l'écart
  `2058×` entre `mu_H6=5e-6` et le `mu` impliqué par le taux mesuré le
  suggère fortement).
- **Conclusion honnête : NI confirmé NI réfuté.** La parenté
  structurelle (fantôme de nœud-col, même famille) est solide ; mais
  savoir si c'est LE MÊME point précis ou un voisin avec sa propre
  géométrie demande un travail que je n'ai pas fait (ajuster le
  coefficient quadratique `a` sur les données brutes, pas le supposer
  à 1 ; redéfinir `mu`/`τ` de façon cohérente entre les deux runs ;
  vérifier la dérive de `R` dans H6-direct aussi). **Je m'étais arrêté
  trop tôt sur un chiffre confirmant une lecture séduisante — exactement
  le moment où la méfiance doit être la plus haute (règle 5ter),
  et je ne l'ai pas appliquée assez fort la première fois.**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| c'est littéralement le MÊME point selle que H6, juste rencontré plus tard | 18/09 (moi, hypothèse de départ) | **rouverte** le 18/09 — le `×17` qui la réfutait était en grande partie un artefact de définition de `τ`, corrigé à `×5,94-6,20`, insuffisant pour trancher |
| c'est un point selle DE LA MÊME FAMILLE (même forme normale) mais avec sa propre géométrie locale | 18/09 (moi, affinée) | **rétrogradée à ouverte** le 18/09 — plausible mais plus confirmée, la distance statique `mu_H6` s'est révélée être une coordonnée probablement incorrecte (`×2058` d'écart avec le `mu` implicite du taux mesuré) |
| la définition de `τ` (temps écoulé total vs fenêtre de mesure réelle) change matériellement la conclusion d'une comparaison de préfacteur `C` | 18/09 (agent) | **confirmée** le 18/09, vérifiée indépendamment — `×17` devient `×5,94` avec une définition cohérente |
| `λ_local` mesuré pour H6-direct (pas 30-50) est dans le régime linéaire propre du fantôme, près du plancher théorique | 18/09 (moi, implicite) | **réfutée** le 18/09 (agent, vérifié indépendamment) — `45,4×` au-dessus du plancher théorique `λ_min=2√μ`, la fenêtre mesure déjà la queue non-linéaire |

**Étape 2 du protocole exécutée (18/09, suite immédiate, Théo :
« continue ») : ajuster `a` directement sur les données brutes de
vitesse `ds3/dpas` vs `x=s3`, séparément par run, sans supposer `a=1`.**

Premier essai, fenêtre trop large (`s3>=0,97`, mélange la phase
d'évacuation précoce du cas retardé avec la vraie zone de
ralentissement) : sommets absurdes (`x0=0,939`), signes incohérents —
**écarté, pas un résultat**. Corrigé en restreignant par INDICE DE PAS
(pas par valeur de `s3`) à la zone de ralentissement réelle de chaque
run (H6-direct : pas 0-24 ; cas retardé : pas 400-700, excluant
l'évacuation ET l'effondrement final) :

```
H6-direct   (pas 0-24)   : a=-98,51   x0=0,994171  (attendu : 0,994300 -- bon accord)
Cas retarde (pas 400-700): a=-9,70    x0=0,995804
ratio |a_delayed|/|a_H6direct| = 0,0985  (soit a_H6direct ~10x plus raide)
```

**Vérification de robustesse (fenêtres légèrement décalées) — résultat
important et honnête : le fit `a_delayed` est STABLE (`7,56` à `11,76`
sur 3 fenêtres, facteur `×1,56`), mais le fit `a_H6direct` est
INSTABLE (`98,5` à `550,7` sur 3 fenêtres, facteur `×5,6`).** La
fenêtre H6-direct n'a que 20-24 points (contre 115-185 pour le cas
retardé) — trop peu de données dans la zone vraiment linéaire pour un
ajustement quadratique robuste. `mu` (valeur au sommet) est lui aussi
instable pour H6-direct (change de signe entre fenêtres, ordre
`1e-5`), alors qu'il est stable et cohérent pour le cas retardé
(`-4,6` à `-4,8e-6`, toujours négatif).

**Conclusion honnête : un écart de géométrie locale est PROBABLEMENT
réel (même la comparaison la plus conservatrice — plus petit `a_H6direct`
mesuré, `98,5`, contre plus grand `a_delayed` mesuré, `11,76` — donne
encore un facteur `~8,4×`), mais son AMPLEUR PRÉCISE n'est pas
mesurable avec les données actuelles — le fit H6-direct n'est tout
simplement pas assez robuste. Pas un chiffre à publier comme final,
juste une direction (probable différence, ordre de grandeur incertain)
avec la raison technique précise de l'incertitude.**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le coefficient `a` de la forme normale diffère substantiellement entre les deux points selles | 18/09 (moi) | **probable mais pas confirmée avec précision** le 18/09 — écart réel même dans la lecture la plus conservatrice (`~×8,4` minimum), mais le fit H6-direct est trop instable pour un chiffre définitif |
| le fit quadratique de `a_H6direct` est robuste au choix de fenêtre | 18/09 (moi, implicite) | **réfutée** le 18/09 — varie `×5,6` selon la fenêtre (20-24 points seulement, trop peu pour la zone vraiment linéaire) |

**Pour vraiment fermer cette piste (pas fait cette session, faute de
données H6-direct assez denses) : régénérer une trajectoire H6-direct
avec BEAUCOUP plus de points dans la zone linéaire** (plusieurs
`cible_s3` très proches de `0,994300`, ou `check_tous` plus fin sur
les tout premiers pas) pour stabiliser le fit, avant de comparer le
`a` final aux `7,56-11,76` déjà bien établis du cas retardé. Restent
aussi les étapes 1, 4, 5 du protocole de l'agent (fenêtre `mu_delayed`
avec le même critère que H6-direct, dérive de `R` dans H6-direct,
biais Adam run-frais-vs-réchauffé) — aucune faite.

**Deux tentatives supplémentaires pour densifier les données H6-direct
(18/09, suite immédiate) — TOUTES LES DEUX ÉCHOUÉES, chacune pour une
raison précise et instructive, rapportées honnêtement plutôt que
cachées :**

**Tentative 2 : un seul pas d'Adam depuis 21 points frais près du
seuil.** Résultat absurde : sommet à `x0=0,998736`, signe de `a`
inversé. **Diagnostic : avec un état Adam FRAIS (`m=0, v=0`), le
PREMIER pas a TOUJOURS une magnitude ≈ `lr` complet, quel que soit le
gradient réel** (`m̂/√(v̂+eps) → sign(gradient)` dès que
`|gradient|≫eps`, par construction de la correction de biais d'Adam au
pas 1). **Cette mesure n'a jamais capté la vraie vitesse locale — elle
mesurait uniquement `±lr`, écrasant toute l'information de magnitude
du gradient.** Piège réel, pas un bug de code : tout protocole qui
mesure "un seul pas Adam depuis un état frais" pour sonder une
dynamique locale fine est structurellement biaisé de cette façon.

**Tentative 3 : regrouper 10 trajectoires courtes (5 sous le seuil, 5
au-dessus), en sautant les 3 premiers pas de chacune (pour éviter le
biais de la tentative 2).** Résultat encore absurde : sommet à
`x0=1,0052368` — **au-dessus de 1,0, physiquement IMPOSSIBLE pour une
probabilité `s3`.** Diagnostic : mélanger des trajectoires des DEUX
côtés du seuil (certaines qui RÉCUPÈRENT vers la branche graduée
`~0,999`, d'autres qui S'EFFONDRENT) revient à ajuster une seule
parabole sur DEUX comportements globaux qualitativement différents —
la forme normale locale `v=a(x-x0)²+μ` ne décrit correctement le
voisinage d'UN point selle que d'un côté cohérent, pas un mélange des
deux issues.

**Bilan de ces trois tentatives : chacune a révélé un piège
méthodologique réel et distinct (fenêtre trop large / biais du premier
pas Adam / mélange des deux issues), aucune n'a produit un `a_H6direct`
fiable.** Ce n'est pas un échec à cacher — chaque diagnostic est en
soi une leçon utile sur comment (ne pas) sonder une forme normale de
pli via un optimiseur Adam discret. **La comparaison quantitative
précise du coefficient `a` reste donc NON RÉSOLUE** — la voie la plus
prometteuse non essayée : ajuster uniquement sur des trajectoires d'UN
SEUL côté du seuil (toutes collapsantes, ou toutes récupérantes, jamais
les deux), en sautant les tout premiers pas (biais Adam) mais en
restant dans la fenêtre où `|x-x0|` est petite (zone linéaire), ce qui
demanderait un point de départ encore plus proche du seuil que ceux
tentés ici.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| mesurer v(x) via un seul pas Adam depuis un etat frais donne une mesure fiable de la vitesse locale | 18/09 (moi) | **réfutée** le 18/09 — le premier pas Adam a toujours une magnitude ≈lr par construction (correction de biais), indépendamment du vrai gradient |
| regrouper des trajectoires des deux côtés du seuil stabilise le fit quadratique | 18/09 (moi) | **réfutée** le 18/09 — sommet physiquement impossible (`x0>1`), mélange de deux dynamiques globales incompatibles |

**Bilan global de la piste "identité du point selle" pour cette
session : caractérisée en profondeur, correction majeure faite en
cours de route (le `×17` initial était un artefact), mais NON
TRANCHÉE.** Ce qui reste solide : le mécanisme est un vrai fantôme de
nœud-col (structure à deux phases confirmée plusieurs fois), la masse
de fond n'est pas le canal causal dominant (protocole 3), et la durée
du goulot correctement mesurée (ratio `τ`) donne un facteur `~5,94-6,20`
raisonnablement cohérent entre deux méthodes. Ce qui reste ouvert :
la comparaison directe du coefficient `a` (3 tentatives, 3 échecs
méthodologiques distincts, tous diagnostiqués) et les étapes 1, 4, 5 du
protocole de l'agent (fenêtre `mu_delayed` avec le même critère que
H6-direct, dérive de `R` dans H6-direct, biais Adam run-frais-vs-
réchauffé).

Scripts : `verifier_h6_direct_trace.py`, `verifier_forme_normale_pli.py`
(script permanent, trois méthodes tentées et documentées, dont deux
échecs instructifs).

**Reprise du 18/09/2026 (suite, même tour) — étapes 4 et 5 du protocole
enfin exécutées, et une explication MÉCANISTIQUE trouvée, pas juste un
quatrième échec de plus.** Script permanent :
`verifier_biais_adam_fenetre.py`.

**Hypothèse H_biais (standard/académique), posée le 18/09 : la
correction de biais d'Adam (`m̂=m/(1-β1^t)`, `v̂=v/(1-β2^t)`) forme une
enveloppe multiplicative `g(t)=bias1(t)/√bias2(t)` qui varie encore
fortement sur les 24 premiers pas (t=1→24 : `g` passe de 0,316 à un
minimum ~0,15 puis remonte à 0,167, ratio max/min=2,08 sur la fenêtre),
contre une variation bien plus douce sur la fenêtre du cas retardé
(t=400→700 : `g` de 0,574 à 0,710, ratio=1,24) — assez pour expliquer,
à elle seule, l'instabilité de `a_H6direct` sans invoquer une géométrie
différente.**

D'abord vérifiée : la formule `g(t)` a été comparée à l'état RÉEL de
l'optimiseur (`opt.state[...]['exp_avg']`, `['exp_avg_sq']`, `['step']`)
plutôt que prise pour acquise (règle 5bis s'applique à mes propres
dérivations aussi) :

| t | g formule | g mesuré sur l'état réel | écart |
|---|---|---|---|
| 1 | 0,316228 | 0,319788 | 3,6e-3 |
| 5 | 0,172499 | 0,172622 | 1,2e-4 |
| 10 | 0,153189 | 0,153276 | 8,7e-5 |
| 24 | 0,167384 | 0,167456 | 7,2e-5 |
| 50 | 0,222039 | 0,222040 | 8,3e-7 |

Formule **confirmée** (écart décroissant, dominé par `eps` non nul
seulement à `t=1`).

**H_biais testée en déflatant `v(x)` par `g(t_milieu)` avant le fit :
RÉFUTÉE le 18/09/2026 comme explication PRINCIPALE.** La correction ne
stabilise rien — elle aggrave même le symptôme le plus grave : sur les
deux sous-fenêtres pas=1-12 et pas=13-24, le `a` BRUT change déjà de
signe (`-48,6` puis `+323,4`), et le `a` CORRIGÉ aussi (`+1183,7` puis
`+2039,3`, cette fois sans changer de signe mais toujours incohérent
d'un facteur ~1,7). Un vrai coefficient quadratique local ne devrait
pas changer de SIGNE d'une sous-fenêtre à l'autre, biais Adam ou pas —
ce n'est donc pas (seulement) l'enveloppe `g(t)` qui casse le fit.

**Étape 4 (dérive de `R` dans la fenêtre H6-direct) — résultat plus
gros que prévu.** `R` (mesuré comme `r[10,4]`, fixé à 0,829390 par
`fixer_r4` avant le premier pas) :

| cas | fenêtre | amplitude de dérive de R | pas pour cette amplitude |
|---|---|---|---|
| H6-direct | pas 1-59 | 0,829390 → 0,996612 (Δ=0,176) | 59 |
| retardé | pas 400-700 | 0,819258 → 0,832248 (Δ=0,013) | 300 |

**Δ(R) par pas est ~100× plus grand pour H6-direct que pour le cas
retardé** (0,176/59≈3,0e-3/pas contre 0,013/300≈4,3e-5/pas) — et ce dès
le tout PREMIER pas Adam (`R` saute de 0,829390 à 0,843024, `ΔR=0,0136`,
alors que le biais Adam à `t=1` DÉFLATE le pas d'un facteur ~3× : le
saut est donc dû à un vrai gradient important sur `R`, pas à un
artefact de démarrage — élimine une objection immédiate).

**Nouvelle hypothèse non standard, née de ce chiffre (18/09/2026) :
la réduction 1D (`v=f(s3)` seul, `R` traité comme asservi/esclave) est
INVALIDE précisément dans la fenêtre utilisée pour `a_H6direct`, parce
que `R` n'a pas eu le temps de se relaxer sur la variété lente avant
que la fenêtre de fit ne commence.** Testée directement en traçant
`|ΔR/Δs3|` pas par pas (au lieu d'une seule amplitude globale) :

| cas | fenêtre | min | max | comportement |
|---|---|---|---|---|
| H6-direct | pas 1-24 | 0,01 | 1688 | **chaotique, changements de signe multiples** (t=13→14→15 : la direction de `ΔR` change deux fois de suite) |
| retardé | pas 400-700 | 2,82 | 6,17 | **lisse, décroissance monotone, jamais de changement de signe** |

**CONFIRMÉE le 18/09/2026, par deux métriques indépendantes (le
ratio ET l'amplitude absolue de `ΔR`, pour écarter l'objection
« un ratio explose juste quand `Δs3≈0` au dénominateur » — l'amplitude
absolue de `ΔR` est ~100× plus grande indépendamment du ratio).**

**Mécanisme (COMMENT) : la forme fermée quasi-statique suppose `R`
esclave de `s3` (relaxation rapide vers `R_br(s3)`) — une approximation
valide pour une trajectoire qui a DÉJÀ eu le temps de converger dessus
(le cas retardé, après 400 pas), mais pas pour une trajectoire posée à
froid EXACTEMENT aux coordonnées `(s3,R)` prédites par cette même
approximation adiabatique. `R` doit d'abord faire son propre transitoire
de relaxation (non monotone, visible pas=1-20) AVANT que la dynamique
locale de `s3` seule devienne un bon résumé 1D — et ce transitoire
recouvre EXACTEMENT la fenêtre `pas≤24` choisie pour le fit, parce que
c'est aussi la fenêtre où `s3` reste proche de son propre point
d'arrêt apparent.**

**QUAND** : le chaos du ratio commence au tout premier pas (`t=1`,
ratio=61,7) et ne se calme qu'à partir de `t≈21-24` (ratio tombe sous
3) — pile la frontière déjà choisie empiriquement (`pas≤24`) pour
séparer « zone lente » de « fuite catastrophique », mais pour une
MAUVAISE raison présumée jusqu'ici (on croyait que `s3` seul y était
lent ; en fait c'est `R` qui y est encore instable).
**COMMENT** : ci-dessus (relaxation non asservie de `R`).
**POURQUOI** : parce que le point de départ H6-direct est construit à
partir d'une solution FERMÉE (adiabatique), pas d'une trajectoire
Adam réelle qui y aurait naturellement convergé — le cas retardé, lui,
y arrive par une vraie trajectoire de 600 pas, donc `R` y est
automatiquement asservi.
**OÙ** : dans le couplage récepteur→émetteur via `R=r[10,4]`
spécifiquement (déjà localisé au récepteur, cohérent avec les tests
précédents qui avaient écarté un artefact purement émetteur).
**COMBIEN** : facteur ~100× sur l'amplitude de `ΔR`/pas, facteur
~5 ordres de grandeur sur le ratio `|ΔR/Δs3|` (0,01 à 1688) contre
un facteur ~2,2× seulement pour le cas retardé (2,82 à 6,17).
**JUSQU'OÙ** : ce diagnostic (borner `|ΔR/Δs3|` et vérifier l'absence
de changement de signe) devrait être un PRÉALABLE obligatoire avant de
faire confiance à tout futur fit 1D de ce genre sur ce système — pas
seulement pour ce cas précis.
**DEPUIS QUAND** : présent dès `t=1`, donc antérieur à toute tentative
de fit déjà essayée (les 3 échecs précédents étaient TOUS mesurés,
même partiellement, à l'intérieur de cette zone contaminée).
**SUR COMBIEN** : vérifié sur une seule paire de trajectoires
(H6-direct côté effondré `0,994295` / retardé `0,999455`) — pas encore
rejoué sur la trajectoire miroir côté gradué (`0,994305`) ni sur
d'autres `R_init`/`masse_fond` du cas retardé, à faire si ce fil
rouvre.

**Ce que ça change pour la conclusion précédente : ce n'est plus
seulement « 3 méthodes ont échoué pour des raisons distinctes et on ne
sait pas pourquoi c'est si dur » — c'est maintenant « les 3 méthodes
ont échoué pour la MÊME raison de fond : elles tentaient toutes
d'extraire un coefficient local 1D d'un régime qui n'est
structurellement PAS 1D, parce que `R` n'y est pas encore asservi ».**
La comparaison directe de `a_H6direct` contre `a_delayed`, telle que
tentée jusqu'ici, n'est donc pas juste difficile à mesurer
précisément : elle compare deux quantités qui ne sont pas définies
dans le même régime dynamique. Un `a_H6direct` fiable demanderait de
laisser `R` s'asservir D'ABORD (partir d'un point plus loin du seuil,
avec assez de pas pour que `R` relaxe, comme le cas retardé) — piste
non essayée, mise de côté pour une session future si ce fil rouvre.

**Méfiance appliquée à ce résultat lui-même : qu'est-ce qui le
ferait basculer s'il était faux ?** Si le même diagnostic
(`|ΔR/Δs3|` chaotique avec changements de signe) apparaissait AUSSI
dans la fenêtre du cas retardé à une résolution plus fine (pas
testée ici — vérifié seulement tous les 50 pas), la distinction
s'effondrerait. Pas vérifié à grain fin sur le cas retardé — limite
explicite de ce résultat, à corriger avant de le citer comme
définitivement tranché.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| H_biais : l'enveloppe de correction de biais Adam g(t) explique (seule) l'instabilité de a_H6direct | 18/09 (moi) | **réfutée** le 18/09 — la déflation par g(t) n'élimine pas le changement de signe entre sous-fenêtres |
| la formule g(t)=bias1(t)/√bias2(t) prédit exactement le facteur multiplicatif réel appliqué par Adam | 18/09 (moi) | **confirmée** le 18/09 — vérifiée contre l'état interne réel de l'optimiseur, écart <1e-3 dès t=5 |
| R n'est pas asservi à s3 pendant la fenêtre de fit de H6-direct (contrairement au cas retardé), expliquant mécaniquement l'instabilité de a_H6direct par les 3 méthodes précédentes | 18/09 (moi) | **partiellement réfutée / affinée** le 18/09 par un agent-dipankar puis confirmée indépendamment — voir bloc ci-dessous : la VRAIE cause est l'état interne d'Adam (m,v), pas la position de R elle-même |

Scripts : `verifier_biais_adam_fenetre.py`.

**Suite du même tour (18/09/2026) — un agent-dipankar challenge ce
résultat, et sa lecture, bien que plus fine, est CONFIRMÉE par un test
que j'ai rejoué moi-même, pas seulement acceptée sur sa parole (règle
5bis).** Points de son rapport, chacun vérifié indépendamment avant
d'être accepté :

1. **La trajectoire MIROIR (0,994305, reste graduée) est TOUT AUSSI
   chaotique** (agent : 3 changements de signe, ratio max 1653 ; ma
   propre re-mesure indépendante : 2 changements, ratio max 392 —
   chiffres exacts différents, mais même verdict qualitatif). **Ma
   lecture « spécifique à l'effondrement » est donc RÉFUTÉE** — ce
   n'est pas parce que la trajectoire va collapser que R est instable,
   les deux issues montrent le même chaos près du point de départ.
2. **La fenêtre précoce du cas retardé LUI-MÊME (pas 1-24, avant
   d'atteindre son propre goulot vers pas=600) est lisse en SIGNE (0
   changement) mais PAS en amplitude de ratio** (agent : ratio
   221,7-893 ; ma remesure : 221,65-552 — le minimum matche
   quasi-exactement, 221,7 vs 221,65). **Ça réfute l'hypothèse
   concurrente « tout démarrage à froid est bruité »** (ici aussi
   démarrage à froid, mais lisse) ET **ça montre que le ratio
   |ΔR/Δs3| n'est pas le bon diagnostic à lui seul** — le nombre de
   changements de signe de `Δs3` est la métrique robuste, le ratio
   brut explose mécaniquement même dans un régime par ailleurs propre.
3. **Ma propre approximation « ~100× » était fausse — le bon chiffre
   est ×67,53 (par pas), pas ×100.** Vérifié indépendamment (recalcul
   direct : `0,175690/59 = 2,9778e-3` contre `0,013228732/300 =
   4,4096e-5`, ratio exact `67,53`) : le chiffre de l'agent est
   confirmé au chiffre près par mon propre calcul, ma formulation
   imprécise (« ~100× ») est corrigée ici.
4. **Softmax couplé aux 25 autres référents et bruit float64 comme
   explications alternatives : écartés par l'agent sur un simple
   argument d'ordre de grandeur (masse des autres catégories 9-10
   ordres sous l'amplitude de R ; `Δs3`/`ΔR` 12-14 ordres au-dessus
   d'epsilon machine) — PAS revérifié indépendamment par moi (limite
   assumée, faible risque vu l'écart d'ordres de grandeur avancé).**

**Le point décisif du rapport, et le plus surprenant : une jacobienne
2x2 du champ CONTINU (avant Adam) au point H6 donnerait deux valeurs
propres RÉELLES de signe opposé (`+3,226e-6` / `-1,128e-4`, séparation
×35) — un vrai col hyperbolique, pas un régime intrinsèquement non-1D.
Le chaos observé serait donc un ARTEFACT DE DÉMARRAGE À FROID D'ADAM
(m=0, v=0), pas une propriété de l'objectif lui-même.**

**Vérifié indépendamment par une méthode DIFFÉRENTE de celle de
l'agent** (`verifier_saddle_h6_jacobien_independant.py` — différences
finies directement en espace `(s3,R)` via un vrai pas SGD brut sur
TOUS les paramètres, pas la réduction analytique `ds/dz=s(1-s)` sur 2
logits isolés utilisée par l'agent) : **CONFIRMÉ, avec des réserves
honnêtes sur ma propre précision.** Aux réglages les plus fiables
(`h=1e-3`, moins sensible à l'annulation flottante que `h=1e-4`) :
valeur propre "instable" `+2,28e-6` à `+3,75e-6` selon `lr_probe`
(encadre bien le `+3,226e-6` de l'agent), valeur propre "stable"
`-2,17e-4` à `-2,17e-4` (même ordre de grandeur que `-1,128e-4`, environ
2× plus grande en magnitude — écart réel mais pas alarmant vu la
différence de méthode). **Mais à `h=1e-4` mes résultats deviennent
franchement bruités** (la valeur propre "instable" oscille entre
`-4,44e-6` et `+7,08e-5` selon `lr_probe`, changeant même de SIGNE) —
noté explicitement plutôt que caché : ma méthode (différences finies
sur un pas SGD brut de magnitude `lr_probe`, à travers ~1458
paramètres) est plus bruitée que celle de l'agent, mais confirme le
signal central (deux valeurs propres réelles, signes opposés, même
ordre de grandeur) aux réglages où le bruit numérique est sous
contrôle. **Structure de vrai col hyperbolique confirmée par deux
méthodes indépendantes.**

**Correction que l'agent a lui-même signalée après coup (bonne
hygiène, à noter) : le rapport de séparation côté "cas retardé"
(cité comme ×114) dépend d'une convention arbitraire** (comment les 10
référents de fond sont gelés pendant la différentiation numérique — à
leur valeur réellement entraînée à pas=550, ou reforcés à 25%
uniforme à chaque échantillon) — l'autre convention donne ×255, avec
même le signe de `J00` qui change. **Le chiffre côté H6 (×35,
masse_fond=0, donc cette ambiguïté ne s'applique pas) reste solide ;
le chiffre côté cas retardé ne doit PAS être cité avec la même
confiance.**

**Test PRÉCOMMIS par l'agent pour trancher entre "R lui-même est en
cause" et "c'est l'état interne d'Adam" — REJOUÉ MOI-MÊME (pas
seulement pris sur sa proposition), et DÉCISIF.**
(`verifier_injection_moments_adam.py`) : aux coordonnées EXACTES de
H6-direct (s3=0,994295, R=0,829390 — rien ne bouge), comparer un Adam
qui démarre réellement à froid (m=0,v=0,t=0) contre un Adam dont
l'état interne (m, v, t=20) est écrasé pour correspondre au gradient
local réel AVANT le premier pas, sur 24 pas :

```
froid    : 5 changements de signe de Δs3, s3 : 0,994295 -> 0,992400 (non monotone)
injecté  : 0 changement de signe,          s3 : 0,994295 -> 0,995523 (parfaitement monotone)
```

**RÉSULTAT NET, PAS AMBIGU : injecter un état Adam cohérent élimine
TOUT le chaos, sans déplacer R ni s3 d'un iota au départ.** Ça tranche
sans appel en faveur de la lecture de l'agent : **ce n'est PAS la
position de R qui est le problème (elle reste identique dans les deux
cas), c'est l'état interne d'Adam (m,v) qui a besoin de vrais pas pour
se verrouiller sur la direction propre locale d'un col authentique.**
Ma phrase « R n'est pas asservi à s3 » était une corrélation vraie
(R bouge beaucoup dans la fenêtre chaotique) mais PAS la bonne causalité
— confirmé en gardant R fixe et en ne changeant QUE l'état de
l'optimiseur.

**Ce que ça change concrètement pour la piste "identité du point
selle" : la comparaison `a_H6direct` vs `a_delayed` n'était pas
condamnée par une différence de régime dynamique — elle était
condamnée par un ARTEFACT D'OPTIMISEUR CORRIGIBLE.** Le cas retardé
arrive à son goulot après 400 pas réels (Adam déjà verrouillé
gratuitement) ; H6-direct n'a jamais eu cette chance (placé à froid
exactement aux coordonnées prédites). **Recette actionnable pour une
session future, non encore essayée : reproduire l'injection de moments
(comme `verifier_injection_moments_adam.py`) avec `t_injecte` plus
grand et des moments dérivés d'un vrai passage lent (pas juste un seul
gradient capturé), puis refaire le fit quadratique sur cette
trajectoire "réchauffée" — devrait enfin donner un `a_H6direct` stable
et comparable à `a_delayed`.**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le chaos de H6-direct vient de la position de R (pas asservie), pas de l'état de l'optimiseur | 18/09 (moi) | **réfutée** le 18/09 — test précommis (injection de moments Adam à R et s3 INCHANGÉS) : chaos disparaît entièrement (5→0 changements de signe) |
| l'état interne d'Adam (m,v) doit se verrouiller sur la direction propre locale d'un col authentique, indépendamment de la position de R | 18/09 (agent, testé par moi) | **confirmée** le 18/09 — test précommis rejoué moi-même, résultat net (0 changement de signe vs 5) |
| au point H6, le champ continu (avant Adam) a une vraie structure de col hyperbolique (valeurs propres réelles, signes opposés) | 18/09 (agent) | **confirmée** le 18/09, par une méthode indépendante (différences finies SGD brutes en espace (s3,R), pas la réduction analytique de l'agent) — signal net à h=1e-3, bruité à h=1e-4 (noté, pas caché) |
| la trajectoire miroir (côté gradué) est plus stable/moins chaotique que le côté effondré | 18/09 (moi, implicite) | **réfutée** le 18/09 par l'agent, confirmé par ma propre remesure — chaos comparable des deux côtés |
| le rapport de séparation propre ×114 côté cas retardé est un chiffre aussi solide que le ×35 côté H6 | 18/09 (agent, puis corrigé par l'agent lui-même) | **réfutée** le 18/09 — convention-dépendant (×114 ou ×255 selon comment les référents de fond sont gelés), non revérifié par moi |

Scripts (permanents) : `verifier_injection_moments_adam.py`,
`verifier_saddle_h6_jacobien_independant.py`.

**Suite du même tour (18/09/2026, sans nouvelle critique de dipankar —
je creuse seul, règle du fichier) : la recette « réchauffer Adam »
identifiée ci-dessus a été mise à l'épreuve pour EXTRAIRE un
`a_H6direct` chiffré, pas seulement pour prouver la causalité. Résultat
: NÉGATIF, mais informatif — deux essais, deux échecs distincts,
documentés plutôt qu'ignorés.**

**Essai 1 — fenêtre tardive (attendre que le chaos s'éteigne tout
seul, sans injection).** Prédiction directe du mécanisme : si on
laisse Adam tourner plus longtemps depuis le MÊME départ froid, l'état
interne devrait se réchauffer tout seul. **Confirmée en partie** :
`verifier_a_h6direct_fenetre_tardive.py` montre que les changements de
signe tombent à 0 dès pas≥18 (7 à 23 points selon la fenêtre). **Mais
le prix à payer casse l'utilité du fit** : à pas=18-24 (fenêtre encore
étroite), `a=+256` à `+406` — signe INVERSÉ par rapport à `a_delayed`
(`-9,70`), sommet `x0≈0,9927` loin du vrai `x0=0,994300`. Aux fenêtres
plus larges (18-40, 25-40), le sommet dérive encore plus loin
(`x0≈0,925-0,938`) et le fit décrit la pente de fuite catastrophique,
pas le voisinage du col. **Diagnostic : le temps qu'il faut pour que
le chaos s'éteigne tout seul est aussi le temps qu'il faut pour que la
trajectoire quitte la zone locale du col — les deux échelles de temps
ne sont pas séparables ici, contrairement au cas retardé qui a 400 pas
de marge avant son propre goulot.**

**Essai 2 — fitter `a` directement sur la trajectoire à moments
injectés (celle qui a servi au test causal décisif).** Résultat plus
riche mais tout aussi négatif pour l'objectif de mesure :

| `t_injecte` | a (pas 1-24) | x0 | x range |
|---|---|---|---|
| 10 | +25,08 | 0,99377 | [0,994311; 0,995330] |
| 20 | +15,05 | 0,99306 | [0,994312; 0,995478] |
| 30 | +9,66 | 0,99211 | [0,994314; 0,995642] |
| 40 | +3,92 | 0,98816 | [0,994316; 0,995801] |

**`a` dépend fortement et continûment de `t_injecte` — pas un
plateau, une dérive monotone sur tout l'intervalle testé.** Un vrai
coefficient local ne devrait pas dépendre du paramètre de la recette
utilisée pour "réchauffer" l'optimiseur. **Sous-fenêtres de robustesse
pour `t_injecte=20`, encore pire** : pas 1-12 donne `a=-18,7`
(NÉGATIF), pas 13-24 donne `a=+28,2` (POSITIF) — changement de signe
à l'intérieur même de la fenêtre "propre" (0 changement de signe de
`Δs3`, pourtant). **Diagnostic : l'état `(exp_avg=g, exp_avg_sq=g²)`
injecté à partir d'UN SEUL gradient capturé n'est pas un proxy fidèle
d'un vrai historique de 20 pas — feule la dépendance systématique à
`t_injecte` (qui ne devrait rien changer si la recette était fidèle)
le prouve directement.** Le test d'injection reste un bon outil CAUSAL
(isoler ce qui cause le chaos brut) mais n'est PAS un bon outil de
MESURE (le nombre qu'il produit dépend de sa propre recette, pas
seulement de la géométrie locale réelle).

**Nouvelles hypothèses sur POURQUOI aucune méthode ne converge (posées
le 18/09/2026, à tester si ce fil rouvre) :**

| # | hypothèse | type | statut |
|---|---|---|---|
| la réduction 1D (v=f(s3) seul) reste invalide même une fois le chaos brut d'Adam éliminé, parce que R continue de bouger de façon significative dans toutes les variantes essayées | standard | ouverte |
| l'injection synthétique (m=g,v=g²) est un proxy trop grossier d'un vrai historique de gradients — un vrai réchauffement (pas juste 1 gradient dupliqué) donnerait un a stable | standard | ouverte, plausible vu la dérive monotone avec t_injecte |
| le passage au col pour H6-direct (masse_fond=0) est intrinsèquement trop RAPIDE (quelques dizaines de pas max) pour qu'aucune fenêtre ne sépare "assez de pas pour lisser" de "assez proche de x0 pour rester locale" — contrairement au cas retardé qui a ~400 pas de marge | non standard | ouverte, cohérente avec l'essai 1 |
| 24-40 points ne suffisent pas pour un fit à 3 paramètres près de x≈1 (mauvais conditionnement numérique d'une quasi-Vandermonde) | standard | non testée |

**Bilan honnête de ce sous-fil à ce stade : le MÉCANISME est
maintenant solide (col hyperbolique réel, artefact de démarrage à
froid confirmé deux fois indépendamment), mais la MESURE chiffrée de
`a_H6direct` reste hors de portée de toutes les méthodes essayées
(6 maintenant, comptant celle-ci). Ce n'est plus « je ne sais pas
pourquoi c'est dur » — c'est « je sais précisément pourquoi chaque
tentative échoue, et la piste la plus prometteuse restante (un vrai
réchauffement multi-pas, pas une injection à 1 gradient) n'a pas
encore été essayée ». Pas soumis à un nouvel agent-dipankar pour ce
résultat précis (volume déjà élevé de rounds ce tour) — limite
assumée, à faire si ce fil rouvre.**

**Essai 3, même tour (18/09/2026), Théo : « on continue une dernière
fois » — la piste "la plus prometteuse" ci-dessus (vrai réchauffement
multi-pas) a été essayée. ÉCHEC, pour une TROISIÈME raison distincte,
elle aussi diagnostiquée.** `verifier_a_h6direct_rechauffement_reel.py` :
au lieu de dupliquer un seul gradient, recalcule un VRAI gradient à
chaque pas de réchauffement (position re-fixée exactement sur la
cible après chaque pas, donc `m,v` accumulent une vraie récursion EMA
d'Adam sur une séquence de gradients réels, pas une copie).

**Résultat : pire, pas mieux.** `a` reste instable ET change de signe
selon `n_chauffe` (`-20,9` à `n=5`, `-7,6` à `n=10`, `-5,1` à `n=20`,
`+2,5` à `n=40`, `+1,2` à `n=80`, `+0,31` à `n=160`) — aucun plateau.
`x0` devient même **physiquement impossible** (`>1`) pour `n_chauffe`
∈ {10,20}. Et le nombre de changements de signe reste bloqué à 3 pour
TOUTE valeur de `n_chauffe` testée (contre 0 pour l'injection à un
seul gradient) — cette recette est donc STRICTEMENT PIRE que l'essai
précédent sur le critère même qu'elle visait à améliorer.

**Pourquoi (COMMENT) : parce que le gradient recalculé est IDENTIQUE à
chaque pas de réchauffement (position remise exactement au même point
à chaque fois, système déterministe, pas de bruit) — `m` et `v`
convergent donc vers `m≈g`, `v≈g²` en quelques pas seulement, et la
correction de biais fait alors que `m̂/√v̂≈sign(g)` : Adam se met à
appliquer un pas de magnitude `lr` COMPLET, dans la même direction, à
CHAQUE pas de réchauffement (pas de décroissance). Plus `n_chauffe`
est grand, plus l'optimiseur est "certain" de vouloir prendre un pas
maximal dans cette direction — et une fois relâché, il fonce à travers
toute la zone locale en quelques pas au lieu de ralentir dessus (`s3`
tombe à 0,868 en seulement 24 pas pour `n_chauffe=160`, contre 0,988
pour `n_chauffe=5`). **Le réchauffement sur point fixe apprend le
mauvais invariant : "ce gradient ne change jamais" au lieu de "voici
la vraie courbure locale", parce que la vraie courbure ne peut
s'observer qu'en laissant la position RÉELLEMENT bouger — exactement
ce que le cas retardé fait (400 pas de vraie trajectoire), et
qu'aucune recette de réchauffement sur point fixe ne peut imiter.**

**Bilan à ce stade (3 recettes de réchauffement essayées, 3 échecs
distincts et compris) : le seul chemin qui reste plausible est de
construire une VRAIE trajectoire d'approche lente pour le cas
masse_fond=0 (partir d'un point suffisamment loin pour que la
dynamique ait le temps de se stabiliser AVANT d'atteindre le
voisinage du col, comme le fait naturellement le cas retardé) — pas
une astuce de réchauffement d'optimiseur sur place. C'est un vrai
travail de modélisation (trouver/construire un point de départ qui
funnelle naturellement vers 0,994300 sur assez de pas), pas un script
de plus. Mis de côté comme travail futur plutôt que retenté sans fin
dans ce tour — la piste `a_H6direct` est refermée ici pour cette
session, avec une carte complète et honnête de six échecs diagnostiqués.**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| un réchauffement Adam sur VRAIS gradients répétés (pas une injection à un seul gradient dupliqué) donnerait un a stable | 18/09 (moi, motivée par les résultats précédents) | **réfutée** le 18/09 — pire que l'injection à un seul gradient sur le critère même qu'elle visait (3 changements de signe pour toute valeur de n_chauffe testée, contre 0) |

Scripts (permanent) : `verifier_a_h6direct_rechauffement_reel.py`.

**Piste 3a (18/09/2026, Théo : « commence par 3a, regarde si on ne l'a
pas fait avant ») — jamais commencée avant cette session (vérifié par
grep sur tout `src/test3_communication/` : seul l'ODE du SYSTÈME RÉEL,
M=0 implicite, existait). Construite ici, avec un vrai va-et-vient
d'auto-correction et d'agents.**

**Fonctions de branche dérivées pour le jouet à M catégories de fond**
(`verifier_ode_jouet_m.py`) : émetteur en sigmoïde binaire (pas un
softmax à 27 voies comme le système réel — bug trouvé et corrigé EN
TOURNANT le script une première fois, `x_br` réutilisait à tort la
forme du système réel) :

```
x_br(R)   = 1 - sigmoid(N*poids3*(1-R)/beta)
R_br(x;M) = e4 / (e3(x) + e4 + M)
```

Sanité M=0 vs forme fermée du système réel : écart <1e-15, formule
`R_br` validée dans son cas limite.

**Premier calcul du pli (comptage de racines sur grille fixe) : FAUX
DE 40%, trouvé par un agent-dipankar.** `delta_c(M)` sortait identique
à 16 décimales pour tout M (0,011151...) — un premier agent a montré
(vérifié indépendamment en mpmath 50 chiffres, confirmé au chiffre
près) que le vrai pli est à `x*≈0,0042146`, `delta_c≈0,018699092479`,
et que ma grille (pas ~5e-6 sur [0,1]) était ~40× trop grossière pour
voir un pli logé dans une fenêtre de cette taille — un artefact de
grille stable, pas une vraie insensibilité à M.

**Le chiffre de sensibilité à M de l'agent lui-même (2e-11) était
FAUX aussi — revérifié indépendamment, agent challengé une deuxième
fois sur ce point précis.** En gardant tout le calcul en mpmath
(jamais de conversion float64 avant la soustraction finale, l'erreur
exacte que je venais de faire une fois moi-même avec la fausse
alerte des trois amplitudes différentes, §7.65 plus haut) : l'écart
relatif réel entre `delta_c(M=25)` et `delta_c(M=0)` est **8,002e-21**,
cohérent avec la borne théorique `M/(e3+e4)=1,68e-21` à un facteur
~4,8 près — PAS `2e-11`, dix ordres de grandeur d'écart avec le
chiffre de l'agent, qui avait lui-même perdu la précision dans sa
propre soustraction. **Mon diagnostic initial (M négligeable sur le
vrai pli, par ~21 ordres de grandeur) tenait — c'est la vérification
de l'agent qui ne tenait pas, pas la mienne.**

Script corrigé (`localiser_pli` par tangence Newton au lieu du
comptage de grille) : `delta_c(M)=0,018699092479043` pour tout
M∈{0,1,3,8,25} testé, points de bascule ODE identiques bit-à-bit
entre M=0 et M=25 (résultat maintenant ATTENDU — 8e-21 est ~5 ordres
sous le plancher de précision float64 ~1e-16, pas un bug).

**LE TEST DÉCISIF — l'hypothèse « budget de convergence fini »
RÉFUTÉE le 18/09/2026.** Le modèle quasi-statique dit M négligeable ;
une expérience dynamique antérieure (entraînement réel) avait mesuré
un vrai décalage de -0,84% entre `delta_c(M=0)` et `delta_c(M=25)`.
Hypothèse : cet écart est un artefact de budget fini (motif déjà
rencontré plusieurs fois cette session — ralentissement critique,
cible mobile). **Test précommis : rejouer la bissection dynamique à
5× le budget (40000→200000 pas).**

```
pas=40000   delta_c(M=0)=0,018711  delta_c(M=25)=0,018555  shift=-0,835%
pas=200000  delta_c(M=0)=0,018711  delta_c(M=25)=0,018555  shift=-0,835%  (IDENTIQUE)
```

**Le décalage ne bouge pas d'un chiffre significatif à 5× le budget.
L'hypothèse budget-fini est RÉFUTÉE.** Le -0,835% est un vrai
phénomène dynamique, reproductible, que la réduction quasi-statique à
2 variables (x, R) ne capture PAS. Un troisième agent est en cours
pour challenger ce résultat et la piste la plus probable pour le
canal manquant (l'hypothèse `s4=1` saturé, empruntée du système réel,
pourrait être fausse spécifiquement sous masse de fond dans ce
jouet — pas encore vérifiée directement).

| # | hypothèse | posée le | statut |
|---|---|---|---|
| M a un effet négligeable sur le pli quasi-statique (~21 ordres de grandeur sous e3+e4) | 18/09 (moi) | **confirmée** le 18/09, deux fois (mpmath direct + agent challengé et re-vérifié sur son propre chiffre de sensibilité) |
| le décalage empirique -0,84% de delta_c(M) est un artefact de budget de convergence fini | 18/09 (moi) | **réfutée** le 18/09 — identique à 5x le budget (40000 vs 200000 pas) |
| s4 (challenger) reste saturé à ~1 sous masse de fond, comme dans le système réel | 18/09 (moi, hypothèse pour le canal manquant) | **réfutée** le 18/09 — s4=1,0000000000 à 10 décimales, M=0 et M=25, de 90% à 99% de delta_c, ce n'est pas le canal manquant |

**Suite, même tour — le -0,835% publié était lui-même gonflé
d'environ 25% par la résolution de MA PROPRE bissection, pas un
artefact de M.** Un 3e agent, challengé sur le canal manquant, a
trouvé la vraie faille avant de la chercher où je pensais : l'erreur
relative plancher d'une bissection à seuil `tol` est
`ε_tol=(tol/2)/delta_c` — `ε_{3e-4}=0,802%`, DU MÊME ORDRE que l'effet
rapporté (0,84%). Ma mesure tournait au plancher de résolution de son
propre outil. Recalculé à `tol=1e-5` (30× plus fin, `ε_{1e-5}=0,027%`)
: **-0,6275%**, pas -0,835% — **REJOUÉ MOI-MÊME, confirmé au 4e
chiffre significatif** (`delta_c(M=0)=0,01867676`,
`delta_c(M=25)=0,01855957`, 946s de calcul, chiffres identiques à
ceux de l'agent).

**L'effet SURVIT à la correction — -0,6275% reste très au-dessus du
plancher de bruit de cette mesure plus fine (0,027%) — donc ce n'est
PAS un pur artefact de bissection non plus : c'est un vrai effet,
juste plus petit de ~25% que ce qui était publié.** Le canal manquant
de la réduction quasi-statique (qui prédit M négligeable à 21 ordres
de grandeur) reste non identifié — s4 est réfuté, `masse_autres`
reste stable (~4,09e-11) pendant toute l'approche du pli (pas
seulement au point fixe final), donc l'hypothèse d'une asymétrie de
fond en transitoire est elle aussi affaiblie. Protocole précommis par
l'agent pour la suite (non lancé, coûteux — chaque point ~15 min) :
comparer le décalage à `lr=0,2` contre `lr=0,05` — s'il varie avec
`lr`, c'est un artefact Adam ; s'il reste proche de -0,63% quel que
soit `lr`, c'est un vrai canal manquant indépendant de l'optimiseur.

**Bilan honnête de la piste 3a pour cette session : construite de
zéro (jamais commencée avant), un bug de grille de 40% corrigé, un
chiffre de sensibilité d'agent revérifié et corrigé (2e-11→8e-21), un
chiffre publié corrigé de 25% par ma propre vérification rejouée, et
un canal manquant restreint (s4 exclu) mais pas encore identifié.**
Bon point d'arrêt : le mécanisme algébrique (M négligeable) est
solide, l'effet dynamique résiduel (-0,63%) est maintenant mesuré
proprement, et le prochain pas (test `lr`) est précommis et prêt à
lancer si ce fil rouvre.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le décalage -0,84% publié est correct en grandeur, pas seulement en signe | 18/09 (moi, implicite) | **réfutée** le 18/09 — gonflé de ~25% par la résolution de bissection (ε_tol du même ordre que l'effet) ; vrai chiffre -0,6275%, revérifié indépendamment |
| après correction du tol, le décalage résiduel (-0,63%) est un pur artefact de bissection | 18/09 (agent) | **réfutée** le 18/09 — -0,6275% reste ~23× au-dessus du plancher de bruit à tol=1e-5 (0,027%), un vrai effet subsiste |

Scripts (permanents) : `verifier_ode_jouet_m.py`.

**Piste 3b (18/09/2026, Théo : « 3b ») — la géométrie locale du pli,
résolue en réutilisant directement les fonctions de branche de 3a.**
Coefficient de courbure `a` (forme normale `dx/dt=μ+a(x-x*)²`,
`a=F''(x*)/2`) calculé pour M∈{0,1,3,8,25} au pli localisé en 3a :
**identique à M=99,064645 pour tout M testé** (stable h∈[1e-6,1e-4]).
**Même conclusion que pour la position (3a) : la géométrie locale du
vrai pli algébrique — position ET courbure — ne dépend pas de M.**

**Le "6,5 ordres de grandeur" qui motivait cette piste ne se
reproduit PAS — trouvé par un agent, confirmé par moi en rejouant le
script tel quel.** `verifier_point_fixe_jouet_m.py` (inchangé) donne
aujourd'hui un ratio de pente M=0/M=25 de **39,50 (1,60 décade)**, pas
6,5. Vérifié indépendamment : rejoué moi-même, ratio=39,4955,
identique à 3 chiffres significatifs. **Et le git log règle la
question en une commande** : le commit original (`fe737ef`) porte lui
même le message *« écart 6 ordres de grandeur à 3000 pas, mais fenêtre
adaptative de M=25 pas encore propre, relance plus longue »* — un
avertissement que J'AVAIS DÉJÀ ÉCRIT MOI-MÊME et qui s'est perdu en
route jusqu'à ETAT.md, où il a fini cité comme un fait propre
("6,5 décades"). **Mécanisme exact, montré par l'agent puis vérifié
(fenêtre adaptative de M=25 : `[143,3001)`, ne se ferme JAMAIS avant
la fin du budget — contre `[190,322)` pour M=0, fermée bien avant)** :
la "pente" mesurée pour M=25 n'est pas une vitesse de relaxation
asymptotique près du pli — c'est une moyenne glissante prise EN PLEIN
TRANSITOIRE D'ÉVACUATION DE MASSE, tronquée par le budget, pas par la
convergence. Confirmé par la trace de `masse_autres(t)` (rejouée
indépendamment) : encore 4,8e-6 de masse sur le fond à t=3000, cinq
ordres de grandeur au-dessus de la valeur d'équilibre réelle
(~4e-11), avec un ralentissement du taux de décroissance entre
t=1000 et t=3000 suggérant un second mode plus lent qui prend le
relais.

**Conclusion pour 3a+3b ensemble : le pli algébrique quasi-statique
(position, courbure) ne dépend pas de M, point final — chaque
différence empirique attribuée à M (décalage de seuil, vitesse de
convergence) s'est révélée être un artefact du budget/de la
résolution de MESURE, pas une propriété du pli lui-même.** Ce qui
reste réel et non expliqué : un effet dynamique résiduel de -0,6275%
sur `delta_c` (piste 3a, confirmé au-dessus du bruit) et un temps
d'évacuation de la masse de fond visiblement multi-échelle (piste 3b,
pas encore caractérisé proprement — fenêtre trop courte pour le
voir converger). Les deux pointent vers le même type de mécanisme :
quelque chose hors de la réduction (x,R) à 2 variables, probablement
lié à la dynamique du fond lui-même (les M catégories), pas à
l'émetteur.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le coefficient de courbure a du pli dépend de M | 18/09 (moi, motivation de 3b) | **réfutée** le 18/09 — identique (99,064645) pour tout M, confirmé stable en h par un agent |
| l'écart de "6,5 décades" en vitesse de convergence (ETAT.md) est un chiffre propre et reproductible | 18/09 (moi, implicite) | **réfutée** le 18/09 — ne se reproduit pas (39,50=1,60 décade en rejouant le script tel quel), le chiffre d'origine portait déjà sa propre mise en garde (commit fe737ef) perdue dans le résumé ETAT.md |
| la pente M=25 mesurée à budget=3000 est une vraie vitesse de relaxation post-transitoire | 18/09 (moi, implicite dans le script d'origine) | **réfutée** le 18/09 — la fenêtre adaptative ne se ferme jamais avant la fin du budget pour M=25, donc c'est une moyenne en plein transitoire d'évacuation, pas une pente asymptotique |

Scripts (permanents) : `verifier_ode_jouet_m.py`, agent a produit
`verifier_evacuation_masse_jouet_m.py` (pas encore rapatrié dans le
dépôt, dans le worktree de l'agent).

**Suite, même tour (Théo : « continue à chercher ») — le test précommis
`lr=0,05` vs `lr=0,2` pour trancher artefact-Adam vs vrai canal
manquant, enfin exécuté.** Résultat : `lr=0,05` donne un décalage de
**-1,0447%** (contre -0,6275% à `lr=0,2`) — **le décalage DÉPEND bien
significativement de `lr`** (facteur ~1,67×), au sens du critère
précommis par l'agent précédent. Un nouvel agent, challengé
là-dessus, a réfuté mon explication naïve (« lr petit = moins de
progrès à budget fixe = pire convergence ») en mesurant l'écart au
point fixe algébrique : à M=0, `lr=0,05` est en fait **1500× MIEUX
convergé** que `lr=0,2` (2,6e-11 contre 3,9e-8), pas pire — mon
mécanisme était faux dans le sens le plus défavorable à ma lecture.
Sa lecture alternative, plus fine : `bissecter_delta_c` ne mesure
jamais un résidu de convergence, il mesure de quel côté de la
séparatrice `s3` tombe à budget fixe — un phénomène de TEMPS DE
FRANCHISSEMENT DU COL (ralentissement critique), pas de résidu final.

**Son propre test précommis (bon marché, 4 trajectoires courtes) a
révélé une VRAIE erreur dans son propre rapport, trouvée en le
rejouant moi-même — règle 5bis, encore.** Sa fenêtre de test
`s3∈[0,45;0,55]` supposait le col près de `s3=0,5` — mais le pli
calculé en 3a est à `x*≈0,0042`, donc `s3*≈0,9958`, pas 0,5. Fenêtre
corrigée (`[0,990;0,999]`) : **M=25 passe 11 à 13× MOINS de temps
dans la région lente du vrai pli que M=0** (371-393 pas contre
4293-4797 pas) — un vrai effet dynamique, distinct de la géométrie
statique (identique, 3a/3b). Mais en recalculant ensuite le résidu au
point fixe de l'agent pour vérifier sa table, **ses chiffres pour
M=25 (`s3=0,9972781760/0,9972781724`) NE SE REPRODUISENT PAS** :
rejoué deux fois moi-même, résultat bit-identique aux deux essais,
`s3=0,4997803416` — pas 0,9973. **Le tableau central du rapport de
cet agent contient une vraie erreur sur M=25**, probablement une
mauvaise valeur transcrite ou un mélange avec les chiffres de M=0
(qui, eux, se reproduisent exactement : `s3=0,9964323591/
0,9964323198`, identiques aux siens).

**Ce que `s3=0,5` pour M=25 veut dire, une fois qu'on lit le
docstring de `bissecter_delta_c`** (déjà écrit AVANT cette session,
pas une découverte nouvelle) : ce jouet a un attracteur "effondré"
lent précisément à `s3=0,5`, documenté comme prenant plus de
40000-100000 pas à se résoudre à `lr=0,05`. `delta_c(M)` est PAR
CONSTRUCTION le point milieu d'un bracket de bissection — training
exactement dessus revient à s'asseoir sur la séparatrice elle-même,
où ce genre de blocage est attendu, pas anormal. **M=0 se résout
proprement (vers son point fixe algébrique) tandis que M=25 reste
bloqué à `s3=0,5` aux deux `lr` testés** — une vraie différence
qualitative entre M=0 et M=25 pile à leur propre seuil, mais dont le
mécanisme (pourquoi M=25 spécifiquement tombe dans cet attracteur
documenté alors que M=0 n'y tombe pas) n'est PAS élucidé ici.

**Bilan honnête, fin de session sur cette piste : le décalage
`delta_c(M)` dépend réellement de `lr` (confirmé), la géométrie
statique du pli n'en dépend pas (confirmé deux fois), et le
mécanisme dynamique réel implique un attracteur `s3=0,5` DÉJÀ
CONNU dans ce jouet (pas une nouveauté) dont l'interaction précise
avec M reste ouverte — un vrai sujet pour une session future, avec
un protocole déjà esquissé (tracer `s3(t)` densément autour de
`s3=0,5` pour M=0 et M=25, comparer les temps de résidence).** Pas
soumis à un nouvel agent pour cette dernière correction (volume déjà
très élevé de rounds ce tour) — limite assumée.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le décalage -0,6275%/-1,0447% dépend significativement de lr | 18/09 (moi) | **confirmée** le 18/09 — facteur ~1,67× entre lr=0,05 et lr=0,2, mesuré directement |
| l'explication "lr petit = moins convergé à budget fixe" pour ce décalage | 18/09 (moi) | **réfutée** le 18/09 par un agent — à M=0, lr=0,05 converge 1500× MIEUX que lr=0,2, sens opposé à l'hypothèse |
| les chiffres de résidu M=25 rapportés par cet agent (s3≈0,9973) sont corrects | 18/09 (agent) | **réfutée** le 18/09 — rejoué deux fois moi-même, résultat bit-identique s3=0,4998, pas 0,9973 ; erreur reelle dans le rapport de l'agent |
| M=25 passe moins de temps dans la région lente du vrai pli (x*≈0,0042) que M=0 | 18/09 (moi, fenêtre corrigée) | **confirmée** le 18/09 — 11-13× moins de temps, mesuré directement, mais l'attracteur s3=0,5 où M=25 finit par se bloquer est un phénomène déjà documenté dans le jouet, pas nouveau |

**Suite, même tour (Théo : « on continue, je n'aime pas rester sans
réponse ») — mécanisme final trouvé, propre, vérifié, cette fois
sans laisser de question ouverte.**

**Trace dense `s3(t)` pour M=0 et M=25, chacun à son propre
`delta_c`, jusqu'à 40000 pas** (`lr=0,2`) : les deux atteignent une
région proche de leur pli respectif très vite (`t≈300-500`), mais
divergent radicalement ensuite :

```
M=0  :  s3 se fixe a 0,99643232 des t=500 et y reste jusqu'a t=40000
M=25 :  s3 traverse la meme region puis tombe et se fixe a 0,50000
        des t~700, y reste (avec un bruit residuel) jusqu'a t=40000
```

**`s3=0,5` pour M=25 n'est PAS un attracteur séparé et mystérieux —
vérifié directement : `x_br(R)` (la fonction de branche de 3a) suit
`s3` mesuré à chaque instant, quasi exactement** (`x_br(r4)=0,49999998`
contre `s3=0,50000001` mesuré à `t=20000`). **Le mécanisme complet :
pour M=25, le récepteur (`r4`) file jusqu'à `R=1` exactement
(committement total au challenger, `r3→0`), au lieu de se stabiliser
sur la valeur intermédiaire de la branche graduée stable comme le
fait M=0.** Et `x_br(R=1)=1-sigmoid(0)=0,5` exactement, par
construction de la formule — donc `s3=0,5` est juste `x_br` évalué à
l'extrême `R=1`, un point parfaitement ordinaire de la MÊME fonction
de branche déjà validée en 3a, pas un nouveau phénomène.

**Ce que ça élucide, enfin proprement : le canal manquant de la
réduction (x,R) n'est pas dans `x_br` ni dans la position/courbure
du pli (les deux confirmés indépendants de M, 3a+3b) — il est dans
LA STABILITÉ DE LA BRANCHE GRADUÉE ELLE-MÊME côté récepteur.** M=0
converge vers un `R` intermédiaire stable (la vraie branche graduée) ;
M=25 déstabilise cette branche et pousse `R` jusqu'à saturation
complète (`R=1`), un point différent bien que gouverné par la même
fonction `x_br`. **Cohérent avec la construction du jouet** (M est
strictement côté récepteur, jamais vu par l'émetteur) — la fonction
de branche de l'émetteur reste inchangée, c'est la DYNAMIQUE du
récepteur (quel `R` il choisit d'atteindre) qui change avec M. Piste
concrète pour trancher le POURQUOI si besoin un jour : comparer le
gradient du récepteur en `R` proche de sa valeur de branche graduée
(pas à `R=1`) entre M=0 et M=25, pour voir si M change le signe/la
magnitude de la force de rappel qui, pour M=0, empêche `R` de
dépasser sa valeur stable.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| s3=0,5 (M=25) est un attracteur séparé, non lié à la fonction de branche x_br déjà établie | 18/09 (moi, implicite) | **réfutée** le 18/09 — x_br(r4) colle exactement à s3 mesuré à chaque instant, s3=0,5=x_br(R=1) exactement |
| M déstabilise la branche graduée côté récepteur (R file jusqu'à 1 au lieu de se stabiliser à sa valeur intermédiaire) plutôt que de changer la géométrie du pli | 18/09 (moi) | **confirmée** le 18/09 — R=1,00000000 exact pour M=25 à t=20000-40000, contre R intermédiaire stable pour M=0 |

**Bilan final de la piste 3 (masse de fond) pour cette session : le
pli algébrique quasi-statique (position, courbure) ne dépend PAS de
M — confirmé trois fois. Ce qui dépend de M, et l'explique
proprement : la stabilité dynamique de la branche graduée côté
récepteur, qui se déstabilise vers une saturation complète (R=1)
plutôt que de rester sur son point d'équilibre intermédiaire. Fil
refermé avec un mécanisme complet, pas juste un constat.**

Scripts (tracés jetables, à rapatrier si besoin futur — pas fait ce
tour, volume déjà élevé) : traces `s3(t)`/`r4(t)` reproductibles
directement depuis `entrainer_toy_m`/`construire_toy_m` de
`verifier_jouet_n_variable.py` et `x_br` de `verifier_ode_jouet_m.py`.

**Dernier pas, même tour (Théo : « continue ») — POURQUOI la
réduction (x,R) ne voit pas ce canal, élucidé structurellement.**
Piste suggérée en fin d'entrée précédente : comparer la stabilité
locale (dérivées de branche) à la valeur R de la branche graduée
STABLE (pas seulement au pli instable) entre M=0 et M=25. Racine
stable localisée (`x_s≈1,141e-3`, `s3_s≈0,998859`, `R_s≈0,862052`) et
dérivées `x_br'(R_s)` et `R_br'(x_s;M)` calculées : **identiques à 15
chiffres significatifs pour tout M∈{0,1,3,8,25}.** Donc la réduction
(x,R) n'est pas seulement M-indépendante AU pli — elle l'est
PARTOUT (pli ET branche stable). **Conclusion structurelle : le
canal manquant ne peut PAS être dans cette réduction à 2 variables,
quel que soit le point qu'on y sonde — il faut une vraie 3e
variable qu'elle omet par construction.** Candidat naturel, déjà
mesuré dans ce même tour (traces `masse_autres(t)`) : la masse de
fond elle-même reste non négligeable (`>1e-6`) jusqu'à `t≈5000-8000`
— exactement la fenêtre où `R` évolue de son départ vers `R=1`. Tant
que cette masse n'a pas drainé, l'hypothèse « `r3+r4≈1` » qui sous-
tend `R_br(x;M)` est fausse EN TRANSITOIRE (elle n'est vraie qu'À
L'ÉQUILIBRE) — la réduction (x,R) suppose R toujours à son
quasi-équilibre instantané, une hypothèse qui s'effondre précisément
pendant la fenêtre où le sort de la branche se joue. **Le canal
manquant est donc, structurellement, la dynamique propre de la masse
de fond pendant le transitoire — pas un défaut de `x_br`/`R_br`
eux-mêmes (tous deux valides à l'équilibre, vérifiés multiples fois),
mais une hypothèse de quasi-stationnarité qui ne tient pas assez
longtemps pour M>0.** Piste complète pour une session future si
besoin : ajouter une 3e équation (`d(masse_autres)/dt`) à l'ODE de
3a et refaire l'analyse de stabilité en 3D plutôt qu'en 2D — pas fait
ce tour (vrai travail de modélisation neuf).

| # | hypothèse | posée le | statut |
|---|---|---|---|
| la structure locale (dérivées de branche) à la branche graduée STABLE dépend de M, comme au pli | 18/09 (moi) | **réfutée** le 18/09 — identique à 15 chiffres significatifs pour tout M, confirme que la réduction (x,R) est M-indépendante partout |
| le canal manquant est structurellement hors de la réduction (x,R), lié à la dynamique transitoire de la masse de fond elle-même | 18/09 (moi) | ouverte mais bien étayée — cohérente avec la fenêtre temporelle mesurée (masse de fond non négligeable jusqu'à t≈5000-8000, exactement quand R évolue vers sa valeur finale) ; pas prouvée formellement (nécessiterait l'ODE à 3 variables) |
| le comportement M-dépendant (R file jusqu'à 1 pour M=25, se stabilise pour M=0) est une propriété du flot de gradient de l'objectif (pas d'Adam) — testable en SGD pur | 18/09 (moi) | **test inconclusif** le 18/09 — à `lr=0,02`, SGD pur laisse `s3` bloqué à 1,00000000 pendant 100000 pas pour M=0 ET M=25 (l'émetteur ne quitte jamais la saturation à ce taux, cohérent avec un fait déjà établi ailleurs dans ce projet : Adam nécessaire pour bouger une sigmoïde saturée), donc la région du pli n'est jamais atteinte — le test ne pouvait pas répondre à la question posée avec ce réglage |

**Bilan final, cette fois pour de vrai, de la piste 3 (masse de fond)
pour cette session.** Six sous-questions tranchées avec mécanisme
(pas juste constat) : position du pli (M-indépendante), courbure du
pli (M-indépendante), stabilité locale à la branche graduée
(M-indépendante), le "6,5 décades" d'origine (ne se reproduit pas,
retracé à sa source), le décalage `delta_c` empirique (réel mais
gonflé de 25% par la résolution de bissection, et lr-dépendant), et
le mécanisme du canal manquant lui-même (R sature à 1 au lieu de se
stabiliser, candidat structurel : la masse de fond transitoire).
**Un test SGD pour distinguer objectif-pur vs artefact-Adam est resté
inconclusif par réglage inadapté — piste complète mais non close,
notée pour une prochaine session** (il faudrait soit un `lr` SGD plus
grand avec le risque de désaturation trop brutale, soit un budget
bien plus long, soit reformuler le test en repartant d'un `s3_init`
déjà hors saturation plutôt que du défaut `0,999999999666`).

**Suite, même tour (Théo : « pose toi des hypothèses et fais les »).
Hypothèses posées et testées sur POURQUOI le SGD reste bloqué :**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le gradient brut de l'émetteur à s3_init est genuinement minuscule (pas un équilibre de forces qui s'annulent) | 18/09 (moi, standard) | **confirmée** le 18/09 — gradient mesuré directement : -6,71e-13 à s3_init=0,999999999666, cohérent avec le facteur s3(1-s3)≈3,34e-10 de la dérivée du sigmoïde |
| repartir d'un s3_init moins saturé (0,999 au lieu de 0,999999999666) permettrait à SGD d'explorer la région du pli en budget raisonnable | 18/09 (moi, standard) | **réfutée** le 18/09 — gradient à s3_init=0,999 huit ordres de grandeur plus grand (-1,3e-5) mais l'émetteur reste quasiment figé (0,999000→0,999020 en 150000 pas) ; le récepteur, lui, évolue normalement (r4 monte de 0,38-0,50 à 0,77-0,88) |
| le blocage est structurel au paramétrage sigmoïde (le gradient s'annule comme s3(1-s3) PARTOUT où la dynamique intéressante se joue, pas seulement au point de départ choisi) | 18/09 (moi, non standard) | **confirmée** le 18/09 par élimination — même en repartant loin de la saturation extrême, le blocage persiste tant que s3 reste proche de 1 (la région même où vit le pli, s3*≈0,9958) ; SGD pur n'est structurellement pas le bon outil pour ce test, indépendamment du réglage |

**Conclusion : la comparaison SGD-vs-Adam pour cette question précise
n'est pas juste mal réglée — elle est structurellement difficile,
parce que la région intéressante (près du pli, s3 proche de 1) est
justement celle où le gradient de l'émetteur en espace logit
s'annule par construction. Un test propre demanderait une
reparamétrisation (par exemple un gradient naturel, en divisant par
s3(1-s3) pour compenser exactement l'aplatissement du sigmoïde) plutôt
qu'un réglage de `lr`/budget — piste concrète pour une session future,
pas tentée ici (vrai travail de méthode, pas un ajustement de
paramètre).** Fil honnêtement refermé sur ce point précis.

Scripts (permanents) : `verifier_sgd_pur_jouet_m.py` (documente les
deux essais, tous deux inconclusifs pour la même raison structurelle).

**Suite, même tour (Théo : « continue, ne t'arrête pas sur des
hypothèses simples ») — le gradient naturel implémenté (diviser la
mise à jour de l'émetteur par `s3(1-s3)`), et un résultat qui remet
en cause une pièce du modèle établie plus tôt ce soir, pas juste un
détail.**

**Le gradient naturel FONCTIONNE (l'émetteur bouge enfin sous une
descente sans Adam) mais donne un résultat qualitativement DIFFÉRENT
d'Adam** : `s3` grimpe continûment vers 1 (0,999→0,99999833 en 40000
pas, M=0, toujours en train de monter, pas de plateau) — au lieu de
se stabiliser à 0,9964 comme sous Adam. **Test direct pour trancher
si le plateau d'Adam est un vrai point fixe ou un blocage lent près
du pli** : trace Adam étendue à 400000 pas (10× le budget original).
**Le plateau TIENT** — `s3=0,99643236` à `t=400000`, stable à 6
chiffres significatifs sur tout l'intervalle (avec une excursion H15
isolée à t=300000, `s3=0,9963183`, qui récupère — signature déjà
connue de ce mécanisme, retrouvée ici sans le chercher). **Donc le
plateau Adam n'est PAS un simple ralentissement transitoire — il
tient sur un ordre de grandeur de budget en plus.**

**Ça a forcé une vérification de ma propre "racine stable" (calculée
plus tôt ce tour pour l'analyse de stabilité en 3b/3a) : ELLE N'EST
PAS UNE VRAIE RACINE.** `F(x_s=1,141e-3)=-1,42e-5`, pas zéro — un bug
de précision dans ma propre bissection (même famille que le bug de
grille du pli trouvé plus tôt ce soir par un agent). **Pire, en
balayant `x` vers 0 en échelle log, `F(x)` ne s'annule JAMAIS — elle
approche un plateau POSITIF (~8e-4), pas zéro.** Dans la réduction
(x,R), `F(x)>0` signifie que `x_br(R)` cible une valeur PLUS GRANDE
que `x` courant — donc `dx/dt>0`, `x` devrait AUGMENTER (donc `s3`
DEVRAIT DIMINUER) — l'inverse exact de ce que montre la simulation
directe (gradient naturel : `s3` monte).

**Conclusion honnête, plus profonde qu'un bug de script : la
réduction quasi-statique (x,R) et la dynamique réelle simulée
directement se contredisent maintenant en dehors du voisinage
immédiat du pli — pas seulement "M n'a pas d'effet dessus" (établi
et solide, vérifié plusieurs fois AU pli), mais "la réduction
elle-même cesse d'être fiable dès qu'on s'en éloigne".** Cohérent
avec ce qui a déjà été trouvé ce tour (R n'est pas réellement
quasi-stationnaire pendant le transitoire de masse de fond) — mais
ici le problème apparaît même sans masse de fond (M=0), donc c'est
plus général qu'un effet de M : c'est une limite de l'hypothèse de
quasi-stationnarité de `R` elle-même, dans TOUTE la réduction, pas
seulement en présence de fond. **Ce n'est plus une question qui se
tranche avec un script de plus — ça demanderait de refaire la
dérivation de `R_br(x)` en abandonnant l'hypothèse d'équilibre
instantané du récepteur, un vrai travail de modélisation neuf.**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le plateau d'Adam à s3=0,9964 est un ralentissement transitoire près du pli, pas un vrai point fixe | 18/09 (moi) | **réfutée** le 18/09 — tient sur 400000 pas (10× le budget), stable à 6 chiffres significatifs |
| ma "racine stable" calculée en 3a/3b (x_s=1,141e-3) est une vraie racine de F(x)=0 | 18/09 (moi, implicite) | **réfutée** le 18/09 — F(x_s)=-1,42e-5, pas zéro ; bug de précision dans ma propre bissection |
| F(x)→0 quand x→0 (s3→1 est cohérent avec la réduction quasi-statique) | 18/09 (moi, implicite) | **réfutée** le 18/09 — F(x) approche un plateau positif ~8e-4, pas zéro, en contradiction directe avec la simulation (s3 monte alors que F>0 prédit qu'il devrait descendre) |

**Bilan honnête final de cette session sur la piste "masse de fond" :
le résultat le plus solide (le pli lui-même, position et courbure,
est M-indépendant) tient toujours, vérifié de multiples façons
indépendantes. Mais la réduction (x,R) censée l'entourer s'est
révélée moins fiable qu'espéré dès qu'on s'écarte du pli lui-même —
une vraie limite de méthode découverte cette nuit, pas résolue, et
qui mérite une reprise complète de la dérivation plutôt que d'autres
scripts ponctuels. Point d'arrêt honnête : on a appris quelque chose
d'important sur les limites de l'outil qu'on a construit ce soir,
même si ça n'a pas donné la réponse finale espérée.**

**RÉTRACTATION, même tour (Théo : « continue, pose des hypothèses
inconnues ») — l'entrée ci-dessus était une fausse alerte, la vraie
cause trouvée et vérifiée deux fois indépendamment.**

La « contradiction de signe » venait de comparer `F(x)` calculé avec
`R_br(x,M,delta)` (la valeur ALGÉBRIQUE d'équilibre du récepteur,
étant donné x) à une trajectoire où `R` n'est PAS à l'équilibre
pendant le transitoire (déjà établi plusieurs fois ce tour). En
substituant le `R` RÉELLEMENT MESURÉ dans la trajectoire au lieu du
`R_br(x)` algébrique : le signe de `x_br(R_mesuré)-x` colle
EXACTEMENT au signe de `dx/dt` observé — **pas de contradiction du
tout, juste une comparaison à la mauvaise variable.**

**Ce qui restait (un facteur d'échelle ~3000-6000x, présenté d'abord
comme "juste une constante de taux manquante") s'est révélé être
autre chose : un CHOIX DE COORDONNÉE INVALIDE, pas une constante.**
Un agent-dipankar (challengé, puis revérifié indépendamment par moi
avec un résultat encore plus net) a montré que ce "facteur" dérive
(3999→4234 sur 2400 pas à M=0, +50% de saut à M=25) parce que `x` et
`x_br(R)` sont tous deux minuscules (~1e-3) même quand le système est
en fait TRÈS loin de l'équilibre en espace LOGIT (`z3=6,91` contre
`z3_br=24,53`, écart de 17,6 unités) — la compression exponentielle
du sigmoïde écrase cet écart en une différence de probabilité de
~2,5e-4, donnant l'illusion d'être "presque à l'équilibre".

**La bonne relation, dérivée puis vérifiée EXACTEMENT (ratio=1,0000,
pas ~99,5%, à deux `lr` différents — 0,02 et 0,002) :**

```
dz3/dt = lr * (poids3*r3 - (beta/N)*z3)
```

**Linéaire en espace LOGIT, exacte par construction** (le gradient
naturel divise déjà par `s3(1-s3)`, qui annule exactement ce même
facteur dans le gradient brut — donc cette relation n'est pas une
approximation, c'est une identité algébrique, vérifiée numériquement
aux deux `lr` sans le moindre écart). Le résidu de 3-4% que l'agent
trouvait en espace `x` vient uniquement de la règle de dérivation
`dx/dz3≈-s3(1-s3)≈-x`, valable seulement près de `x=0` — pas d'un
terme manquant dans la dynamique elle-même.

**Et le facteur ~2x supplémentaire trouvé à M=25 (ratio ~6000 au lieu
de ~4000) s'explique complètement par l'approximation `r3≈1-R` de
`x_br`, déjà documentée dans son propre docstring comme approximative
— fausse de 67% à M=25 (`r3` réel=0,375 contre `1-R`=0,625).** Pas un
second phénomène physique.

**Conclusion propre, cette fois définitive pour ce tour : la
réduction (x,R) n'est pas cassée — elle était exprimée dans la
MAUVAISE VARIABLE (`x` au lieu de `z3`, l'espace logit) pour analyser
la dynamique loin du pli. En espace logit, la relaxation est linéaire
et exacte. Le "facteur M" à M=25 est entièrement expliqué par
l'approximation déjà connue `r3≈1-R`, pas par un canal manquant
supplémentaire.** Piste ouverte notée par l'agent, non testée :
vérifier si le plateau Adam à `s3=0,9964` (400000 pas, confirmé
robuste) s'explique par la même relation exacte en espace logit, ou
si l'adaptivité propre d'Adam (sans le dénominateur explicite
`s3(1-s3)` du gradient naturel) produit un point fixe différent —
pas encore vérifié.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| la réduction (x,R) se contredit avec la simulation directe (contradiction de signe) hors du pli | 18/09 (moi) | **RÉTRACTÉE** le 18/09 — comparaison à la mauvaise valeur de R (algébrique au lieu de mesurée) ; aucune contradiction une fois corrigé |
| le facteur d'échelle ~3000-5000x est une simple constante de taux manquante côté émetteur | 18/09 (moi) | **réfutée** le 18/09 par un agent, confirmé par moi — ce n'est pas une constante (dérive de 6%), c'est un choix de coordonnée invalide (x au lieu de z3, espace logit) |
| la relation dz3/dt=lr*(poids3*r3-(beta/N)*z3) décrit exactement la dynamique en espace logit | 18/09 (agent, revérifié par moi) | **confirmée** le 18/09 — ratio=1,0000 exact à deux lr différents (0,02 et 0,002), pas une approximation |
| le facteur ~2x supplémentaire à M=25 est un second phénomène physique | 18/09 (moi, implicite) | **réfutée** le 18/09 — entièrement expliqué par l'approximation déjà connue r3≈1-R, fausse de 67% à M=25 |

Scripts (permanent, mis à jour) : `verifier_gradient_naturel_jouet_m.py`.

**Clôture finale de tout le fil "Adam vs flot de gradient réel",
même tour (goal actif : « continue, ne t'arrête pas »).**

Un troisième agent, challengé sur "Adam et le flot réel convergent-ils
vers le même point ?", a dérivé la relation exacte manquante côté
RÉCEPTEUR (jusque-là non écrite), symétrique à celle de l'émetteur :

```
u* = (1/beta) * [(1-delta)*s3 - (1+delta)*s4]      (u=q3-q4, r3=sigmoid(u))
r4* = 1 - sigmoid(u*)
```

**Vérifiée indépendamment (recalcul direct, pas de trace) : `r4*=
0,8852106009` contre le plateau Adam observé `r4=0,8852130348` — écart
2,434e-6, confirmé au chiffre près.** L'agent a aussi précommis un
test binaire décisif : le système COUPLÉ (émetteur ET récepteur tous
deux en gradient naturel) doit converger vers `r4*` en un budget
comparable à Adam (quelques milliers de pas) si "même destination,
Adam juste plus rapide" est la bonne lecture.

**Premier essai (lr=0,02, jusqu'à 20000 pas) : test ÉCHOUÉ** —
`r4=0,698` à t=20000, encore à 0,187 de la prédiction. Plutôt que de
conclure à un problème de fond, testé l'explication la plus triviale
d'abord : `lr` insuffisant pour CE système rescalé. **À `lr=5,0` :
convergence quasi parfaite** — `r4=0,88521079` à t=40000, écart à la
prédiction analytique de **1,845e-7**, et `s3=0,99643232` correspond
EXACTEMENT au plateau Adam (`0,9964323591`, accord à 7 chiffres
significatifs).

**Conclusion définitive, la plus solide de tout ce fil : Adam et le
flot de gradient "vrai" (sans aucune adaptivité, juste un `lr`
suffisant et les bons facteurs de gradient naturel émetteur+récepteur)
convergent vers EXACTEMENT le même point fixe.** Le "facteur 500x" qui
semblait séparer Adam du gradient naturel n'était pas un mécanisme
manquant — juste un `lr` insuffisant dans la première tentative
(0,02 au lieu de ~1-5 nécessaires pour ce système rescalé). Le plateau
Adam à `s3=0,9964`/`r4=0,8852` (confirmé robuste sur 400000 pas plus
tôt ce tour) est un vrai point fixe de l'objectif lui-même, avec
maintenant une forme fermée exacte pour les DEUX coordonnées
(émetteur ET récepteur), vérifiée indépendamment à chaque étape.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| une relation exacte symétrique existe côté récepteur pour son propre point fixe sous gradient naturel | 18/09 (agent) | **confirmée** le 18/09, revérifiée par moi — écart 2,434e-6 avec le plateau Adam observé |
| le système couplé en double gradient naturel converge vers r4* en quelques milliers de pas (test précommis) | 18/09 (agent) | **réfutée** à lr=0,02 (encore à 0,187 de r4* à t=20000) — MAIS confirmée à lr=5,0 (écart 1,845e-7 à t=40000) : le budget nécessaire dépend fortement du lr, pas une réfutation du mécanisme |
| Adam et le flot de gradient réel convergent vers le même point fixe, Adam accélérant juste la convergence | 18/09 (moi, puis confirmé analytiquement) | **CONFIRMÉE définitivement** le 18/09 — accord à 7 chiffres significatifs sur s3 ET r4 entre Adam et le double-gradient-naturel à lr=5,0 |

**Ce fil est maintenant clos avec la conclusion la plus solide
possible : fermée analytiquement des deux côtés (émetteur ET
récepteur), vérifiée trois fois indépendamment (moi, deux agents),
avec un test précommis qui a d'abord semblé échouer puis a été
expliqué et confirmé en creusant l'explication la plus simple
d'abord (lr insuffisant) plutôt que de sauter à un mécanisme
compliqué.** Exactement la discipline que ce projet répète depuis le
début : chercher la version la plus triviale d'une explication avant
d'en construire une plus exotique.

Scripts (permanent) : traces reproductibles depuis
`verifier_gradient_naturel_jouet_m.py` (à étendre avec le gradient
naturel du récepteur si ce fil est repris).

**Note de synthèse finale, même tour — l'outil qui répondait à tout
existait déjà avant que la question soit posée.**

`R_br(x;M)` (utilisé depuis 3a) n'était pas une approximation qui
avait besoin d'être complétée par le gradient naturel du récepteur —
**c'est déjà, algébriquement, l'exact maximiseur softmax sous
contrainte** (récompense linéaire + entropie ⇒ distribution de Gibbs
`r_i ∝ exp(N·reward_i/β)`, exactement la forme déjà codée). Seule
l'approximation `r3≈1-R` (utilisée UNIQUEMENT dans la réduction ODE
simplifiée `x_br`, jamais dans `R_br` lui-même) portait l'erreur à
M=25. **Et `point_fixe(M,delta)` — codé bien avant ce fil, dans
`verifier_point_fixe_jouet_m.py`, pour une tout autre raison (mesurer
des taux de relaxation) — EST la solution jointe exacte** (émetteur
ET récepteur, sans aucune approximation) : vérifié contre le plateau
Adam (M=0) avec un écart de **3,929e-8 sur s3, 2,238e-6 sur r4** —
retombant exactement sur les précisions déjà trouvées séparément côté
émetteur et récepteur ce tour, parce que c'est littéralement le même
système d'équations.

**Ce que ça dit sur toute la piste 3 (masse de fond), avec le recul :**
le pli, sa position, sa courbure — tout ça ne dépendait jamais de M
parce que le VRAI point fixe joint (`point_fixe`) ne dépend pas de M
là où on l'a sondé (confirmé, exponentielles dominantes). Ce qui
dépendait de M — et qui a occupé l'essentiel de ce tour — n'était
jamais la DESTINATION mais le CHEMIN pour y arriver : quelle
dynamique transitoire, à quelle vitesse, via quelle variable
(logit vs probabilité), sous quel optimiseur. Le fil est refermé avec
la conscience claire que ces deux questions (où est le point fixe /
comment y arrive-t-on) sont restées séparées tout du long, et que la
confusion entre les deux a produit la plupart des fausses alertes
diagnostiquées et corrigées ce soir.

**Tentative de transfert au VRAI système (H6, six échecs
diagnostiqués plus tôt dans la session) — essayée, et honnêtement
non concluante, pour une raison structurelle claire, pas un échec de
réglage.** L'idée : appliquer le même gradient naturel (diviser par
`s3(1-s3)`) au point H6-direct pour enfin obtenir un coefficient `a`
stable, sans les artefacts d'Adam déjà catalogués. Testé à
`lr=0,01/0,05/0,2` : `s3` bouge à peine (0,994295→0,994298 en 60 pas
au mieux). **Ce n'est pas un problème de réglage — c'est que le
gradient naturel corrige un ralentissement dû au CHOIX DE COORDONNÉE
(sigmoïde saturé, le problème du jouet), pas un ralentissement dû à
la PROXIMITÉ D'UN VRAI POINT FIXE (le problème de H6-direct, placé
délibérément à 5e-6 du col, un vrai ralentissement critique
`μ≈0`).** Ces deux lenteurs se ressemblent en surface mais ont des
causes différentes ; l'outil qui répare l'une ne répare pas l'autre.
**La conclusion de la session précédente sur H6 (mécanisme compris —
démarrage à froid d'Adam sur un vrai col hyperbolique — mais mesure
précise de `a` toujours hors de portée, chemin restant : une vraie
trajectoire d'approche lente) reste donc l'état de l'art, pas
remplacée par cette tentative.** Noté honnêtement plutôt que forcé
vers une fausse victoire — transférer une bonne leçon au mauvais
problème est aussi une leçon.

**Essai rapide de la recette "trajectoire d'approche naturelle"
elle-même (repartir de `R_init=0,5` par défaut, sans le placer
artificiellement, et bissecter `s3_init` pour retrouver le seuil
H6/H11 historique de `verifier_sonde_bassin.py` ROUND 1) — NE
REPRODUIT PAS le seuil attendu.** Le ROUND 1 historique rapportait un
effondrement net à `s3_init=0,99400` (`DELTA=0,013`, `PAS=40000`).
Rejoué ici avec `continuer_sous_prior` et les mêmes paramètres
apparents : **tout reste gradué, y compris à `s3_init=0,994`**
(`s3_final≈0,999`, pas d'effondrement observé sur la plage
0,994-0,995 testée). Un paramètre diffère quelque part (peut-être
`R_init` par défaut de `construire_mur23` n'est pas exactement 0,5,
ou un autre détail de configuration a changé depuis ce test
historique) — pas élucidé ici, pas le budget pour cette session.
**Confirme ce qui avait déjà été anticipé : construire une vraie
trajectoire d'approche lente pour `masse_fond=0` est un vrai travail
de modélisation neuf, pas un raccourci d'une heure — le premier essai
naïf ne reproduit même pas le point de départ historique attendu.**
Laissé comme piste pour une session dédiée, avec ce premier essai
documenté pour ne pas le retenter à l'identique.

**PERCÉE, même tour (Théo, goal actif : « continue sans arrêter,
cherche, pense, réfléchis... n'oublie pas les agents ») — un agent en
worktree isolé a repris exactement là où l'essai précédent s'était
arrêté, et a trouvé la trajectoire naturelle cherchée.**

**Le vrai seuil ROUND 1 de `verifier_sonde_bassin.py` (celui que je
n'avais pas su reproduire) est en réalité à `s3_init≈0,9726`, PAS à
0,994 — vérifié : ROUND 1 reproduit tel quel aujourd'hui donne
GRADUÉ partout de 0,98 à 0,98 (aucune collision testée sous 0,98),
et c'est en étendant la plage bien plus bas (jusqu'à 0,001) qu'un
effondrement net apparaît, entre `s3_init=0,90` (effondre) et
`0,98` (gradué).** Bissecté finement (R laissé NATUREL, `~0,5`, PAS
placé artificiellement) : seuil net entre **`s3_init=0,972646`**
(effondre à `pas=4000`) et **`0,972661`** (reste gradué) — stable de
`pas=4000` à `40000` (vérifié indépendamment par l'agent, pas un
artefact de fenêtre trop courte).

**En traçant la trajectoire "SOUS le seuil" (celle qui finit par
s'effondrer) pas par pas : sa vitesse `|Δs3|` est MINIMALE à
`t≈263-265`, à `s3=0,994327`, `R[10,4]=0,829197` — à seulement
0,000027 et 0,000193 du point H6 publié (0,994300 / 0,829390).**
Quasi immobile de `t≈235` à `t≈290` (55 pas), après ~260 pas de
réchauffement Adam RÉEL. **Reproduit indépendamment par l'agent au
chiffre près (6 décimales), avec un croisement de vitesse net (pas
de bruit) entre `t=264` et `t=266`.** C'est la première trajectoire
naturelle (pas de départ à froid, pas de `fixer_r4` artificiel) qui
passe véritablement par le voisinage du point H6, avec un optimiseur
déjà chaud — exactement l'ingrédient qui manquait depuis six échecs.

**Vrai bug trouvé au passage, corrigé** : les fonctions
`tracer_trajectoire_naturelle`/`balayage_delta_naturel` du script
`verifier_trajectoire_naturelle_mur23.py` (committé plus tôt ce
tour) appelaient `continuer_sous_prior()` EN BOUCLE par blocs de 20
pas — or cette fonction reconstruit un `torch.optim.Adam` NEUF à
CHAQUE appel, donc l'optimiseur était remis à zéro tous les 20 pas.
**Ça invalide (en partie) la conclusion précédente "aucune
trajectoire naturelle ne ralentit"** — testé par l'agent : à `t=40`,
la version buguée donne `s3=0,973571` contre `s3=0,994822` pour la
version corrigée, un écart de 0,021 qui grossit ensuite jusqu'à
l'effondrement complet côté buggé. **Corrigé (optimiseur unique
continu) et recommité.** Ma propre découverte du seuil `0,972646`
n'était PAS affectée par ce bug (bissection par appel unique, trace
manuelle avec boucle Adam continue) — confirmée indépendante et
correcte par l'agent.

**Le coefficient `a` sur cette nouvelle trajectoire : PAS UTILISABLE,
même échec (en pire) que `a_H6direct`.** Premier fit (fenêtre
`pas=100-400`, coordonnées centrées pour éviter le mauvais
conditionnement déjà documenté) donnait `a≈+1990` à `+2016` selon la
foulée de différenciation, stable à 2% près sur 3 réglages — signe
POSITIF, à l'opposé de `a_delayed=-9,70`. **Mais l'agent a montré
que ce résultat ne survit PAS à un test de robustesse simple** : en
étendant la fenêtre de seulement 100 pas de plus (`pas=100-500`), le
signe s'INVERSE (`a≈-282` à `-350`) ; sur des sous-fenêtres
disjointes de 75 pas, `a` varie de `-1683` à `+9865` — **un facteur
>11500, pire que la dispersion `×5,6` qui avait déjà disqualifié
`a_H6direct`.** Mécanisme identifié : sur cette fenêtre, `x` (=`s3`)
ne varie que de `6e-5` — le terme quadratique `a·(Δx)²` au bord de la
fenêtre (`~1,8e-6`) dépasse à peine `μ` (`~1,4e-7`, marge `×12,4`
seulement), et le solveur `moindres_carres_quadratique` (élimination
de Gauss SANS PIVOT, déjà utilisé ailleurs dans ce projet) est
structurellement mal conditionné dans ce régime — le même mode
d'échec que celui d'`a_H6direct`, mais encore plus sévère ici.

**Protocole précommis proposé par l'agent pour la suite (pas encore
exécuté) — la méthode qui a DÉJÀ produit `a_delayed` de façon
stable, pas un nouveau fit de courbure brute** : mesurer un TEMPS DE
RÉSIDENCE (nombre de pas où `|s3-0,9943|<1e-3`) pour plusieurs
`s3_init` bissectés autour de `0,972646`-`0,972661`, et vérifier la
loi d'échelle `τ~C/√μ` (la même méthode déjà validée pour
`a_delayed`, prédiction précommise : si c'est le même point selle
que H6, l'exposant doit être `-1/2` et le préfacteur `C` du même
ordre que celui déjà mesuré pour `a_delayed` — pas à ×200 près).

**Bilan honnête de cette percée : la COORDONNÉE du ralentissement
(0,994327 ; 0,829197) colle au H6 publié à moins de 0,0002 près,
vérifiée deux fois indépendamment — la meilleure preuve dynamique
d'identité obtenue à ce jour. Mais le coefficient `a` reste
insaisissable, pour une raison numérique maintenant bien comprise
(fenêtre trop étroite en `x`, mauvais conditionnement), pas pour un
défaut de la trajectoire elle-même.** L'identité exacte du point
selle reste donc NI confirmée NI réfutée — mais avec, pour la
première fois, un signal positif fort (la coïncidence de
coordonnées) plutôt qu'une liste d'échecs de mesure seuls.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| une trajectoire naturelle (sans départ à froid artificiel) peut passer près des coordonnées H6 publiées | 18/09 (moi) | **confirmée** le 18/09, deux fois indépendamment — écart <0,0002 sur les deux coordonnées, croisement de vitesse net (pas de bruit) |
| le coefficient a mesuré sur cette trajectoire (+1990 à +2016) est fiable | 18/09 (moi, implicite) | **réfutée** le 18/09 par un agent — signe s'inverse en élargissant la fenêtre de 100 pas, variation ×11500 sur sous-fenêtres disjointes, pire que a_H6direct |
| verifier_trajectoire_naturelle_mur23.py (committé plus tôt) a un bug d'optimiseur qui invalide sa conclusion négative | 18/09 (agent) | **confirmée** le 18/09, vérifié directement dans le code (continuer_sous_prior reconstruit Adam a chaque appel, appelee en boucle) — corrigé et recommité |

Scripts (permanents, corrigés) : `verifier_trajectoire_naturelle_mur23.py`.

**Suite, même tour — protocole du temps de résidence exécuté, et une
reformulation profonde de toute la question d'identité du point
selle, appuyée par deux lignes de preuve convergentes.**

**Temps de résidence mesuré (bande `|s3-0,9943|<1e-3`, `pas=3000`,
des deux côtés du seuil `0,972646/0,972661`, distance à l'exacte
frontière de `1e-3` à `1e-5`) :**

```
offset       n (SOUS)   n (SUR)
1e-3         236        278
3e-4         351        398
1e-4         445        488
3e-5         540        573
1e-5         628        638
```

**Ajustement logarithmique : `R²=0,998325` (excellent). Ajustement en
loi de puissance : exposant `-0,208`, `R²=0,965825` (nettement pire,
et loin de `-1/2`, la signature attendue d'un fantôme de pli).**
**Reproduit EXACTEMENT (bit-à-bit) par un agent, et confirmé ROBUSTE**
au changement de bande/budget (`bande=5e-4`, `pas=5000`) : `λ_SOUS`
et `λ_SUR` convergent à moins de 1 % l'un de l'autre — exactement la
signature symétrique attendue d'un vrai col hyperbolique (échappement
de part et d'autre à la même vitesse locale), pas d'un fantôme
asymétrique.

**Correction importante trouvée par le même agent, vérifiée
indépendamment ici (CARNET, lignes ~8759-8765) : l'accord "à 6% près"
du préfacteur `C` entre les configs "survit"/"effondre" du cas
retardé (`0,920` / `0,977`, cité plus tôt cette session comme
"confirmation d'une vraie structure de fantôme") ÉTAIT UNE TAUTOLOGIE
ALGÉBRIQUE, pas un test indépendant.** `C_i = τ_partagé·√(μ_i)` avec
le MÊME `τ=600` pour les deux configs ⇒ `C_effondre/C_survit ≡
√(μ_effondre/μ_survit)` par construction, quel que soit `τ`. Vérifié
par calcul direct : `√(2,65e-6/2,35e-6)=1,061913`, et
`0,977/0,920=1,061957` — identique à la précision de calcul près.
**Cette "confirmation" doit être rétrogradée : elle ne prouve rien
sur l'exposant `-1/2`, seulement une identité arithmétique.**

**Reformulation profonde, appuyée par DEUX preuves convergentes,
établies par des méthodes complètement différentes :**
1. **La jacobienne 2×2 du VRAI système (pas le jouet — vérifié :
   `verifier_saddle_h6_jacobien_independant.py` utilise bien
   `replay_mur23_referent3.construire_mur23`) au point H6 exact
   (`s3=0,994295`, `R=0,829390`) donne deux valeurs propres RÉELLES
   de signe opposé — un col hyperbolique ORDINAIRE, établi PLUS TÔT
   cette même session, confirmé deux fois indépendamment.**
2. **Le temps de résidence mesuré ce soir près de ce même point suit
   une loi LOGARITHMIQUE (`R²=0,998`), pas une loi de puissance en
   `1/√μ` — exactement ce qu'un col hyperbolique ordinaire prédit
   (échappement linéaire/exponentiel, temps de résidence
   `~-log(distance)/λ`), PAS ce qu'un fantôme de pli dégénéré
   prédirait.**

**Ces deux preuves, indépendantes et convergentes, pointent la même
conclusion : H6 (à `delta=0,013` FIXE, strictement sous `delta_c`)
est structurellement un col hyperbolique ORDINAIRE — pas un fantôme
de nœud-col.** Le "fantôme" caractérisé ailleurs dans cette session
(`a_delayed`, le point rencontré DYNAMIQUEMENT sous masse de fond
décroissante) reste, lui, un phénomène de type différent — une
bifurcation réellement dégénérée rencontrée alors que le `delta_c`
EFFECTIF dérive dans le temps (masse de fond qui s'évacue), pas un
col fixe à `delta` constant. **La question "même point selle ou
voisin" était peut-être mal posée depuis le début : ce ne sont
vraisemblablement pas deux points selles de la MÊME FAMILLE
dynamique du tout, mais deux TYPES DE STRUCTURES CRITIQUES
DIFFÉRENTES (hyperbolique ordinaire vs nœud-col dégénéré) qui se
trouvent avoir des coordonnées proches dans le même coin de l'espace
des phases.**

**Ce qui reste ouvert, honnêtement non résolu :** la prédiction
naïve `λ=2√(delta_c-delta)=0,0418`/pas ne colle PAS à la valeur
mesurée (`λ≈0,0021-0,0031`/pas, écart `×13-20`) — un vrai trou
quantitatif, candidats non tranchés : unité de temps (pas Adam ≠
temps de flot continu), mauvaise coordonnée de forme normale (déjà
suspecté ailleurs dans le projet), ou réduction 1D en `delta` brut
invalide pour ce système couplé `(s3,r3,R)`. **Protocole précommis
pour trancher, pas encore exécuté : étendre la trace à offset=1e-5
au-delà de `pas=628` (la résidence mesurée) pour trouver le vrai
croisement d'instabilité (comme `g_e3` changeant de signe à
`pas=600,283` pour le cas retardé), mesurer `λ_instable` APRÈS ce
croisement, et comparer à `1/(|a_log|·ln10)≈0,0021-0,0022`/pas — un
facteur <2 confirmerait le lien col-hyperbolique/loi-log ; un facteur
>5 indiquerait que la loi log mesure autre chose (peut-être un temps
d'évacuation transverse, pas la fuite le long de la variété
instable).**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| le temps de résidence près de H6 suit une loi logarithmique, pas une loi de puissance en 1/√μ | 18/09 (moi) | **confirmée** le 18/09, reproduite bit-à-bit par un agent, robuste au changement de bande/budget (λ_SOUS≈λ_SUR à <1%) |
| l'accord à 6% du préfacteur C (a_delayed, cas retardé) confirme empiriquement l'exposant -1/2 | tour précédent (moi) | **RÉTRACTÉE** le 18/09 — c'est une tautologie algébrique (même τ partagé pour les deux configs), vérifié par calcul direct |
| H6 (delta fixe, vrai système) est un col hyperbolique ordinaire, pas un fantôme de pli dégénéré | 18/09 (moi, appuyée sur un résultat antérieur du même tour) | **confirmée** le 18/09 par deux preuves indépendantes convergentes (jacobienne à valeurs propres réelles établie plus tôt ce tour + loi de résidence logarithmique mesurée ce soir) |
| H6 et le point rencontré sous masse de fond (a_delayed) sont deux points selles de la même famille dynamique | implicite depuis plusieurs tours | **affaiblie/reformulée** le 18/09 — probablement deux TYPES de structures critiques différents (hyperbolique vs nœud-col dégénéré), pas juste "même point ou voisin" |
| la prédiction naïve λ=2√(delta_c-delta) prédit le taux d'échappement mesuré | 18/09 (agent) | **réfutée** le 18/09 — écart ×13-20, cause non élucidée (unités, coordonnée, réduction 1D invalide) |

Scripts (permanents) : `verifier_temps_residence_traj_naturelle_180926.py`,
`verifier_lambda_local_traj_naturelle_180926.py` (produits par l'agent,
pas encore rapatriés sous un nom permanent).

## VRAIE CRITIQUE DE DIPANKARSARKAR, 20/09/2026 (tour 53, pas une critique simulée)

**Le vrai relecteur externe a répondu, après plusieurs jours de
silence.** Trois corrections sur le tour précédent (k(R), beta1/beta2,
test hybride Adam/SGD) — réponse complète dans
`docs/REPONSE_ORDRE54.md`. Résumé ici pour le journal, daté.

**1. H_momentum : sa lecture en rééchelonnage est meilleure que la
mienne.** Vérifié : `beta1=0` agit comme un rééchelonnage quasi
uniforme de `k` (facteur `c≈1,4389`, résidus 1,5-3,3%), pas comme un
vrai changement de forme. Les statistiques invariantes d'échelle
(`k(0,75)/k(0,50)`, `spread/k(0,50)`) RÉTRÉCISSENT légèrement
(-5,2%, -12,3%) au lieu de grossir. **H_momentum reste réfutée, mais
pour la bonne raison cette fois** — la forme de `k(R)` survit à un
rééchelonnage de 1,44× de l'horloge, un argument plus fort pour
"intrinsèque à l'état" que l'ancien "l'écart a grossi".

**2. beta2 : un seul point ne peut pas parler de dépendance à
l'état — corrigé.** Le balayage beta2 d'origine était fait à
`R_init=0,60` SEUL. Testé son croisement précommis (`beta2=0,99` à
`R_init=0,50` ET `0,75`) :
```
R_init=0.75  beta2=0.99  flip=0.989714  k_fit=2.4357
R_init=0.50  beta2=0.99  flip=0.972195  k_fit=1.3971
```
`k(0,75)/k(0,50)=1,7434` contre la référence `1,7271` — écart
**0,94%**, sous la barre du simple rééchelonnage. Sa prédiction
tient.

**3. Le "200x" comparait `s3` (hybride) à `R` (Adam complet) — vraie
erreur, corrigée à ~3100x en comparant `R` à `R`.** Mais son propre
test précommis (le rapport local `s3_dip/R_dip=21,3` de l'hybride
devrait prédire un `s3` tombant vers ~0,957 sous Adam complet, si
c'est le même mécanisme) — REJOUÉ et RÉFUTÉ : `s3` reste quasi figé
(`~1e-5` à `2e-5`) pendant que `R` fait des excursions de `~2e-3` à
`2,6e-3` sous Adam complet. **Les deux optimiseurs n'excursent PAS
dans la même direction de l'espace d'état** — les excursions
hybride et Adam-complet sont probablement deux mécanismes
différents, pas le même à deux amplitudes.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| l'écart absolu de k entre beta1=0,9 et beta1=0 (1,326 contre 1,031) est le bon test de H_momentum | tour précédent (moi) | **réfutée/affinée** le 20/09 par dipankarsarkar — l'écart absolu n'est pas invariant à un rééchelonnage ; les statistiques invariantes d'échelle rétrécissent au lieu de grossir |
| le sweep beta2 à un seul R_init suffit pour juger de sa dépendance à l'état | tour précédent (moi) | **réfutée** le 20/09 par dipankarsarkar, confirmé par le croisement (0,94% d'écart, cohérent avec un simple rééchelonnage) |
| l'amplitude de l'excursion hybride (s3) et Adam complet (R) sont directement comparables ("~200x") | tour précédent (moi) | **réfutée** le 20/09 par dipankarsarkar — mauvaise variable comparée ; en comparant R à R, l'écart réel est ~3100x |
| si le rapport local s3_dip/R_dip=21,3 de l'hybride tient sous Adam complet, R dip=1,957e-3 devrait donner s3 tombant vers ~0,957 | 20/09 (dipankarsarkar, précommise) | **réfutée** le 20/09, rejouée moi-même — s3 reste figé (~1e-5) pendant que R excurse de ~2e-3, aucun signe de la chute prédite |

Scripts : `verifier_derive_k.py` (croisement beta2),
`verifier_localisation_excursion_full_adam.py` (trace Adam complet
s3+R ensemble, nouvelle, permanente).

**Vérification indépendante des deux résultats numériques ci-dessus
(agent, 20/09/2026, avant envoi de `REPONSE_ORDRE54.md`), puis rejeu
personnel du point 2 incomplet par l'agent — deux issues opposées.**

- **Résultat 3 (trace Adam complet) : CONFIRMÉ, à grille plus fine
  encore.** L'agent a réécrit son propre script (pas copié), 100 000
  pas, log tous les 2000 pas (10× plus fin que les 20000 d'origine).
  Trajectoire identique à la mienne au pas près (reproductibilité
  seed/config confirmée, pas supposée). Beaucoup plus d'excursions
  trouvées à cette résolution (7 au lieu de 2), mais **aucune
  n'atteint `s3≈0,957`** — `s3` reste borné à des variations `~1e-5`
  sur toute la fenêtre pendant que `R` dip jusqu'à `~3,1e-3`. La
  lecture "réfuté nettement, pas la même direction d'espace d'état"
  tient à grille fine aussi.
- **Résultat 2 (croisement beta2) : l'écart de 0,94% N'EST PAS
  fiable à ce niveau de précision — REJOUÉ moi-même, corrigé.**
  L'agent n'a pas pu boucler son propre rejeu dans le temps imparti
  (bissection à budget original trop coûteuse, ~37s/appel ×
  ~10-12 appels/point). J'ai rejoué moi-même avec un budget réduit
  (`pas=15000, tol=2e-4` contre `pas=40000, tol=1e-5` d'origine, pour
  rester dans un temps raisonnable) :
  ```
  flip75 = 0,989717   (original : 0,989714)
  flip50 = 0,972154   (original : 0,972195)
  k75 = 2,4373          (original : 2,4357)
  k50 = 1,3953          (original : 1,3971)
  ratio = 1,7468          (original : 1,7434)
  écart vs référence 1,7271 : 1,14%   (original annoncé : 0,94%)
  ```
  Les points de bascule (`flip`) reproduisent à 4-5 chiffres
  significatifs — pas de doute sur la trajectoire elle-même. Mais
  **l'écart lui-même bouge de 0,94% à 1,14% (+0,2 point, ~20%
  relatif) rien qu'en changeant le budget de bissection.** C'est
  précisément la question laissée ouverte par l'agent ("l'écart est-il
  sous la barre du rééchelonnage seul, ou comparable au bruit de
  bissection ?") — la réponse est que **le bruit de bissection à lui
  seul produit une variation du même ordre de grandeur que l'effet
  rapporté.** Le verdict qualitatif (rééchelonnage dominant, pas un
  effet de forme) tient probablement toujours, mais le chiffre précis
  "0,94%" ne doit pas être cité dans `REPONSE_ORDRE54.md` comme une
  mesure de précision inférieure au point, sans requalification.
  **À faire avant l'envoi : soit refaire au budget original pour une
  vraie comparaison à isoconfiguration, soit reformuler la conclusion
  en "écart de l'ordre de 1%, dans la même gamme que le bruit de
  bissection à cette résolution — cohérent avec un rééchelonnage, pas
  une preuve à la décimale près".**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| l'écart de 0,94% (croisement beta2) est net, sous la barre du rééchelonnage seul | 20/09 (moi, dans REPONSE_ORDRE54.md) | **affaiblie** le 20/09, rejeu à budget réduit donne 1,14% — le bruit de bissection à budget différent est du même ordre que l'effet rapporté ; conclusion qualitative probablement OK, le chiffre précis non |
| la trace Adam complet (s3 figé, R excurse) résiste à une grille de mesure plus fine | 20/09 (moi) | **confirmée** le 20/09 par agent, grille 10× plus fine (2000 vs 20000 pas), 7 excursions trouvées au lieu de 2, aucune n'approche s3≈0,957 |
| hypothèse standard n°1 de REPONSE_ORDRE54.md : la sensibilité locale du récepteur en R est plus grande que celle de l'émetteur en s3, expliquant pourquoi R excurse et pas s3 | 20/09 (moi) | **confirmée en conclusion, réfutée dans son mécanisme initial** — mesurée directement (différences finies) le 20/09 : en espace LOGIT (celui où Adam agit), l'émetteur est ~150× PLUS PLAT que le récepteur (pas plus raide comme je l'avais écrit) ; c'est la compression de saturation du softmax (`s3(1-s3)=1,04e-3` contre `R(1-R)=0,163`, facteur 157×) qui inverse tout en espace probabilité, où l'émetteur redevient ~160× plus raide. Script permanent : `verifier_courbure_s3_vs_r.py` |

**Hypothèse standard n°1 (courbure `s3` vs `R`) testée directement le
20/09/2026, script `verifier_courbure_s3_vs_r.py`.** Au plateau convergé
(`s3=0,998963`, `R=0,794756`), différences finies de `d²J/dlogit²` :
```
d2J/dlogit_s3^2  = -7,9e-7
d2J/dlogit_R4^2  = -1,21e-4     (ratio |emetteur/recepteur| = 0,0065)
```
**En espace logit — celui où Adam applique réellement ses pas —
l'émetteur est ~150× PLUS PLAT, pas plus raide.** Ma phrase initiale
dans `REPONSE_ORDRE54.md` ("s3 sits near a saturated optimum where its
own second derivative is large and stiff") était fausse dans son
mécanisme, bien que sa conclusion finale (s3 stable, R mobile) reste
correcte. Ce qui inverse le signe de l'argument : la compression de
saturation du softmax, `ds3/dlogit = s3(1-s3) ≈ 1,04e-3` contre
`dR/dlogit = R(1-R) ≈ 0,163` (facteur 157×). En convertissant via la
règle de la chaîne (valide ici : gradient quasi nul au plateau, donc le
terme correctif habituel s'annule) :
```
d2J/ds3^2  (approx) ≈ -0,734
d2J/dR4^2  (approx) ≈ -0,0045    (ratio |emetteur/recepteur| ≈ 162)
```
**La raideur visible en espace `(s3,R)` est réelle et ~160× plus
grande pour `s3`, mais ce n'est pas une raideur intrinsèque — c'est la
compression au carré d'un paysage en fait plus mou côté émetteur.**
Une fluctuation du second moment d'Adam de taille comparable en espace
logit est donc amortie de ~3 ordres de grandeur côté émetteur en
espace probabilité, sans qu'aucun mécanisme séparé côté récepteur ne
soit nécessaire pour expliquer l'asymétrie observée. Corrigé dans
`REPONSE_ORDRE54.md` avec les chiffres exacts avant envoi.

**JUSQU'OÙ ce mécanisme tient-il ? Testé le 20/09/2026 — CASSE, comme
pressenti, et un agent-dipankar a trouvé pourquoi.** Lancé
automatiquement (règle CLAUDE.md : chaque résultat substantiel passe
par un agent-dipankar avant clôture), isolation worktree, contrainte
Bash-only. Son verdict : **la conversion `comp²` (Hessienne en espace
logit → probabilité) n'est PAS la bonne question.** Adam normalise
chaque coordonnée par `m/(√v+eps)` — la courbure brute (Hessienne) n'a
aucun effet DIRECT sur la taille du pas. Ce qui compte est la marche en
espace LOGIT elle-même. Si Adam égalise cette marche entre coordonnées
(ce qui est son but de conception), une SEULE puissance du facteur de
compression du softmax (157×) suffit à expliquer l'écart observé en
probabilité (~150-160×), pas son carré (162×) que j'avais utilisé.

Snapshot à un seul pas (bruité, au plancher de précision float64,
gradients ~1e-12-1e-13) : mon calcul indépendant donne un ratio de pas
effectif `|s3/R4|=2,30`, l'agent avait rapporté `0,58` — **désaccord
net, mais peu importe : un seul pas au plateau convergé est une
mesure de bruit, pas un test.** Le vrai test (précommis par l'agent,
« expérience 1 ») : mesurer `Δlogit_s3` et `Δlogit_R4` directement sur
toute la durée d'une excursion RÉELLE (fenêtre `[57500,62500]`, autour
du dip à `pas=60000`), pas un instantané.

```
logit_s3 range sur la fenêtre = 0,023438
logit_R4 range sur la fenêtre = 0,023084
ratio |Δlogit_R4/Δlogit_s3| = 0,985
```

**Quasi 1:1, pas 150×.** La lecture Hessienne (`comp²`) est
**RÉFUTÉE** — pas juste affaiblie. La lecture de l'agent (`comp¹`,
marches logit comparables sous Adam) est **CONFIRMÉE**, de façon nette
et précommise avant le résultat. Script permanent :
`verifier_delta_logit_excursion.py`.

**COMMENT le mécanisme fonctionne réellement, reformulé :** Adam
égalise (à ~1,5% près, pendant une vraie excursion) le déplacement en
espace logit entre émetteur et récepteur — ce n'est pas une propriété
de la Hessienne locale, c'est une propriété de conception de
l'optimiseur lui-même (normalisation coordonnée par coordonnée). Le
déplacement en espace PROBABILITÉ hérite ensuite d'un facteur de
compression du softmax à la puissance UN (via la règle de la chaîne au
premier ordre, `ds/dlogit = s(1-s)`), pas au carré. `s3` ne bouge presque
pas non pas parce que son puits est profond, mais parce que le même
déplacement logit, appliqué à une probabilité déjà saturée
(`s3(1-s3)≈0,00104`), produit mécaniquement un déplacement de
probabilité ~157× plus petit que le même déplacement appliqué à `R`
(`R(1-R)≈0,163`, une variable non saturée).

**Conséquence pour `REPONSE_ORDRE54.md` : le paragraphe sur
l'hypothèse standard n°1 (déjà corrigé une fois avec les chiffres de
courbure) doit être corrigé UNE SECONDE FOIS** — sa conclusion reste
juste (s3 stable, R mobile, ratio ~150-160×), mais son mécanisme
(« courbure Hessienne amplifiée par la compression au carré ») est
maintenant identifié comme faux ; le bon mécanisme est « Adam égalise
les marches en espace logit, la compression au premier ordre fait le
reste ». **Pas encore appliqué au fichier — Théo a demandé de ne plus
y toucher sans son accord explicite (20/09/2026).**

| # | hypothèse | posée le | statut |
|---|---|---|---|
| la raideur en espace probabilité vient d'une conversion Hessienne au carré (`comp²`) du facteur de compression du softmax | 20/09 (moi) | **réfutée** le 20/09 par agent-dipankar puis rejeu personnel précommis — le ratio `Δlogit` mesuré pendant une vraie excursion est 0,985 (quasi 1:1), pas ~150× ; la Hessienne brute n'entre pas dans le pas d'Adam |
| Adam égalise les marches en espace logit entre coordonnées, la compression au premier ordre (`comp¹`) suffit à expliquer l'écart observé en probabilité | 20/09 (agent-dipankar) | **confirmée** le 20/09, test précommis avant le résultat : ratio mesuré 0,985 contre une prédiction "<~3×" |
| les excursions à pas=60000 et pas=160000 sont deux événements distincts (signes opposés), pas un seul mécanisme | 20/09 (moi, dans REPONSE_ORDRE54.md, hypothèse non-standard n°3) | **réfutée** le 20/09 — une troisième occurrence à pas≈258000 a la même amplitude (`R4_min/max`, `v_r_max` à ±0,2% des deux premières) ; ce n'est ni un mode qui change de signe ni deux accidents, c'est un CYCLE LIMITE PÉRIODIQUE de période ~95000-100000 pas. Scripts : `verifier_periodicite_excursion.py` (3 fenêtres), `exp_avg_sq` déjà loggé sur 200000 pas |

**Le cycle limite périodique (nouvelle découverte, non anticipée par
aucune des hypothèses posées dans REPONSE_ORDRE54.md) — QUAND/COMBIEN
mesurés, POURQUOI pas encore élucidé.**
```
fenêtre  60000 : R4_min=0,791024  R4_max=0,798478  v_r_max=1,5539e-13  s3_min=0,998944
fenêtre 160000 : R4_min=0,791037  R4_max=0,798508  v_r_max=1,5527e-13  s3_min=0,998944 (7 chiffres identiques)
fenêtre 260000 : R4_min=0,790931  R4_max=0,798543  v_r_max=1,5746e-13
```
**QUAND** : la première occurrence identifiée est à `pas≈60000`,
espacement `≈98000-100000` pas entre occurrences successives — pas
encore mesuré AVANT `pas=60000` (DEPUIS QUAND ce cycle existe-t-il ?
reste ouvert, la fenêtre `[0,60000]` complète n'a pas été scannée à
grille fine). **COMBIEN** : amplitude quasi constante sur 3
occurrences (`R4` oscille entre `~0,791` et `~0,798`, un intervalle de
`~0,0075`, constant à `±0,2%`). **POURQUOI** ce cycle existe et garde
une amplitude stable plutôt que de s'amortir ou de diverger : PAS
ENCORE TESTÉ — hypothèses à poser au prochain tour (candidate
standard : régime limite d'un oscillateur de van der Pol effectif
émergent du couplage bias-correction/second-moment d'Adam avec un
gradient quasi nul ; candidate non-standard : la période ~100000
pourrait être liée à un ratio simple avec `1/beta2=1000` ou
`1/(1-beta2)=1000` du planning de bias-correction — à tester en
variant beta2 et en regardant si la période bouge proportionnellement).

**Candidate non-standard testée le 20/09/2026 — résultat NI CONFIRMÉ
NI RÉFUTÉ, méthode de détection inadaptée, honnêtement rapporté plutôt
que forcé.** Rejoué avec `beta2=0,99` (`1/(1-beta2)=100` au lieu de
`1000`) sur les premiers 40000 pas, en cherchant si la première
occurrence apparaît ~10× plus tôt (`~6000` au lieu de `~60000`).
**Deux bugs de méthode trouvés et corrigés en route** (bonne pratique
à retenir) : (1) première tentative avec `baseline` fixée à `pas=0`
(avant convergence, `R4=0,525`) — tout ressortait comme "excursion"
par construction, corrigé en fixant la baseline après convergence
(`pas=2000`) ; (2) même corrigée, la valeur à `pas=2000` elle-même
(`R4=0,796172`) s'est révélée être encore dans le transitoire, pas le
vrai plateau — **résultat** : à `beta2=0,99`, `R4` n'a PAS de plateau
stable ponctué d'excursions discrètes comme à `beta2=0,999` ; il
montre une **agitation continue** (`R4` oscille en permanence dans
`[0,7933; 0,7956]`, jamais immobile plus de quelques dizaines de pas)
— **le comportement change de NATURE, pas seulement de période.** La
détection par seuil de déviation (utile à `beta2=0,999` où le signal
est un plateau ponctué de pics nets) ne peut pas isoler un « premier
événement » à `beta2=0,99` puisqu'il n'y a plus de plateau de repos
identifiable. **Ni confirmé ni réfuté** : la prédiction `1/(1-beta2)`
n'a pas pu être testée proprement avec cette méthode — il faudrait une
détection par autocorrélation ou FFT sur la série `R4(t)` complète,
pas un seuil de déviation par rapport à une baseline qui n'existe plus
clairement à ce réglage. **Reste ouvert, méthode à changer avant de
retenter.** Ce résultat est lui-même informatif : `beta2` module la
FORME de la dynamique (impulsions discrètes vs bruit continu), pas
seulement une échelle de temps — cohérent avec `1/(1-beta2)` étant la
fenêtre effective de mémoire de `v`, mais pas une preuve du lien
period∝`1/(1-beta2)` spécifiquement.

**RÉTRACTATION le 20/09/2026 — « cycle limite périodique, période
~100000 » était un biais de sélection, pas un vrai résultat. Trouvé en
cherchant DEPUIS QUAND (méfiance appliquée à mon propre résultat qui
venait de confirmer une hypothèse, cf. règle CLAUDE.md).** En balayant
`[0,60000]` (grille 200 pas) puis `[0,200000]` (grille 1000 pas) sans
se limiter aux trois fenêtres déjà repérées par la grille grossière à
20000 pas, des fluctuations d'amplitude COMPARABLE apparaissent
PARTOUT, pas seulement près de `pas=60000/160000/260000` :
```
pas=22000  dev=+0,00185   pas=33000  dev=+0,00195   pas=50000  dev=+0,00130
pas=60000  dev=+0,00105   pas=69000  dev=-0,00092   pas=84000  dev=+0,00165
pas=85000  dev=-0,00069   pas=108000 dev=-0,00269   pas=112000 dev=+0,00051
pas=118000 dev=-0,00050   pas=142000 dev=-0,00065   pas=160000 dev=-0,00212
pas=194000 dev=+0,00120   pas=195000 dev=-0,00224
```
**Ce ne sont PAS trois occurrences isolées d'un même cycle espacé de
~100000 pas — ce sont des fluctuations FRÉQUENTES (au moins une
notable toutes les 10000-30000 pas), de magnitude comparable, sur toute
la trajectoire.** Mon repérage initial des « trois occurrences
identiques » souffrait d'un biais de sélection méthodologique clair :
je n'avais fait un balayage FIN que sur les trois fenêtres DÉJÀ
repérées par la grille grossière à 20000 pas de la trace originale —
bien sûr qu'elles se ressemblaient, aucune comparaison avec le reste
de la trajectoire n'avait été faite. **Le vrai phénomène est plus
proche d'un bruit/jitter quasi continu de magnitude caractéristique
`~0,002-0,004` (sous-échantillonné à toute grille plus grossière que
~200 pas), pas un oscillateur périodique à période fixe.** La
« hypothèse non-standard n°3 » de `REPONSE_ORDRE54.md » (deux
événements distincts vs un seul mécanisme) reste réfutée dans sa forme
originale (pas juste deux événements — il y en a beaucoup plus), mais
ma propre reformulation en « cycle limite périodique » ne tient pas
non plus. **Statut correct, plus modeste : les excursions de `R` sous
Adam complet sont des fluctuations fréquentes et récurrentes de
magnitude caractéristique bornée (~0,002-0,004), pas un mode
périodique isolé ni deux accidents isolés.** Reste à tester : la
distribution complète des amplitudes (est-elle bornée par une valeur
max stable, ou a-t-elle une queue lourde ?), et si leur fréquence
elle-même est stationnaire dans le temps ou dérive.

**Distribution des amplitudes caractérisée le 20/09/2026 (même tour) —
BORNÉE, pas de queue lourde.** Balayage complet `[0,400000]`, grille
500 pas (789 échantillons après retrait du transitoire initial) :
```
mean |dev| = 0,000155
top 20 |dev| (décroissant) : 0,00357  0,00332  0,00315  0,00285  0,00271
                              0,00260  0,00249  0,00249  0,00247  0,00240
                              0,00215  0,00212  0,00206  0,00204  0,00202
                              0,00192  0,00185  0,00170  0,00168  0,00161
top1/top10moyenne = 1,27      top10moyenne/top50moyenne = 1,82
```
**Décroissance lisse, pas de saut brutal entre le plus grand écart et
les suivants** — cohérent avec un processus de bruit borné (amplitude
caractéristique max ~0,0035-0,004 observée sur 400000 pas), pas une
distribution à queue lourde ni un événement catastrophique rare.
COMBIEN est maintenant répondu : l'amplitude typique est `~0,0002`
(moyenne), le maximum observé sur toute la trajectoire `~0,004`, un
facteur ~20 entre les deux — répartition continue, pas bimodale.
POURQUOI cette borne existe précisément à cette valeur : toujours
ouvert.

**Stationnarité testée le 20/09/2026 (même tour) — CONFIRMÉE, sur les
quatre quarts de la trajectoire complète.** Le processus est-il
stationnaire dans le temps (fréquence/amplitude constantes) ou
dérive-t-il ? Découpage `[0,400000]` en 4 segments de 100000 pas,
comptage des fluctuations `>0,0008` par segment :
```
segment [0,100000)     : n=10  mean|dev|=0,000152  max|dev|=0,003319
segment [100000,200000): n=10  mean|dev|=0,000172  max|dev|=0,003570
segment [200000,300000): n=9   mean|dev|=0,000148  max|dev|=0,003147
segment [300000,400000): n=11  mean|dev|=0,000148  max|dev|=0,002491
```
**Compte quasi identique (9-11) sur les quatre segments, moyenne quasi
identique (`0,000148-0,000172`), maximum dans la même gamme
(`0,0025-0,0036`, sans tendance monotone claire).** Pas de dérive, pas
de croissance ni de décroissance sur 400000 pas. Le processus de bruit
borné est stationnaire — DEPUIS QUAND répondu implicitement : il ne
« démarre » pas à un moment particulier, il est présent avec la même
statistique dès le début du régime post-transitoire (`pas>5000`)
jusqu'à la fin de la trajectoire mesurée. Script à sauvegarder :
la version courante tournait encore en `python -c`, à committer sous
nom permanent au prochain geste d'écriture.

| # | hypothèse | posée le | statut |
|---|---|---|---|
| les trois occurrences à pas=60000/160000/260000 forment un cycle limite périodique de période ~100000 pas | 20/09 (moi) | **RÉTRACTÉE** le 20/09, même tour — biais de sélection trouvé en balayant le reste de la trajectoire : des fluctuations comparables apparaissent toutes les 10000-30000 pas partout, pas seulement à ces trois points |
| le processus de bruit (fréquence, amplitude) est stationnaire sur toute la trajectoire, pas de dérive | 20/09 (moi) | **confirmée dans l'esprit (stationnaire), mais les chiffres qui la portaient étaient faux — voir SECONDE RÉTRACTATION ci-dessous** |

**SECONDE RÉTRACTATION, le même jour — trouvée par un agent-dipankar,
puis vérifiée indépendamment. Le bug d'échantillonnage que je venais
de diagnostiquer et « corriger » existait encore, un niveau plus bas,
dans ma propre correction.** Mes deux balayages de rétractation
(grille 1000 sur `[0,200000]`, grille 500 sur `[0,400000]`) n'avaient
JAMAIS été sauvés en script permanent — violation directe de la règle
« les scripts vont dans le dépôt » (CLAUDE.md), et l'agent a dû
reconstruire ma méthodologie à l'aveugle à partir des conventions des
scripts voisins pour pouvoir la rejouer. En rejouant le système
IDENTIQUE à grille 1 (aucun sous-échantillonnage, 400000 pas, 304s),
l'agent trouve :
```
A_kick = 0,003669 ± 0,000205   (n=1672 événements, CV=5,6%)
         93% des 1672 événements dans [0,0034; 0,0040)
N_events([5500,400000)) = 1672  ->  espacement médian ≈ 462-500 pas
```
**Ce n'est PAS un bruit de magnitude variable toutes les 10000-30000
pas (ma première rétractation) — c'est un « kick » de magnitude
QUASI CONSTANTE (~0,0037, CV 5,6%) qui revient environ toutes les
~470 pas.** Sur les 14 points que j'avais flaggés dans ma première
rétractation, l'agent a mesuré le vrai pic local à ±500 pas de chacun :
LES 14 tombent tous dans la même bande `0,0037-0,0040` que partout
ailleurs — rien ne les distinguait, parce qu'il y a un kick presque
partout à cette fréquence et qu'une grille à 1000 pas peut à peine en
rater un. **Vérifié indépendamment par moi (pas juste accepté) :**
rejeu à grille 1 sur `[50000,55000)`, un écart de `427` pas trouvé
entre deux rafales de dépassement de seuil (très proche du `462-500`
annoncé), amplitudes mesurées `-0,00151` à `+0,00366` — cohérent avec
la gamme `~0,0037` rapportée. **Ma "distribution lisse bornée, pas de
queue lourde" et mon "toutes les 10000-30000 pas" étaient TOUS LES
DEUX des artefacts d'échantillonnage** — exactement le type de biais
que je venais de diagnostiquer et de corriger une fois, présent encore
un niveau plus bas dans ma propre méthode de correction. Ironie notée
explicitement par l'agent, vérifiée honnête.

**Mécanisme partiel proposé par l'agent (PAS encore vérifié
indépendamment par moi — à faire avant d'accepter comme confirmé, cf.
règle 5bis) : oscillateur de relaxation par plancher de `v`.**
`exp_avg_sq` (v) décroît géométriquement (taux `beta2=0,999`) vers un
plancher numérique pendant la phase calme ; une petite perturbation de
gradient contre un historique de variance minuscule produit un pas
normalisé démesuré (`lr*m/√v` bondit de `~4e-4` à `~2,6e-2` en ~14
pas, rapporté par l'agent sur la fenêtre `[51320,51500]`) ; le grand
gradient du kick lui-même regonfle `v`, amortissant le ratio, et `R4`
sonne en retour vers la baseline en 15-20 pas avec une oscillation
visiblement amortie. Cohérent avec des phénomènes déjà nommés dans le
projet (plancher `adam_eps`, biais de fenêtre Adam) mais à une échelle
de temps jamais regardée ici. **Statut : plausible, chiffres internes
(m, v) pas encore reproduits indépendamment par moi.**

**Test précommis par l'agent sur `beta2` — sharpen, pas résout, le
« ni confirmé ni réfuté » du 20/09 plus haut.** Si le mécanisme
plancher-de-`v` est juste, l'espacement devrait suivre
`~1/(1-beta2)`. Ancre `beta2=0,999` → espacement `462-470`. Rapporté :
```
beta2=0,999  n=230 (60000 pas)  espacement médian=462  prédit=470   (ancre)
beta2=0,995  n=725              espacement médian=110  prédit=94    (17% d'écart, même ordre)
beta2=0,99   n=286              espacement médian=163  prédit=47    (×3,5 d'écart, NON monotone)
```
La loi naïve tient de 0,999 à 0,995 (17%) puis CASSE entre 0,995 et
0,99 — l'espacement remonte (110→163) alors que la loi prédit une
poursuite de la baisse (94→47), un vrai changement de signe de
tendance, pas du bruit. **Localise la frontière de régime entre
`beta2=0,995` et `beta2=0,99`, non testée indépendamment par moi
encore.** Question ouverte posée par l'agent, pas encore répondue :
`ADAM_EPS=1e-10` est bien en dessous de `√v≈3e-7` mesuré au point
loggé — n'entre pas en jeu comme plancher ici ; tester avec
`adam_eps≈1e-6` (comparable au plancher de `√v`) pour trancher entre
la lecture « plancher de `v` » et une lecture concurrente « plancher
d'`eps` ».

**Script manquant à créer avant de clore ce fil** : ni le balayage de
rétractation ni celui de l'agent (grille 1, détection d'événements par
seuil+fusion) n'ont de script permanent committé — à faire au
prochain geste d'écriture, conformément à la règle « les scripts vont
dans le dépôt ».

| # | hypothèse | posée le | statut |
|---|---|---|---|
| l'amplitude des excursions est distribuée en continu (bornée, sans queue lourde), la fréquence est de l'ordre de 10000-30000 pas | 20/09 (moi) | **RÉTRACTÉE** le 20/09, même tour, par agent-dipankar puis vérification indépendante — c'est un kick de magnitude quasi constante (~0,0037, CV 5,6%) toutes les ~470 pas, pas une distribution ni un espacement de cet ordre |
| le mécanisme est un oscillateur de relaxation par plancher numérique de `v` (exp_avg_sq) | 20/09 (agent-dipankar) | **plausible, PAS encore vérifiée indépendamment** — chiffres internes (m,v) de l'agent pas rejoués par moi |
| l'espacement des kicks suit `~1/(1-beta2)` | 20/09 (agent-dipankar, précommise) | **partiellement confirmée puis réfutée** — tient de beta2=0,999 à 0,995 (17%), casse entre 0,995 et 0,99 (tendance inversée, ×3,5 d'écart) ; frontière de régime localisée, pas encore expliquée |

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
