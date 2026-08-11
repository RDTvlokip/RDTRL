# Test 3 — Efficacité communicative : conception, instruments, seuil

État : **instruments construits, étape 3 traitée, aucun entraînement lancé.** Ce
document fixe le dispositif et le critère de falsification **avant** d'écrire du
code, ce que les tests 1 et 2 n'ont jamais eu. Les étapes 1 à 3 de §7 existent
(`grammaire3.py`, `loi_nulle_longue.py`, `variabilite_du_maximum.py`,
`appariement_vs_distance.py`, `certificat_deux_agents.py`) ; les étapes 4 à 7
n'existent pas. Aucune concentration émergente n'a donc été mesurée à ce jour, ce
qui rend les corrections d'instrument du 11/08/2026 vérifiables : elles ne peuvent
pas avoir été choisies au vu d'un résultat.

---

## 1. La question, reformulée

Les tests 1 et 2 utilisaient tous deux une récompense qui est un **vérificateur
écrit à la main** — égalité de chaînes, puis parser. On mesurait donc la
spécification, pas l'agent. La reformulation qui supprime ce défaut :

> Un agent RL, initialisé avec des poids aléatoires et sans aucune donnée
> textuelle humaine, peut-il découvrir une **représentation linguistique
> générale** à partir d'un signal de récompense qui ne récompense pas la validité
> des phrases, mais **l'efficacité de la communication** ?

Le succès de tâche est vérifiable sans modèle de langue : le récepteur a
reconstruit le bon référent ou non. C'est le premier signal du projet qui
satisfait le critère « vérificateur sans modèle et non vacuel » de §5bis.6 du
carnet.

**Ce que ça ne supprime pas.** On déplace la spécification humaine du signal vers
le **monde** : on ne déclare plus ce qu'est une phrase valide, mais on choisit les
référents et leur structure d'attributs. Or c'est exactement cette structure qui
détermine si un code compositionnel *peut* émerger. Vérificateur sur les phrases,
dehors ; oracle sur le monde, dedans.

---

## 2. Principe de conception : les instruments d'abord

Ce qui a rendu le test 2 profond n'est pas la mesure, c'est que l'espace était
**énumérable**. Distribution exacte, optimum en forme close, gradient exact
calculable, paramétrisation tabulaire en contrôle, sonde de capacité. Sans ça il
ne restait qu'un chiffre.

La contrainte de dimensionnement du test 3 n'est donc pas le réalisme, c'est **que
ces instruments restent disponibles**.

### Dispositif retenu

Jeu de **reconstruction** plutôt que de discrimination : plus propre à énumérer.

| élément | valeur |
|---|---|
| attributs | 3 |
| valeurs par attribut | 3 |
| référents | **27** |
| longueur du message | 3 tokens |
| taille du vocabulaire | 3 |
| messages possibles | **27** |
| émetteur | loi 27 → 27, énumérable |
| récepteur | loi 27 → 27, énumérable |
| récompense | 1 si le récepteur reconstruit le référent, 0 sinon, partagée |
| hasard | 1/27 = 3,7 % |

Aucun langage humain nulle part. Les deux agents partent de poids aléatoires.

---

## 3. Un résultat obtenu avant toute implémentation

Un code parfait est une **bijection** entre les 27 référents et les 27 messages.

```
bijections parfaites          : 27! = 10 888 869 450 418 352 160 768 000 000 ≈ 1,089 × 10^28
bijections compositionnelles  : 3! (assignation des slots aux attributs)
                                × (3!)^3 (bijection valeur→token dans chaque slot)
                              = 6 × 216 = 1 296
fraction compositionnelle     : 1 296 / 27! ≈ 1,19 × 10^-25
```

**Toutes ces bijections rapportent exactement 1.** Elles sont à égalité parfaite.

**Correction du 11/08/2026 — le certificat invoqué ici ne s'applique pas.** J'avais
écrit que le certificat de §2.2 du carnet s'appliquait « tel quel ». C'est faux, et
§6.7 le vérifie (`certificat_deux_agents.py`).

Le certificat exige que **les objets à égalité soient le support de la loi dont
l'entropie figure dans l'objectif**. Au test 2 c'était le cas : les objets à
égalité étaient des séquences, et l'entropie portait sur la loi des séquences,
donc étaler la masse sur les optima était gratuit. Ici les objets à égalité sont
des **codes**, et aucune loi sur les codes n'apparaît dans l'objectif —
l'entropie porte sur les lignes de `S` et de `R`. La récompense est de plus une
récompense de **coordination**, donc étaler l'émetteur sur plusieurs codes casse
le décodage. Mesuré, en mélangeant K codes tirés au hasard des deux côtés :

| K | 1 | 2 | 3 | 5 | 10 | 27 |
|---|---|---|---|---|---|---|
| E[R] | 1,0000 | 0,5000 | 0,3416 | 0,2237 | 0,1511 | 0,0713 |

Il n'existe donc aucune loi optimale qui « charge également » les 27! optima : la
phrase n'a pas de sens dans ce cadre.

**Le chiffre survit, par un argument de symétrie qui est plus fort.** Renommer les
27 messages par une permutation π agit sur les codes par `c → π ∘ c`, et cette
action est **transitive** sur les 27! bijections — pour aller de `c₁` à `c₂`, il
suffit de prendre `π = c₂ ∘ c₁⁻¹`. Si la paramétrisation et l'initialisation sont
équivariantes sous ce groupe, alors les 27! codes sont **exactement**
équiprobables, sans aucune hypothèse de Gibbs.

> **À l'optimum, pour une paramétrisation tabulaire, la probabilité que le code
> soit compositionnel vaut exactement 1 296/27! ≈ 1,19 × 10⁻²⁵.**

C'est désormais un **théorème sur la paramétrisation**, pas une conséquence du
max-entropie, et il vaut pour tout algorithme équivariant. Le prix à payer est
qu'il ne vaut plus que pour le cas tabulaire : voir §6.7 pour ce qui se passe
quand la paramétrisation voit les tokens et les positions.

### Ce que ça signifie

La récompense est **indifférente à la compositionnalité**. Ce n'est pas que le RL
échoue à la trouver : rien dans l'objectif ne la demande. Elle ne peut venir que
d'une contrainte **extérieure à la récompense** — goulot d'étranglement de canal,
renouvellement de population, limite de mémoire, pression de longueur.

C'est le « pourquoi » du test 3, et il est démontrable sans lancer un seul
entraînement. Ce qui reste à mesurer devient plus intéressant qu'un pass/fail :

> **Quelle contrainte, et quelle quantité de contrainte, déplace cette probabilité
> de 10⁻²⁵ à quelque chose d'observable ?**

C'est une courbe, pas un verdict.

---

## 4. La formulation algébrique, et pourquoi elle rend tout exact

Notons `S` la matrice 27 × 27 de l'émetteur, `S[r, m] = P(message m | référent r)`,
et `R` celle du récepteur, `R[m, r̂] = P(reconstruction r̂ | message m)`. Les lignes
de chacune somment à 1.

Avec des référents tirés uniformément, la récompense espérée s'écrit exactement :

```
E[R] = (1/27) · Σ_r Σ_m S[r,m] · R[m,r] = (1/27) · tr(S R)
```

Trois conséquences immédiates, toutes utilisables comme instruments.

**a) L'objectif est bilinéaire.** Il est linéaire en `S` à `R` fixé, et
réciproquement. C'est un jeu de coordination pur : aucun conflit d'intérêt, un
seul optimum de valeur. Ça n'a rien à voir avec l'optimisation à un seul agent du
test 2, et ça se traite avec des outils différents.

**b) L'ensemble des optima globaux est exactement `{(P, Pᵀ) : P matrice de
permutation}`.** `tr(S R) ≤ 27` avec égalité si et seulement si `S` est une
permutation et `R` son inverse. C'est la reformulation rigoureuse du comptage de
§3 : il y a exactement 27! optima globaux, tous de valeur 1.

**c) Le gradient de l'un est la politique de l'autre.**

```
∂E[R] / ∂S[r,m] = (1/27) · R[m,r]
∂E[R] / ∂R[m,r] = (1/27) · S[r,m]
```

L'émetteur monte le gradient de la **table de décodage courante du récepteur**, et
inversement. C'est la structure de co-adaptation sous sa forme la plus nue, et
elle est exacte, pas approchée.

**Prédiction dérivable de (c), avant toute expérience** : à l'initialisation les
deux politiques sont quasi uniformes, donc chaque gradient est quasi uniforme,
donc **il n'existe aucune direction préférée**. La brisure de symétrie ne peut
venir que du bruit d'initialisation et de l'échantillonnage. Contraste net avec le
test 2, où le signal d'ordre 1 imposait une direction dès le premier pas à cause
d'un déséquilibre du lexique.

---

## 5. Ce qui remplace le seuil : une prédiction avec mécanisme

On abandonne délibérément le critère pass/fail. Raison : **le verdict est déjà
démontré**. Le comptage de §3 et le point (b) ci-dessus établissent que la
récompense est indifférente à la compositionnalité, sans qu'aucun entraînement
soit nécessaire. Un seuil empirique sur la précision zéro-shot ne ferait que
re-mesurer bruyamment ce qu'on sait déjà exactement.

Un seuil ne limite d'ailleurs pas ce qu'on mesure, seulement ce qu'on a le droit
de conclure. Ce qu'on garde à la place est plus contraignant et plus intéressant :

> **Engagement enregistré le 29/07/2026.** Les codes émergents seront des
> bijections quasi parfaites (succès de tâche élevé) et non compositionnels. La
> mesure de concentration positionnelle définie en §6.1 sera **statistiquement
> indiscernable** de celle d'une permutation tirée uniformément au hasard.

C'est falsifiable de façon bien plus tranchante qu'un seuil arbitraire. **Si la
concentration mesurée dépasse significativement le tirage uniforme, alors quelque
chose sélectionne les codes en dehors de la récompense** — et le raisonnement des
optima à égalité, qui porte tout le projet depuis le test 2, comporte une faille.
Ce serait le résultat majeur de RDTRL.

### Verdict sur cet engagement, 11/08/2026 : il était sous-spécifié

Il faut le dire avant de donner les chiffres, parce que c'est le défaut le plus
sérieux trouvé aujourd'hui, et il est de moi.

**L'engagement ne nomme pas la paramétrisation.** Or c'est elle qui décide. Mesuré
en §6.1 : `z = −0,12 ± 0,22` en tabulaire, `−0,25 ± 0,25` en factorisé,
`+9,92 ± 0,78` en structuré. Le même engagement est donc **confirmé sur deux
paramétrisations et réfuté sur une troisième**, et rien dans son énoncé ne permet
de dire laquelle il visait. Ce n'est ni une réussite ni un échec : c'est un critère
de falsification qui a omis la variable dont dépend la réponse.

**Et sa clause d'interprétation est fausse.** L'engagement dit qu'un dépassement
signifierait que « le raisonnement des optima à égalité comporte une faille ». Le
dépassement a lieu, et cette conclusion **ne suit pas**. §6.7 montre pourquoi : le
raisonnement était depuis toujours conditionnel à la symétrie de la
paramétrisation, ce que ni §3 ni §5 n'énonçaient. La récompense reste rigoureusement
indifférente à la compositionnalité ; c'est la paramétrisation qui sélectionne. La
clause a donc mal nommé ce qu'un dépassement prouverait.

**Sa première moitié est fausse aussi.** « Les codes émergents seront des bijections
quasi parfaites » : ils ne sont pas des bijections. Une sur vingt en tabulaire,
zéro sur vingt ailleurs, avec 2 à 5 collisions. Le succès de tâche est bien élevé
(E[R] ≈ 0,92), mais « bijection quasi parfaite » et « récompense élevée » ne sont
pas la même chose, et je les avais confondues en écrivant l'engagement.

Ce qui survit intact : **non compositionnels**, sur toutes les paramétrisations,
distance minimale au compositionnel de 13 référents sur 27 même dans le meilleur
cas.

---

## 6. Le programme d'investigation

Sept questions, dans l'ordre où chacune devient posable. Pour chacune :
l'instrument, et ce que chaque issue signifierait.

### 6.1 Quel code émerge, et où tombe-t-il parmi les 27! ?

**Instrument.** La loi jointe étant énumérable, on calcule la matrice d'information
mutuelle attribut × position :

```
M[i, j] = I(A_i ; M_j)     pour i, j ∈ {1, 2, 3}
```

Pour toute bijection, `I(référent ; message) = log 27 = 3 log 3`. Ce qui distingue
les codes, c'est la **répartition** de ces 3 log 3 dans la matrice 3 × 3.

- Code compositionnel : matrice diagonale à permutation près, `log 3` sur trois
  cases, 0 partout ailleurs.
- Code holistique : information étalée, chaque position portant un peu de chaque
  attribut.

D'où le scalaire :

```
concentration = ( Σ_j max_i M[i,j] ) / log₂(27)     ∈ [0, 1]
```

Majorée par 1 puisque `Σ_j max_i I(A_i;M_j) ≤ Σ_j H(M_j) ≤ 3·log₂3 = log₂27`.

**Correction d'une affirmation écrite sans vérification.** J'avais annoncé un
minorant de 1/3, en supposant implicitement que l'information mutuelle
s'additionnait à travers la matrice. C'est faux, et le calcul le montre : le
minimum observé sur 20 000 bijections uniformes est **0,0305**.

### Valeurs mesurées — `loi_nulle_longue.py`, 10 000 000 tirages

| | max par colonne | appariée |
|---|---|---|
| code compositionnel (les 1 296, vérifiés un par un) | **1,000000** | **1,000000** |
| bijection uniforme, moyenne | **0,1269** | 0,1168 |
| bijection uniforme, écart-type | 0,0330 | 0,0315 |
| quantile 99 % | 0,2161 | 0,2045 |
| quantile 99,9 % | 0,2525 | 0,2430 |
| quantile 99,99 % | 0,2862 | 0,2788 |
| quantile 99,999 % | 0,3196 | 0,3126 |
| maximum observé | 0,3979 | 0,3944 |

Un code compositionnel est à **26,5 écarts-types** de la moyenne nulle.

**Le seuil « ~0,35 » est retiré (11/08/2026).** Il figurait ici, dérivé du maximum
observé sur 20 000 tirages (0,3305). Deux raisons de le retirer, et la première
est la pire.

**1. Il contredisait §5.** Trois paragraphes plus haut, §5 écrit « on abandonne
délibérément le critère pass/fail » et enregistre un engagement portant sur une
**distribution** — indiscernabilité statistique. §6.1 réintroduisait un pass/fail
et se félicitait qu'il soit dérivé plutôt qu'arbitraire. Un seuil dérivé d'une
mauvaise ligne reste un mauvais seuil ; le défaut n'était pas l'arbitraire, c'était
le pass/fail lui-même, que §5 avait déjà écarté pour de bonnes raisons.

**2. Un maximum d'échantillon n'estime rien d'utile ici.** Les 1 296 codes
compositionnels **sont** des bijections : ils appartiennent à la loi nulle, avec
probabilité 1 296/27! ≈ 1,19 × 10⁻²⁵, et ils valent 1. Le supremum de la loi nulle
vaut donc exactement **1** — la valeur même qu'on voulait déclarer hors d'atteinte.
Le maximum d'échantillon n'estime pas un seuil, il estime 1, infiniment lentement.
Aucune taille de tirage n'y change quoi que ce soit.

Ça se voit sans théorie, en tirant douze blocs indépendants de 10 000 000
(`variabilite_du_maximum.py`) :

| ligne | moyenne sur 12 blocs | étendue entre blocs |
|---|---|---|
| moyenne | 0,1269 | 0,0000 |
| écart-type | 0,0330 | 0,0000 |
| quantile 99,9 % | 0,2527 | **0,0006** |
| quantile 99,99 % | 0,2863 | 0,0019 |
| **maximum** | 0,3950 | **0,0509** |

L'étendue du maximum vaut **1,54 écart-type de la loi nulle elle-même**, contre
0,02 pour le quantile 99,9 %. Un seuil posé sur cette ligne hérite de cette
dispersion : il mesure la patience du tirage, pas la loi.

**Ce qui le remplace** : on cite un quantile, qui est un estimateur. q99,9 % =
**0,2525**, stable à la quatrième décimale entre blocs. Et surtout, le seuil ne
servait à rien que §6.2 ne fasse mieux — voir le calcul de puissance là-bas.

### Le max par colonne compte parfois deux fois le même attribut

`concentration()` prend, colonne par colonne, l'attribut le mieux expliqué, sans
contrainte entre colonnes. Un même attribut peut donc gagner deux positions.
**C'est le cas dans 74,6 % des bijections uniformes** (Dipankar Sarkar, 11/08/2026,
reproduit ici). D'où une seconde statistique, `concentration_appariee()`, qui
impose un attribut par position — hongrois exact, six appariements en 3 × 3, donc
énumérés.

La question n'est pas de savoir laquelle est « la bonne » dans l'abstrait, mais
laquelle lit la structure que §6.1 prétend lire. Mesuré sur des codes dont la
structure est **connue par construction** — k positions sur 3 encodent proprement
un attribut, le reste étant brouillé conditionnellement (`appariement_vs_distance.py`) :

| k | valeur attendue | max par colonne | appariée | double compte |
|---|---|---|---|---|
| 0 | 0,0000 | 0,1268 | 0,1170 | 74,0 % |
| 1 | 0,3333 | 0,4111 | 0,4059 | 48,5 % |
| 2 | 0,6667 | 0,7045 | **0,7045** | 0,8 % |
| 3 | 1,0000 | 1,0000 | **1,0000** | 0,0 % |

Et le long d'une échelle allant du code compositionnel au code quelconque par
transpositions, l'écart entre les deux statistiques est **exactement nul jusqu'à
9 transpositions**, puis monte pour rejoindre 0,0103 à 21 — c'est-à-dire la valeur
qu'il prend sur la loi nulle.

**Conclusion, contre-intuitive et mesurée** : l'écart entre les deux statistiques
ne vit pas « au milieu de l'échelle ». Il vit **là où le code n'a aucune structure
positionnelle**, et il est nul partout où il y en a une à lire. Le double compte ne
peut donc pas fausser une position lue par §6.1 : il ne se produit pas dans cette
région. Le vérifier valait mieux que le supposer — le premier balayage, mené sur la
loi nulle, donnait l'inflation en hausse avec la concentration, et l'échelle donne
l'inverse au même niveau de concentration. Les deux sont vrais : l'inflation suit
la structure, pas le niveau.

Le pire cas existe quand même et il est borné (montée locale, donc minorants) :
l'écart maximal trouvé vaut **0,1443**, sur un code dont la concentration vaut
0,2473 — soit à l'intérieur du corps de la loi nulle, là où il n'y a de toute façon
rien à conclure. La plus haute concentration atteinte **avec** un double compte
vaut 0,6314, et ce code vaut encore 0,5560 en apparié : même poussé, le max ne
transforme pas un code sans structure en code structuré.

**Les deux sont publiées, et c'est délibéré.** La forme sans contrainte est celle
du standard du domaine — posdis (Chaabouni et coll. 2020) prend lui aussi l'argmax
indépendamment par position — donc la retirer coûterait la comparabilité. La forme
appariée est celle que §6.1 lit comme une position, parce qu'elle ordonne les codes
un peu mieux contre une vérité terrain combinatoire (86,82 % contre 86,34 % de
paires bien classées) et que, là où ça compte, elle est numériquement identique.

**Ce qu'on apprend.** Pas un binaire, une **position** dans l'espace des codes
parfaits. C'est la différence entre « ce n'est pas compositionnel » et « voici à
quelle distance et dans quelle direction ». Le sommet de cette échelle est isolé :
1,0000 pour un code compositionnel, puis **0,9294** pour le meilleur code non
compositionnel trouvé.

### Mesuré le 11/08/2026 — `code_emergent.py`

**La loi nulle est d'abord corrigée.** Les codes atteints n'étant pas bijectifs
(§6.5), la référence est tirée sur la classe réellement atteinte : à chaque run on
associe son **profil de fibres**, le multi-ensemble des tailles de préimages, et sa
nulle est tirée uniformément parmi les applications de **même profil**.

Ce n'est pas une référence « plus proche », c'est la seule correcte, et la raison
est exacte : le groupe `S₂₇ × S₂₇` agit par `(π, ρ)·c = π ∘ c ∘ ρ⁻¹`, et deux
applications sont dans la **même orbite si et seulement si** elles ont le même
profil de fibres. La paramétrisation tabulaire étant équivariante des deux côtés
(§6.7 pour les messages, §6.5 pour les référents), la loi de sortie conditionnée au
profil est **exactement uniforme sur ce profil**. Donc `z = 0` est ici un théorème,
pas une attente.

| paramétrisation | concentration appariée | z dans sa propre nulle | > q99,9 | distance au compositionnel |
|---|---|---|---|---|
| `tabulaire` | 0,1131 ± 0,0296 | **−0,12 ± 0,22** | 0 / 20 | 21,4 (min 20) |
| `factorise` | 0,1091 ± 0,0343 | **−0,25 ± 0,25** | 0 / 20 | 21,8 (min 20) |
| `structure` | 0,4240 ± 0,1056 | **+9,92 ± 0,78** | **19 / 20** | **15,8 (min 13)** |

**Le théorème tient, et pas seulement en moyenne.** Si la sortie est uniforme sur
l'orbite, les centiles des runs dans leur propre nulle doivent être uniformes sur
[0, 1]. Kolmogorov-Smirnov, n = 20 : `tabulaire` **D = 0,090, p ≈ 0,995** ;
`factorise` D = 0,226, p ≈ 0,225 ; `structure` D = 0,999, p ≈ 0. Et l'écart-type
des z vaut **0,97** en tabulaire — la nulle appariée a donc la bonne **forme**, pas
seulement la bonne moyenne, ce qui valide la construction entière.

**La distance au compositionnel confirme, sans information mutuelle.** 21,4 → 15,8,
minimum 13. Deux lectures indépendantes qui s'accordent.

**Et la correction de la loi nulle, sur laquelle j'ai insisté deux fois, ne change
rien.** Sur les onze profils rencontrés, la nulle appariée s'écarte de la bijective
de **−0,0001 à +0,0005**, quand l'effet mesuré vaut 0,30. Elle était nécessaire à
vérifier — sans quoi tout écart mesuré aurait été suspect — et elle ne déplace pas
la conclusion. C'est un résultat, pas un échec : il fallait le savoir, et il n'y
avait aucun moyen de le savoir sans le faire.

Ce qui **change** vraiment avec la non-bijectivité, ce n'est pas la nulle, c'est le
**choix de la statistique** : voir la section précédente, où un code dégénéré
obtient 1,0000 sous la version max.

### Tout ce qui précède suppose une bijection, et §6.7 dit que ce ne sera pas le cas

Ajouté le 11/08/2026, après §6.7, et ça change la conclusion précédente.

L'argument qui rendait le sommet de l'échelle sûr est : une concentration de 1
force chaque colonne à déterminer entièrement un attribut, et deux positions
déterminant le **même** attribut effondreraient neuf référents sur trois messages,
ce qu'une bijection ne peut pas faire. L'argument est juste. Sa prémisse ne l'est
pas dans le régime qui nous attend — §6.5 mesure que les codes atteints ont 1 à 4
collisions.

Le contre-exemple se construit à la main. Soit le code `m₁ = a₁`, `m₂ = a₁`,
`m₃ = a₂`, qui duplique le premier attribut sur deux positions et jette purement
et simplement le troisième :

```
messages distincts utilises : 9 sur 27          bijectif : non

I(A_i ; M_j) en bits        concentration MAX      : 1.000000
[[1.585 1.585 0.   ]        concentration APPARIEE : 0.666667
 [0.    0.    1.585]
 [0.    0.    0.   ]]
```

> **La statistique publiée décerne 1,0000 — le sommet réservé aux codes
> compositionnels — à un code qui jette un attribut sur trois et n'utilise qu'un
> tiers de l'espace des messages.** La statistique appariée rend 0,667, qui est la
> bonne réponse : deux attributs lus sur trois.

Le double compte n'est donc pas un défaut « du milieu de l'échelle » réservé aux
codes sans structure, comme la mesure sur les bijections le laissait croire. Dès
qu'on quitte les bijections, **il atteint le sommet**. Ça règle la question de
§6.1 sans discussion : la version appariée n'est pas une amélioration marginale de
0,48 point de concordance, c'est la seule des deux qui reste interprétable dans le
régime où l'expérience se trouvera.

Les trois bornes du paragraphe précédent — 0,1443 · 0,6314 · 0,9294 — ont été
obtenues par montée locale **sur des permutations**. Elles restent vraies sous
cette condition, et cessent de valoir hors d'elle.

**Changer l'instrument maintenant est gratuit, et ça ne le restera pas.** Aucun
entraînement du test 3 n'a été lancé : il n'existe aucune concentration émergente
mesurée. La même correction faite après un premier run serait invérifiable de
l'extérieur, et devrait être refusée.

### 6.2 La dynamique tire-t-elle vraiment au hasard parmi les codes parfaits ?

**La question la plus prometteuse du lot**, parce qu'elle teste l'hypothèse qui
porte tout le reste.

**Le piège à éviter.** Tester directement « P(compositionnel) = 1,19 × 10⁻²⁵ » n'a
aucune puissance statistique : sous l'hypothèse nulle on attend zéro succès en
cinquante runs, et on en observera zéro. Le test ne peut rien distinguer.

**L'instrument qui a de la puissance.** Construire la **loi nulle de la
concentration** en tirant N permutations uniformément parmi les 27!, et calculer
la distribution de la statistique de §6.1. Puis comparer la concentration des
codes réellement émergents à cette loi nulle.

- Concentration des runs indiscernable de la nulle → la dynamique échantillonne
  bien uniformément, le calcul de §3 tient, prédiction confirmée.
- Concentration **supérieure** → quelque chose sélectionne hors récompense.
  Candidats : la factorisation autorégressive de l'émetteur, l'ordre de génération
  des tokens, la structure de partage de paramètres. C'est exactement ce qui
  s'était passé au test 2, où le biais d'ordre 1 décidait la branche.
- Concentration **inférieure** → la dynamique fuit activement les codes
  structurés, ce qui serait très surprenant et demanderait une explication.

Le modèle est minuscule, donc 50 à 100 graines sont accessibles. C'est ce qui rend
ce test faisable ici et nulle part ailleurs.

**Ce que cet instrument résout, chiffré avant de le lancer** (Dipankar Sarkar,
11/08/2026, reproduit ici). Test unilatéral à p < 0,001 et 80 % de puissance, donc
δ = 3,93 · σ/√n :

| graines | max par colonne | appariée |
|---|---|---|
| 50 | 0,0184 | 0,0175 |
| 100 | **0,0130** | 0,0124 |

À comparer à ce que le seuil retiré exigeait : **0,223 sur un seul run**. À 100
graines, le test distributionnel est donc **dix-sept fois plus sensible** que le
seuil qui figurait à côté de lui. Et il détecte la bonne chose : une pression
extérieure faible a bien plus de chances de soulever tous les runs de 0,02 — un
résultat à six sigma ici, invisible pour le seuil — que d'en projeter un seul
au-delà de 0,35.

**La statistique appariée est le bon choix pour ce test, et l'argument n'est pas
celui qu'on croit.** Sous l'hypothèse nulle enregistrée en §5 — les codes émergents
sont des bijections quelconques — les deux statistiques sont décalées de la même
quantité et le changement s'annule exactement. Mais sous l'alternative, si une
pression crée de la structure positionnelle, le double compte disparaît : l'écart
de 0,0101 s'applique alors à la loi nulle et **pas** à la population émergente.
Changer de statistique baisse la référence sans bouger le signal. C'est neutre là
où ça doit l'être, et favorable là où on veut de la puissance.

### Mesuré le 11/08/2026 — `dynamique_uniforme.py`, 100 graines

§6.1 avait répondu à 20 graines. C'était insuffisant, et il faut le dire : sous le
critère du document lui-même (unilatéral p < 0,001, puissance 80 %), 20 graines ne
résolvent que **0,027**. Le scénario « une pression faible soulève tous les runs de
0,02 » y serait passé inaperçu. Conclure « indiscernable » à 20 graines aurait été
une conclusion que le dispositif ne portait pas.

| paramétrisation | n | concentration appariée | z moyen | IC 95 % | KS *p* | > q99,9 |
|---|---|---|---|---|---|---|
| `tabulaire` | 100 | 0,1164 | **−0,01 ± 0,10** | [−0,21 ; +0,19] | 0,386 | 0 / 100 |
| `factorise` | 100 | 0,1152 | **−0,05 ± 0,10** | [−0,25 ; +0,15] | 0,613 | 0 / 100 |
| `structure` (témoin) | 20 | 0,3971 | **+9,01 ± 0,60** | [+7,84 ; +10,18] | 0,000 | 20 / 20 |

**Le négatif, énoncé avec sa borne.** « On n'a rien vu » ne veut rien dire sans dire
ce qu'on aurait vu. À 100 graines, avec un écart-type de nulle de 0,0312 :

| critère | 20 graines | 100 graines |
|---|---|---|
| bilatéral p < 0,05, puissance 80 % | 0,0195 | **0,0087** |
| unilatéral p < 0,001, puissance 80 % (celui du tableau ci-dessus) | 0,0274 | **0,0123** |

> Toute sélection résiduelle par la dynamique, sur une paramétrisation équivariante,
> est **plus petite que 0,0087 de concentration**. Ce n'est pas une absence, c'est
> une borne.

**Le balayage en β, parce qu'un seul β n'est pas une propriété.** La région
[0,037 ; 0,170] étant bistable (§6.7), rien ne garantissait que ce qui vaut à 0,02
vaille dans tout le régime de code. 20 graines par β, tabulaire :

| β | 0,005 | 0,010 | 0,020 | 0,030 | 0,037 |
|---|---|---|---|---|---|
| E[R] | 0,8870 | 0,9074 | — | 0,9166 | 0,9314 |
| collisions | 2,95 | 2,45 | — | 2,20 | 1,75 |
| z moyen | +0,12 ± 0,15 | +0,41 ± 0,24 | −0,01 ± 0,10 | +0,12 ± 0,27 | +0,02 ± 0,22 |
| KS *p* | 0,342 | 0,070 | 0,386 | 0,519 | 0,999 |

Aucun β ne sort. Au passage, une observation qui n'était pas cherchée : **monter β
jusqu'au seuil améliore le code**, E[R] de 0,887 à 0,931 et collisions de 2,95 à
1,75. L'entropie aide la coordination tant qu'elle ne détruit pas le code.

### Comment lire ce résultat, et comment ne pas le lire

Ce n'est **pas** « la dynamique tire au hasard ». C'est :

> La dynamique tire au hasard **sur l'orbite**, quand la paramétrisation est
> équivariante.

Le profil de fibres, lui, n'est pas tiré au hasard du tout — c'est la dynamique qui
le choisit, et c'est précisément pourquoi on conditionne dessus. Un test qui ne
conditionnerait pas mesurerait le choix du profil et l'appellerait sélection de
code.

Et les trois issues que §6.2 avait listées sont toutes réalisées, selon la
paramétrisation : indiscernable pour les deux équivariantes, supérieure pour la
structurée. La troisième — « la dynamique fuit activement les codes structurés » —
n'est réalisée nulle part : les z négatifs observés (−0,01 et −0,05) sont à moins
d'un demi écart-type de zéro.

### 6.3 Qui écrit le code, l'émetteur ou le récepteur ?

**Instrument.** Quatre conditions, en gelant l'un des deux :

| condition | question |
|---|---|
| `S` gelé compositionnel, `R` libre | le récepteur apprend-il à décoder un code structuré ? (attendu : oui, c'est un problème d'apprentissage supervisé déguisé) |
| `R` gelé compositionnel, `S` libre | l'émetteur retrouve-t-il l'encodeur correspondant ? |
| `S` gelé sur une permutation aléatoire, `R` libre | le récepteur apprend-il aussi bien un code arbitraire ? |
| les deux libres | ce qui émerge réellement |

**Ce qu'on apprend.** Si un code compositionnel s'apprend aussi facilement qu'un
code arbitraire une fois l'autre agent gelé, alors la difficulté n'est pas dans
l'apprentissage du code mais dans la **coordination** sur lequel choisir. C'est
une localisation, pas un constat — la version test 3 du gel de position qui avait
tout localisé au test 2.

### Mesuré le 11/08/2026 — `qui_ecrit_le_code.py`

**Un agent gelé n'a pas de paramétrisation** : il est représenté par une matrice
stochastique fixe. Sans ça, « geler sur un code aléatoire » serait impossible à
poser pour la paramétrisation structurée, qui ne sait pas écrire la plupart des
bijections (§6.5), et on confondrait une limite de représentabilité du **gelé**
avec une difficulté d'apprentissage du **libre**.

| condition | pas pour 99 % de sa propre valeur finale | E[R] final |
|---|---|---|
| S gelé compositionnel, R libre | 139 | 0,99992302 |
| S gelé aléatoire, R libre | 139 | 0,99992303 |
| S gelé à 2 collisions, R libre | 139 | 0,9259 |
| R gelé compositionnel, S tabulaire libre | 139 | 0,99992302 |
| R gelé aléatoire, S tabulaire libre | 139 | 0,99992302 |
| R gelé compositionnel, S `structure` libre | 148 | 0,99991 |
| R gelé aléatoire, S `structure` libre | 308 | **0,5924** |
| les deux libres, S tabulaire | 260 | 0,9111 ± 0,0296 |
| les deux libres, S `structure` | 216 | 0,8777 ± 0,0525 |

**La réponse à la question du titre est : ni l'un ni l'autre.** Geler l'émetteur et
laisser apprendre le récepteur, ou l'inverse, donne **139 pas dans les deux sens**
et la même valeur finale **à huit décimales**. Le problème est exactement
symétrique — c'est la bilinéarité de §4 rendue visible. Aucun des deux agents
n'écrit le code ; il est écrit par la coordination.

**Et geler sur le compositionnel ou sur un code quelconque est le même problème**,
à 6 × 10⁻⁹ près pour le récepteur et 8 × 10⁻¹⁰ pour l'émetteur tabulaire. Les deux
codes sont reliés par un renommage des messages, et un agent tabulaire est
équivariant : c'est encore §6.7, sous une troisième forme.

### Le déficit n'est pas dans l'apprentissage, il est dans le code choisi

C'est le résultat de cette section, et **mon premier libellé disait l'inverse**.
J'avais écrit « coût de la coordination = 0,049 », en comparant la paire libre à un
agent gelé sur une **bijection**. Deux défauts, tous deux miens :

1. un code à *k* collisions plafonne **arithmétiquement** à (27 − *k*)/27 — deux
   référents envoyés sur le même message sont indistinguables, quoi que fasse le
   récepteur. Comparer une paire à 2,4 collisions à un agent qui a reçu une
   bijection compare deux plafonds, pas deux apprentissages ;
2. mon seuil de vitesse était « pas pour atteindre 0,99 », **inatteignable dès la
   première collision**. Il mesurait une capacité en croyant mesurer une vitesse.
   Remplacé par « 99 % de sa propre valeur finale ».

Le plafond est **vérifié et non supposé** : gelé sur un code à 2 collisions,
l'agent libre atteint 0,9259, soit exactement 25/27, à −0,0000 de la prédiction.

| les deux libres | E[R] | collisions | plafond | **E[R] / plafond** |
|---|---|---|---|---|
| S tabulaire | 0,9111 | 2,40 | 0,9110 | **1,0000** |
| S `structure` | 0,8777 | 3,30 | 0,8777 | **1,0000** |

> La paire libre exécute son code **exactement aussi bien** qu'un agent à qui on
> aurait donné ce même code tout fait. Le déficit n'est pas dans l'apprentissage :
> il est entièrement dans le code sur lequel les deux se posent.

La coordination coûte donc en **vitesse** (260 pas contre 139) et en **qualité du
code atteint** (2,4 collisions au lieu de 0), mais **rien** en exécution une fois
le code choisi. C'est la localisation que §6.3 cherchait.

**Une condition où l'échec change de nature.** `R gelé aléatoire, S structure
libre` plafonne à 0,5924, et `R gelé à 2 collisions` à 0,4990. C'est la première
condition du test 3 où un échec vient de la **représentabilité** et non de la
coordination : l'émetteur structuré ne sait pas écrire l'encodeur demandé. §6.5
l'avait mesuré en supervisé, on le retrouve ici dans le jeu.

### 6.4 Que voit le gradient au premier pas, avec un partenaire qui apprend aussi ?

**Instrument.** Le point (c) de §4 donne le gradient exact et gratuit. On le
calcule à l'initialisation, on mesure sa norme et son alignement avec les
directions compositionnelles.

**Ce qu'on apprend.** L'analogue de l'analyse d'ordre 1 du test 2, mais dans un
cadre où la « fonction » de récompense bouge. Prédiction déjà dérivée en §4 :
aucune direction préférée à l'initialisation. Si on observe le contraire, c'est
que la paramétrisation en introduit une, et il faudra la nommer.

### Mesuré le 11/08/2026 — `gradient_premier_pas.py`

**L'instrument.** Pour un code `c`, sa vraisemblance jointe
`L(c) = Σ_r log S[r, c(r)] + Σ_r log R[c(r), r]`, et on mesure le **cosinus entre
∇J et ∇L(c) dans l'espace des paramètres**. C'est exactement « le premier pas
rapproche-t-il du code `c` ? », et c'est la bonne question à poser à une
paramétrisation, qui vit dans l'espace des poids et non dans celui des lois.

**La prédiction de §4 tient.** Coefficient de variation du gradient dans l'espace
des lois : **1,0 × 10⁻²** pour `∂E[R]/∂S`, **9,8 × 10⁻³** pour `∂E[R]/∂R`. Aucune
direction préférée. C'est le contraste net avec le test 2, où le déséquilibre du
lexique imposait une direction dès le premier pas.

**Et la prédiction que j'avais ajoutée est fausse.** J'avais écrit, avant de
mesurer, que la paramétrisation structurée devait préférer le code compositionnel
dès le premier pas, puisqu'elle y va à z = +9,9 à convergence. Mesuré :

| paramétrisation | cos compositionnel | cos témoins | z | centile |
|---|---|---|---|---|
| `tabulaire` | 0,004243 | 0,000902 ± 0,0111 | **+0,32 ± 0,22** | 0,579 |
| `structure` | 0,033347 | 0,041210 ± 0,0536 | **−0,08 ± 0,24** | 0,459 |

Rigoureusement rien, des deux côtés. **La préférence de la paramétrisation
structurée n'est pas dans son gradient initial.**

### Alors quand apparaît-elle ? Entre le pas 10 et le pas 30

Même mesure à plusieurs profondeurs, en repartant de la **même** initialisation à
chaque fois. z du code compositionnel contre 100 bijections témoins, 10 graines :

| pas | 0 | 10 | 30 | 100 | 300 | 1 000 | 3 000 |
|---|---|---|---|---|---|---|---|
| `tabulaire` | +0,07 | +0,07 | −0,30 | +0,30 | +0,19 | +0,16 | +0,19 |
| `structure` | −1,18 | −0,29 | **+4,36** | +4,25 | +3,91 | +5,81 | +5,85 |

> La paramétrisation tabulaire ne préfère **jamais** le code compositionnel, à
> aucune profondeur. La structurée ne le préfère pas non plus au départ, puis s'y
> met **brutalement entre le pas 10 et le pas 30**, et n'en bouge plus.

Le mécanisme se laisse nommer, ce que §6.4 demandait explicitement : près de
l'uniforme, la contrainte de la paramétrisation **ne mord pas**, puisque toute loi
est représentable à faible confiance. Elle n'apparaît qu'à mesure que la loi se
concentre, et le seuil est franchi en quelques dizaines de pas.

### L'issue est-elle déjà écrite dans l'initialisation ?

On entraîne, on lit le code atteint, puis on revient à l'initialisation mesurer son
alignement — contre des témoins **appariés au profil de fibres** du code atteint,
faute de quoi on mesurerait l'effet du profil en l'appelant prédictibilité.

| paramétrisation | z du code atteint | centile | argmax initial conservé |
|---|---|---|---|
| `tabulaire` | **+6,80 ± 0,18** | 1,000 | 8,7 % ± 5,5 |
| `structure` | −0,52 ± 0,18 | 0,333 | 3,7 % ± 3,7 |

**Deux lectures, et il faut les concilier plutôt que choisir la plus flatteuse.**
Le hasard vaut 1/27 = 3,7 %. En tabulaire, l'initialisation porte une empreinte
réelle mais **modeste** : assez pour classer le code final **premier sur 300**
alternatives appariées, pas assez pour le lire dans les poids — 8,7 %, soit 2,3
référents sur 27. Dire « l'issue est décidée à l'initialisation » serait donc
exagéré. La formulation juste est : *l'initialisation biaise fortement en agrégat,
sans écrire le code*.

En structuré, l'empreinte est **exactement nulle** (3,7 %, le hasard). Tout vient
de la trajectoire.

C'est le miroir du test 2, où l'initialisation décidait le coin et la trajectoire
son remplissage (carnet §7.11sexies). Ici le partage dépend de la paramétrisation,
et il s'inverse d'une paramétrisation à l'autre.

### 6.5 Représentable, atteignable, stable — les trois séparément

Au test 2 ces trois réponses étaient **différentes**, et c'est précisément ce qui a
fait basculer tout le verdict : le modèle pouvait représenter l'optimum, pouvait à
peu près s'y maintenir, et ne pouvait pas l'atteindre.

**Instrument.** On construit à la main le code compositionnel canonique
(`S = P`, `R = Pᵀ`, où `P` encode l'attribut *i* dans le token *i*), puis :

1. **Représentable** — ajustement supervisé des deux réseaux vers ce code. Y
   arrivent-ils exactement ? (sonde de capacité, purement diagnostique)
2. **Atteignable** — l'entraînement depuis l'aléatoire y arrive-t-il jamais ?
3. **Stable** — en démarrant *dessus*, REINFORCE y reste-t-il, ou dérive-t-il vers
   un attracteur comme les 45,3 modes du test 2 ?

**Ce qu'on apprend.** Trois échecs possibles qui portent le même nom et appellent
des remèdes opposés. Si le code compositionnel est instable, aucune initialisation
intelligente ne sauvera quoi que ce soit.

### Traité le 11/08/2026 — `representable_atteignable_stable.py`

§6.7 avait rendu cette étape décisive plutôt que descriptive, en fournissant une
prédiction à réfuter : sous paramétrisation tabulaire, la montée est équivariante
sous le renommage des messages, lequel est transitif sur les 27! bijections, donc
**le code compositionnel doit se comporter exactement comme une bijection tirée au
hasard**. Pas « à peu près » : à la précision machine.

**Le contraste est construit pour n'avoir qu'une seule différence** : même
objectif, même optimiseur, même récepteur tabulaire, et surtout même expressivité
pour les deux premières paramétrisations. Seule la carte des paramètres change.

| paramétrisation | ce que voit l'émetteur | poids |
|---|---|---|
| `tabulaire` | le référent comme **indice**, 27 → 27 | 729 libres |
| `factorise` | idem, mais `p(m₁) p(m₂\|m₁) p(m₃\|m₁,m₂)` par référent | 1 053 libres |
| `structure` | le référent par ses **attributs**, poids **partagés** | 81 + 9 |

**1. Représentable.** Ajustement supervisé vers un code imposé, 3 000 pas :

| paramétrisation | compositionnel | aléatoire (3 codes) | écart |
|---|---|---|---|
| `tabulaire` | 0,99947 en **2 198** pas | 0,99947 en **2 198** pas | −1,1 × 10⁻⁷ |
| `factorise` | 0,99939 en 2 365 pas | 0,99939 en 2 364–2 365 pas | −1,1 × 10⁻⁷ |
| `structure` | 0,99939 en 2 367 pas | **0,11573**, jamais atteint | **+0,884** |

Les deux premières lignes sont l'équivariance rendue visible **jusque dans le
nombre de pas**, identique à l'unité près. La troisième dit que la
représentabilité, pour une paramétrisation structurée, **n'est pas la même pour
tous les codes** — ce qui est exactement ce que §6.5 cherchait. Contrôle : poussé à
20 000 pas et lr 0,2, `structure` atteint 1,00000 sur le compositionnel en **704
pas** et plafonne à 0,09–0,24 sur les bijections quelconques. C'est une limite de
capacité, pas un échec d'optimisation.

**2. Stable.** On repart de l'état ajusté :

| paramétrisation | code | E[R] exact | E[R] REINFORCE | conservé |
|---|---|---|---|---|
| `tabulaire` | compositionnel | 1,000000 | 0,996297 | oui |
| `tabulaire` | aléatoire | 1,000000 | 0,996238 | oui |
| `factorise` | compositionnel | 1,000000 | 0,996168 | oui |
| `factorise` | aléatoire | 1,000000 | 0,996151 | oui |
| `structure` | compositionnel | 1,000000 | 0,997763 | oui |
| `structure` | aléatoire | **0,740621** | **0,704479** | **non** |

Écart compositionnel − aléatoire, montée exacte : **−2,3 × 10⁻¹⁰** en tabulaire,
**−2,4 × 10⁻¹⁰** en factorisé, **+0,235** en structuré.

**3. Atteignable.** Depuis l'aléatoire, 20 graines, β = 0,02 :

| paramétrisation | E[R] moyen | bijections | collisions | concentration appariée |
|---|---|---|---|---|
| `tabulaire` | 0,9240 | **1 / 20** | 2,00 | 0,1283 ± 0,0405 |
| `factorise` | 0,8092 | 0 / 20 | 5,15 | 0,1270 ± 0,0379 |
| `structure` | 0,8573 | 0 / 20 | 3,80 | **0,4233 ± 0,1233** |

### Ce que ça dit, et ce que ça ne dit pas

**Ce que ça dit.** Le contraste `factorise` − `tabulaire` vaut **−0,0013** : deux
paramétrisations très différentes, indiscernables. Le contraste `structure` −
`tabulaire` vaut **+0,2950**, soit 7,3 écarts-types. Récompense identique,
objectif identique, optimiseur identique : **la paramétrisation décide seule**.

**Et ça corrige §6.7 sur un point que j'avais manqué.** J'y avais raisonné sur le
renommage des **messages**. Mais `c → c ∘ ρ⁻¹`, le renommage des **référents**, est
lui aussi transitif sur les 27! bijections. L'équivariance d'**un seul des deux
côtés** suffit donc à égaliser tous les codes. C'est pourquoi `factorise`, malgré
sa factorisation en tokens, ne préfère rien : ses paramètres sont indexés par
référent, sans partage. Conséquence pratique, et elle n'est pas anodine : **une
table d'embedding libre par référent annule d'avance tout ce que la structure du
message pourrait apporter.** La plupart des implémentations feraient ça sans le
savoir.

**Ce que ça ne dit pas, et c'est important.** `structure` **ne peut pas** écrire la
plupart des bijections. Trouver qu'il produit des codes structurés est donc en
grande partie une conséquence de ce qu'il peut écrire, et non une émergence. Ce
n'est pas « la compositionnalité est apparue » : c'est une **contrainte de
capacité**, exactement ce que §6.6 prévoyait, mesurée ici contre une ligne de base
calculée exactement plutôt que contre une intuition.

Trois nuances qui empêchent de surinterpréter :

1. `structure` **n'atteint pas** un code compositionnel. Il rendrait 1,0000 ; il
   rend 0,4233. La contrainte produit de la structure **partielle**.
2. Elle la paie : E[R] tombe de 0,9240 à 0,8573, et les collisions passent de 2,00
   à 3,80. C'est la taxe de mise en forme de §2.3 du carnet, dans un autre décor.
3. La ligne « loi nulle appariée = 0,1168 ± 0,0315 » n'est **pas** une référence
   valide ici, les codes atteints n'étant pas bijectifs. Le contraste entre
   paramétrisations reste valide, lui, puisque les trois populations sont
   comparées entre elles avec la même statistique.

### 6.6 La courbe qui remplace le verdict

**La vraie mesure du test 3.** Puisque la récompense est indifférente à la
compositionnalité, celle-ci ne peut venir que d'une contrainte **extérieure**. La
question devient quantitative : combien de contrainte, et laquelle ?

| contrainte | bouton | pourquoi ça devrait marcher |
|---|---|---|
| bruit de canal | probabilité ε qu'un token soit corrompu | un code compositionnel ne perd qu'un attribut quand un token est corrompu ; un code holistique perd tout |
| goulot de vocabulaire | `V^L` juste au-dessus de 27, puis en dessous | force la réutilisation des tokens, donc la structure |
| renouvellement de population | période K de remplacement d'un agent par un agent neuf | un code structuré se réapprend plus vite, donc survit aux générations |
| pression de longueur | pénalité sur la longueur du message | favorise les codes réguliers |

On trace **concentration en fonction du bouton**, avec la loi nulle de §6.2 en
référence horizontale. C'est la courbe qui remplace le pass/fail.

**Filiation à citer, ce n'est pas nouveau.** Le renouvellement de population
produisant de la compositionnalité, c'est l'*iterated learning* (Kirby et coll.).
La robustesse au bruit favorisant la structure est un argument classique en
théorie du codage. Ce qui serait nouveau ici, ce n'est pas le phénomène : c'est de
le mesurer **contre une ligne de base calculée exactement** plutôt que contre une
intuition.

### Mesuré le 11/08/2026 — `courbe_de_contrainte.py`

**La justification écrite dans la table ci-dessus est fausse, et ça se voit sans
entraîner quoi que ce soit.** « Un code compositionnel ne perd qu'un attribut quand
un token est corrompu ; un code holistique perd tout » — vrai en information, sans
effet sur cette récompense. Pour un émetteur déterministe sur un code `c` et le
décodeur optimal, `E[R]* = (1/27) Σ_m' max_r C[c(r), m']`, et `c` étant une
bijection sur les 27 messages, `max_r C[c(r), m'] = max_m C[m, m']` :
**indépendant de `c`**.

| ε | 0,00 | 0,05 | 0,10 | 0,20 | 0,30 | 0,50 |
|---|---|---|---|---|---|---|
| compositionnel − 200 bijections | 0 | −1,1e−16 | −1,1e−16 | 0 | +1,1e−16 | −5,6e−17 |

Perdre « un seul attribut » ne rapporte rien quand le crédit est tout-ou-rien sur
le référent exact.

**Mais le canal brise bel et bien la symétrie**, et c'est ce qui rend l'expérience
intéressante. À ε = 0,2, l'écart maximal entre `C` et sa version permutée vaut
**0,00e+00** sur le groupe structurel et au minimum **0,050** sur 200 permutations
quelconques. Donc :

> Le certificat des optima à égalité continue de dire que **rien** ne distingue les
> bijections en récompense, exactement, à tout ε. Et le théorème d'équivariance de
> §6.7 ne s'applique plus. Toute sélection observée opérerait donc **entièrement
> hors de la récompense** — le cas le plus pur que ce banc pouvait produire.

**Et il ne se passe rien.** 15 graines par ε, émetteur tabulaire :

| ε | 0,00 | 0,05 | 0,10 | 0,20 | 0,30 | 0,50 |
|---|---|---|---|---|---|---|
| E[R] | 0,9333 | 0,8344 | 0,7598 | 0,6104 | 0,4840 | 0,0370 |
| concentration appariée | 0,1030 | 0,1152 | 0,1080 | 0,1286 | 0,1216 | 0,1160 |
| **z** | −0,44 ± 0,20 | −0,05 ± 0,22 | −0,28 ± 0,30 | **+0,38 ± 0,29** | +0,15 ± 0,26 | −0,02 ± 0,20 |
| > q99,9 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |

Aucun ε ne sort. À 15 graines la borne est |z| < 0,72, soit 0,024 de concentration —
à comparer aux **+9,9** que la paramétrisation structurée produit en §6.1, un
facteur quatorze. À ε = 0,5 le système s'effondre au babil pur (E[R] = 1/27, 9,4
collisions) : le canal détruit le code avant de le structurer.

> **Briser la symétrie est nécessaire, et pas suffisant.** C'est un affaiblissement
> de l'hypothèse unificatrice que j'avais tirée de §6.7 le matin même.

**Le renouvellement de population ne fait rien non plus, et c'était prédit.**
Remplacer un récepteur tabulaire par un neuf est une opération échangeable, donc
l'équivariance sous `S₂₇` **survit** et le théorème de §6.7 s'applique encore :

| période | aucun | 1 000 | 300 | 100 |
|---|---|---|---|---|
| E[R] | 0,9333 | 0,9476 | 0,9449 | 0,8821 |
| z | +0,33 ± 0,29 | +0,17 ± 0,25 | −0,34 ± 0,25 | −0,15 ± 0,27 |

Prédiction confirmée. Si l'*iterated learning* produit de la compositionnalité, ça
ne peut donc pas venir du renouvellement seul — il faut un biais inductif du
réapprenant ou un goulot structuré. Je le formule comme une conclusion sur **ce
banc** ; je n'ai pas fait la revue de littérature.

### Ce qu'il aurait fallu, calculé exactement

Le mécanisme de la table existe — il demande une récompense à **crédit partiel par
attribut**, où le décodeur est noté sur le nombre d'attributs qu'il retrouve. Alors
l'égalité se brise, et largement :

| ε | 0,05 | 0,10 | 0,20 | 0,30 | 0,50 |
|---|---|---|---|---|---|
| compositionnel − aléatoire, crédit partiel | +0,034 | +0,063 | **+0,108** | +0,138 | +0,153 |

**Mais ça déplace la compositionnalité dans la spécification.** Une récompense par
attribut dit à l'agent que les attributs comptent séparément, ce qui est
exactement l'information qu'un code compositionnel encode. C'est le « oracle sur le
monde, dedans » de §1, d'un cran plus profond : non plus dans le choix des
référents, mais dans la fonction de récompense elle-même.

### La conclusion de §6.6, et elle est plus dure que la courbe attendue

De tout ce qui a été testé aujourd'hui, **une seule chose a produit de la
compositionnalité : la paramétrisation** — et elle l'a fait en rendant les
alternatives inécrivables, pas en les départageant. La seule contrainte
d'environnement qui marcherait le fait en mettant la préférence dans la récompense.

> Sur ce banc, la compositionnalité n'a jamais été **sélectionnée**. Elle a été
> soit impossible, soit spécifiée.

### 6.7 Le certificat des optima à égalité survit-il à un jeu à deux agents ? — **TRAITÉ le 11/08/2026**

C'était « la question la plus inconfortable, et je ne connais pas la réponse ».
Réponse : **le certificat ne survit pas**, il est remplacé par un argument plus
fort, et ce remplacement désigne l'endroit exact où la compositionnalité peut
naître. `certificat_deux_agents.py`, aucun entraînement, tout en calcul exact.

**Ce qui casse, et pas pour la raison que j'avais écrite.** J'avais soupçonné le
terme d'entropie, qui porte sur deux politiques séparément. Le vrai défaut est
plus simple : le certificat exige que les objets à égalité soient **le support de
la loi dont l'entropie est dans l'objectif**. Au test 2, les objets à égalité
étaient des séquences et l'entropie portait sur les séquences. Ici les objets à
égalité sont des codes, et il n'y a aucune loi sur les codes dans l'objectif.
Mesuré en §3 : mélanger K codes fait chuter `E[R]` comme 1/K. Les 27! optima ne
sont pas occupables ensemble.

**Ce qui remplace le certificat.** L'action `c → π ∘ c` du renommage des messages
est transitive sur les 27! bijections. Une paramétrisation **tabulaire** avec
initialisation échangeable est équivariante sous `S₂₇` tout entier, donc les 27!
codes sont exactement équiprobables. Vérifié numériquement, montée de gradient
exacte donc déterministe : **8 essais sur 8** rendent exactement `π ∘ c` après
renommage des messages. Avec échantillonnage, l'équivariance devient
distributionnelle et non plus exacte par run — ce qui est précisément ce que teste
§6.2, une loi sur 50 à 100 graines.

### Le diagramme de phase, et un seuil en forme close

Deux seuils, et il fallait les deux : un seul aurait masqué que le système est
**bistable** entre les deux.

**β_c = 1/27**, où le babil cesse d'être instable. Dérivé par linéarisation de la
meilleure réponse `S[r,m] ∝ exp(R[m,r]/β)` autour de l'uniforme : une perturbation
du récepteur revient multipliée par `(1/(27β))²`, donc l'aller-retour est
contractant dès que `27β > 1`.

Mesuré, 8 départs par β, montée de gradient exacte :

| β | 0,010 | 0,020 | 0,030 | 0,035 | 0,037 | 0,040 | ≥ 0,050 |
|---|---|---|---|---|---|---|---|
| quitte le babil | 100 % | 100 % | 100 % | 100 % | **100 %** | **0 %** | 0 % |
| E[R] moyen atteint | 0,907 | 0,931 | 0,935 | 0,921 | 0,926 | 0,037 | 0,037 |

**Le seuil est exact, et c'est le hessien qui le dit, pas la dynamique.** Une
bissection sur la montée donnait 0,0381, soit 3 % au-dessus de la prédiction, et
réduire la perturbation de 10⁻² à 10⁻⁵ ne refermait pas l'écart (0,0383 · 0,0381 ·
0,0382 · 0,0375). Ce n'était donc pas la taille de la perturbation. La plus grande
valeur propre du hessien au point de babil — gradient nul à 2,7 × 10⁻²⁰, donc point
critique — croise zéro en **0,037037037**, à 2,4 × 10⁻¹¹ de 1/27.

> Le 0,0381 mesurait **Adam**, pas l'objectif. Adam normalise ses pas, donc il ne
> ralentit pas là où le gradient s'annule et quitte un maximum local que
> l'objectif dit stable. Mesurer un seuil de stabilité à travers un optimiseur,
> c'est mesurer l'optimiseur.

**Le second seuil, β ≈ 0,17**, où un code cesse d'être un maximum local. J'avais
prédit 0,1461 en égalisant `J = 1` d'un code **pur** et `J = 1/27 + 2β ln 27` du
babil. La prédiction est un minorant : la branche optimisée garde de l'entropie et
vaut donc plus que 1 (J = 1,0089 à β = 0,146), donc elle survit au-delà. Mesuré
par bissection : **0,1701**.

Entre les deux, sur `β ∈ [0,040 ; 0,146]` d'après la grille, **les deux sont des
maxima locaux** et l'issue dépend entièrement de l'initialisation.

### Et un résultat de §6.5 qui arrive avec deux étapes d'avance

Les valeurs atteintes depuis le babil ne sont pas quelconques : 0,8518, 0,8888,
0,9259, 0,9629, 1,0000. Soit exactement **23/27, 24/27, 25/27, 26/27, 27/27**.

> **La montée de gradient exacte, sans le moindre échantillonnage, ne rejoint
> presque jamais un code parfait depuis le babil.** Un départ sur 40. Elle se pose
> sur des codes où 1 à 4 référents entrent en collision. Partie **sur** un code
> parfait, elle y reste à E[R] = 1,0000.

C'est la séparation *atteignable ≠ stable* de §6.5, obtenue ici gratuitement, et
sans pouvoir l'imputer au bruit d'échantillonnage puisqu'il n'y en a aucun.

**Conséquence de méthode, à traiter avant §6.1.** La loi nulle de §6.1 est tirée
**sur des bijections**, et les deux statistiques de concentration s'appuient sur
cette hypothèse — le chemin vectorisé de `loi_nulle_longue.py` suppose les deux
marges uniformes, ce qui n'est vrai que pour une bijection, et rend sinon des
nombres faux **sans lever d'erreur** (mesuré : 0,110573 au lieu de 0,108071, soit
0,0025, un cinquième de ce que §6.2 doit résoudre). Une garde a été ajoutée.
Mais le fond reste : comparer un code émergent non bijectif à une loi nulle
bijective compare deux supports différents. Trois issues, et la troisième est celle
à retenir :

1. ne garder que les runs atteignant une bijection parfaite — ça vide l'expérience,
   c'est 1 sur 40 ;
2. garder la nulle bijective telle quelle — malhonnête, l'écart va dans le sens du
   résultat prédit ;
3. **tirer la loi nulle sur la classe réellement atteinte** : codes uniformes parmi
   ceux ayant le même nombre de collisions que le run auquel on les compare.
   Appariable run par run, et toujours énumérable.

C'est un **théorème sur la paramétrisation**, pas un résultat d'optimisation : il
tient pour tout algorithme équivariant. Conséquence directe pour le plan
d'expériences : **§6.1 et §6.2 sur un émetteur tabulaire ne peuvent rien
découvrir**. Leur issue est connue d'avance. Ils gardent une valeur, mais comme
**détecteur de bogue** — si la mesure s'écarte du hasard, c'est l'implémentation
qui a cassé la symétrie, pas le monde.

**Où la garantie tombe, et le compte qui tombe juste.** Renommer les messages
n'agit naturellement sur les paramètres que si le renommage respecte la
décomposition en `(m₁, m₂, m₃)`. Ces permutations forment un groupe dont l'ordre
exact, **compté par retour arrière et pas seulement construit**, vaut **1 296** —
les « lignes » du monde étant exactement les triangles du graphe de Hamming
H(3,3), préserver la structure de produit revient à préserver l'adjacence, et
l'énumération est alors complète. Le groupe tombe donc de 27! ≈ 1,09 × 10²⁸ à
1 296, et il n'est plus transitif.

Et le compte tombe juste :

> **Les 1 296 codes compositionnels sont exactement l'orbite du code canonique
> sous ce groupe d'ordre 1 296.** Vérifié : les deux ensembles, construits par
> deux chemins de code indépendants, sont identiques.

Autrement dit, la seule paramétrisation dont le groupe de symétrie est plus petit
que `S₂₇` est précisément celle dont le groupe de symétrie **distingue les codes
compositionnels**.

**Le piège où je suis tombé en écrivant ce fichier, et qui vaut d'être noté.**
J'avais d'abord écrit que la ligne d'un émetteur autorégressif est une loi produit.
C'est faux : `P(m₁)·P(m₂|m₁)·P(m₃|m₁,m₂)` représente **n'importe quelle** loi sur
les 27 messages. L'argument ne porte pas sur l'**expressivité** mais sur la
**symétrie de la paramétrisation** : un π quelconque ne correspond à aucune
permutation des poids, donc l'équivariance tombe même à expressivité pleine. C'est
exactement la forme de l'argument du test 2, où l'effondrement de mode venait de la
factorisation et non de l'objectif. Le 1 296 est donc un **majorant** pour toute
paramétrisation qui voit tokens et positions ; une architecture concrète peut être
bien moins symétrique — un émetteur récurrent à couche de sortie partagée n'admet
même pas les permutations de positions. Le majorant suffit, puisqu'il suffit de
perdre la transitivité.

**Ce que ce résultat ne dit pas.** Perdre la transitivité ne **prédit pas** que les
codes compositionnels reçoivent plus de masse. Ça retire seulement la garantie
qu'ils en reçoivent exactement 1 296/27!. C'est une **impossibilité** d'un côté
(tabulaire ⟹ hasard, quoi qu'on fasse) et une simple ouverture de l'autre. Le sens
du déplacement reste entièrement à mesurer, et c'est §6.2.

---

## 7. Ordre de construction

**Attention : ce n'est pas l'ordre du §6.** Le §6 range les questions par
dépendance conceptuelle, pour qu'elles se comprennent en se lisant à la suite.
L'ordre ci-dessous est celui d'exécution, et il diffère pour deux raisons : ce qui
peut invalider le reste passe en tête, ce qui coûte cher passe en dernier.

| étape | à construire | question traitée |
|---|---|---|
| 1 | `grammaire3.py` : les 27 référents, les 27 messages, le comptage exact des bijections et des codes compositionnels, la matrice d'information mutuelle, la statistique de concentration | — infrastructure |
| 2 | la loi nulle : distribution de la concentration sur des permutations tirées uniformément, en long, avec la statistique appariée et la dispersion du maximum entre blocs | — référence de tout le reste |
| 3 | ~~vérification du certificat des optima à égalité en cadre à deux agents~~ **FAIT le 11/08/2026** (`certificat_deux_agents.py`) : le certificat ne survit pas, un argument d'équivariance le remplace | **§6.7** |
| 4 | ~~sonde de capacité et code compositionnel construit à la main~~ **FAIT le 11/08/2026** (`representable_atteignable_stable.py`) : les trois réponses sont différentes, et seule une paramétrisation à poids partagés voyant les attributs les sépare | **§6.5** |
| 5 | ~~entraînement multi-graines~~ **§6.1 et §6.2 FAITS le 11/08/2026** (`code_emergent.py`, `dynamique_uniforme.py`), loi nulle appariée au profil de fibres, 100 graines, balayage en β | **§6.1** puis **§6.2** |
| 6 | ~~gel d'agent, analyse du gradient initial~~ **§6.3 et §6.4 FAITS le 11/08/2026** (`qui_ecrit_le_code.py`, `gradient_premier_pas.py`) : ni l'un ni l'autre n'écrit le code, le déficit est dans le code choisi, et la préférence de la paramétrisation structurée apparaît entre le pas 10 et le pas 30 | **§6.3** puis **§6.4** |
| 7 | ~~courbe de contrainte~~ **FAIT le 11/08/2026** (`courbe_de_contrainte.py`) : ni le bruit de canal ni le renouvellement ne produisent quoi que ce soit, et la justification écrite dans la table de §6.6 était fausse pour une récompense tout-ou-rien | **§6.6** |

Soit, en termes de questions : **6.7 → 6.5 → 6.1 → 6.2 → 6.3 → 6.4 → 6.6.**

**Pourquoi §6.7 remontait en troisième position.** C'était la seule question dont
une réponse négative invalidait tout le reste du document. Si le certificat des
optima à égalité ne tenait pas dans un jeu à deux agents, le calcul des 10⁻²⁵ de §3
était faux et il fallait le refaire avant de lancer le moindre entraînement. La
faire en dernier revenait à construire six expériences sur une hypothèse non
vérifiée.

**Ce que ça a rapporté, le 11/08/2026.** Le certificat ne tenait effectivement pas.
Le chiffre survit par un argument de symétrie, mais l'ordre du programme change :

- **§6.1 et §6.2 sur un émetteur tabulaire ne peuvent rien découvrir** — leur
  résultat est un théorème. Ils restent utiles comme détecteurs de bogue, pas comme
  mesure. L'expérience réelle est le contraste tabulaire / structuré, et c'est ce
  contraste qu'il faut construire, pas un entraînement de plus.
- **§6.5 est partiellement répondu sans être lancé** : atteignable ≠ stable, avec
  gradient exact et sans échantillonnage.
- **Une correction est due avant §6.1** : la loi nulle est bijective, les codes
  atteints ne le sont pas.
- **§6.6 reçoit une hypothèse unificatrice** : bruit de canal, goulot de
  vocabulaire, pression de longueur et renouvellement de population brisent tous
  `S₂₇` vers un groupe respectant la structure de produit, dont les codes
  compositionnels sont exactement une orbite. À vérifier contrainte par contrainte,
  mais ça remplace une liste de recettes par une question unique.

  **Corrigé le soir même par §6.6.** L'hypothèse est fausse sur deux points. Le
  renouvellement de population **ne brise pas** `S₂₇` du tout, étant échangeable.
  Et le bruit de canal le brise sans produire quoi que ce soit : z reste nul à tous
  les ε. **Briser la symétrie est nécessaire, pas suffisant.**

**Pourquoi §6.6 descend en dernier.** C'est la plus coûteuse — un balayage complet
par contrainte, sur plusieurs graines — et elle n'a aucun sens tant que la ligne de
base de §6.2 n'existe pas : on mesurerait un déplacement sans savoir par rapport à
quoi.
