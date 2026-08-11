# Test 3 — Efficacité communicative : conception, instruments, seuil

État : **instruments construits, aucun entraînement lancé.** Ce document fixe le
dispositif et le critère de falsification **avant** d'écrire du code, ce que les
tests 1 et 2 n'ont jamais eu. Les étapes 1 et 2 de §7 existent (`grammaire3.py`,
`loi_nulle_longue.py`, `variabilite_du_maximum.py`, `appariement_vs_distance.py`) ;
les étapes 3 à 7 n'existent pas. Aucune concentration émergente n'a donc été
mesurée à ce jour, ce qui rend les corrections d'instrument du 11/08/2026
vérifiables : elles ne peuvent pas avoir été choisies au vu d'un résultat.

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

Le certificat de §2.2 du carnet s'applique donc tel quel : sous max-entropie, des
optima à égalité sont équiprobables. **À l'optimum de l'objectif, la probabilité
que le code soit compositionnel est de l'ordre de 10⁻²⁵.**

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

### 6.4 Que voit le gradient au premier pas, avec un partenaire qui apprend aussi ?

**Instrument.** Le point (c) de §4 donne le gradient exact et gratuit. On le
calcule à l'initialisation, on mesure sa norme et son alignement avec les
directions compositionnelles.

**Ce qu'on apprend.** L'analogue de l'analyse d'ordre 1 du test 2, mais dans un
cadre où la « fonction » de récompense bouge. Prédiction déjà dérivée en §4 :
aucune direction préférée à l'initialisation. Si on observe le contraire, c'est
que la paramétrisation en introduit une, et il faudra la nommer.

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

### 6.7 Le certificat des optima à égalité survit-il à un jeu à deux agents ?

**La question la plus inconfortable, et je ne connais pas la réponse.**

Tout le raisonnement de §3 importe un résultat du test 2 : *sous max-entropie, des
optima à récompense égale sont équiprobables*. Ce résultat a été établi pour un
**agent unique** optimisant `E[R] + β·H`.

Ici l'objectif conjoint est `(1/27)·tr(S R) + β·(H(S) + H(R))`. Rien ne garantit
que son optimum soit encore une loi de Gibbs sur les paires, ni que les 27! optima
globaux restent équiprobables. Le terme d'entropie porte sur **deux** politiques
séparément, pas sur la distribution jointe des messages.

**Instrument.** L'espace étant énumérable, on peut résoudre numériquement le
problème `max (1/27)·tr(S R) + β·(H(S)+H(R))` sur les matrices stochastiques, par
montée de gradient exacte en paramétrisation tabulaire, et **regarder à quoi
ressemble l'optimum**. Si les 27! solutions n'y sont pas équiprobables, alors
mon calcul des 10⁻²⁵ repose sur une hypothèse fausse et il faut le refaire.

**C'est le premier endroit du projet où un résultat que j'ai publié pourrait
s'effondrer sur un point technique que je n'ai pas vérifié.** Il passe donc en
tête de liste, avant tout entraînement.

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
| 3 | vérification du certificat des optima à égalité en cadre à deux agents, paramétrisation tabulaire et gradient exact | **§6.7** |
| 4 | sonde de capacité et code compositionnel construit à la main | **§6.5** |
| 5 | entraînement multi-graines | **§6.1** puis **§6.2** |
| 6 | gel d'agent, analyse du gradient initial | **§6.3** puis **§6.4** |
| 7 | courbe de contrainte | **§6.6** |

Soit, en termes de questions : **6.7 → 6.5 → 6.1 → 6.2 → 6.3 → 6.4 → 6.6.**

**Pourquoi §6.7 remonte en troisième position.** C'est la seule question dont une
réponse négative invalide tout le reste du document. Si le certificat des optima à
égalité ne tient pas dans un jeu à deux agents, le calcul des 10⁻²⁵ de §3 est faux
et il faut le refaire avant de lancer le moindre entraînement. La faire en dernier
reviendrait à construire six expériences sur une hypothèse non vérifiée.

**Pourquoi §6.6 descend en dernier.** C'est la plus coûteuse — un balayage complet
par contrainte, sur plusieurs graines — et elle n'a aucun sens tant que la ligne de
base de §6.2 n'existe pas : on mesurerait un déplacement sans savoir par rapport à
quoi.
