# Changelog — RDTRL

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versionnage manuel.

**DOI de concept, toutes versions confondues :**
[10.5281/zenodo.21726216](https://doi.org/10.5281/zenodo.21726216) — c'est celui
du badge et du BibTeX, il résout toujours vers la version la plus récente. Les
DOI de version, propres à une release donnée et figés, sont indiqués sous chaque
entrée ci-dessous.

## [Non publié]

### Corrigé — la revue de littérature, faite après la 0.5.0 et pas avant

Carnet §7.23. C'était le dernier blocage de fond identifié, et il fallait le lever :
l'article 3 écrivait « someone has very likely written it down » à propos du
théorème d'équivariance. Quelqu'un l'a écrit — **Kuciński, Korbak, Kołodziej et
Miłoś, NeurIPS 2021** ([arXiv:2111.06464](https://arxiv.org/abs/2111.06464)).

- **Leur théorème 1 est le no-go de §6.7**, côté loi des données plutôt que côté
  carte des paramètres. L'argument de symétrie est publié depuis 2021.
- **Leur théorème 2 explique §6.6 et le confirme** : la compositionnalité devient
  optimale sous deux conditions conjointes, une perte **factorisée** par trait et
  un canal bruité. Le négatif de §6.6 est leur première condition manquante, et le
  +0,108 mesuré sous crédit partiel est la contrepartie empirique de leur théorème.
  Leur « le bruit est nécessaire mais pas suffisant » est mot pour mot §1.16.
- **Ce qui survit comme mien** : une ligne de base calculée exactement plutôt
  qu'une régularité empirique, le corollaire au niveau de l'architecture, et les
  huit hypothèses mortes.
- **Les notes de la 0.5.0 ne sont pas réécrites.** Elles sont déposées sur Zenodo
  sous [10.5281/zenodo.21895549](https://doi.org/10.5281/zenodo.21895549) ; la
  revue leur est postérieure, et corriger un artefact archivé serait pire que le
  laisser daté. L'article 3, lui, n'était pas publié : il est corrigé.

## [0.5.0] — 2026-08-12

DOI de version : [10.5281/zenodo.21895549](https://doi.org/10.5281/zenodo.21895549)

**La version où le test 3 a enfin tourné.** Conçu, discuté et critiqué sans avoir
jamais été exécuté, son programme d'investigation en sept questions est passé de 0
à 7 en une journée. Notes de version : [docs/RELEASE_v0.5.0.md](docs/RELEASE_v0.5.0.md).
Récit complet : [article 3](https://huggingface.co/blog/RDTvlokip/i-made-my-world-small-enough-to-compute-everything), mirroré dans [docs/ARTICLE3.md](docs/ARTICLE3.md).

Le résultat central est que le certificat des optima à égalité **ne survit pas** à
un jeu à deux agents, qu'un argument d'équivariance le remplace en donnant le même
chiffre pour tout algorithme équivariant, et que **seule la paramétrisation** a
produit de la compositionnalité — en rendant les alternatives inécrivables, pas en
les départageant. Le constat non prévu est que huit hypothèses sont mortes en un
jour dans un monde entièrement énumérable, **aucune par erreur de calcul**.

### Ajouté — article 3

[« I made my world small enough to compute everything exactly. It caught none of
my eight mistakes »](https://huggingface.co/blog/RDTvlokip/i-made-my-world-small-enough-to-compute-everything),
mirroré dans [docs/ARTICLE3.md](docs/ARTICLE3.md). Couvre le programme
complet du test 3, les sept questions traitées en un jour, le théorème
d'équivariance et son corollaire sur les tables d'embedding par référent, et le
constat qui n'était pas prévu : **huit hypothèses mortes, aucune arithmétique**.
L'URL de l'article 2 est par ailleurs câblée dans le README, `CITATION.cff` et
`.zenodo.json`, qui pointaient encore tous sur l'article 1 ou sur le miroir local.

### Corrigé — le test 3 perd son seuil, avant d'avoir jamais tourné

Cinquième série de critiques de
[dipankarsarkar](https://orcid.org/0000-0001-5431-6367), cette fois sur
[docs/TEST3.md](docs/TEST3.md), qui décrit un test **non implémenté**. Carnet
§7.14, hypothèses mortes §1.9 et §1.10.

- **Le seuil « ~0,35 » de §6.1 est retiré.** Il était dérivé du maximum observé
  sur 20 000 tirages de la loi nulle. Deux raisons, et la première est la pire :
  §5 abandonnait explicitement le critère pass/fail trois paragraphes plus haut,
  et §6.1 en réintroduisait un. Ensuite, un maximum d'échantillon n'estime rien
  ici — les 1 296 codes compositionnels **sont** des bijections, donc le supremum
  de la loi nulle vaut exactement 1. Douze blocs indépendants de 10 000 000
  donnent des maxima de 0,3775 à 0,4283, étendue 1,54 écart-type de la loi ;
  q99,9 % varie de 0,0006 sur les mêmes blocs. Remplacé par un quantile.
- **La loi nulle passe de 20 000 à 10 000 000 de tirages**, queue exacte et non
  échantillonnée. Une seule ligne du tableau bouge, celle qui portait le seuil.
- **`concentration_appariee()` ajoutée** : un attribut par position, hongrois
  exact sur six appariements. `concentration()` prend le max colonne par colonne
  sans contrainte, donc un attribut peut gagner deux positions — 74,6 % des
  bijections uniformes. Les deux restent publiées : la forme sans contrainte est
  celle du standard du domaine (posdis, Chaabouni et coll. 2020), l'appariée est
  celle que §6.1 lit comme une position.
- **Mesure qui corrige ma propre lecture** : l'écart entre les deux statistiques
  ne vit pas « au milieu de l'échelle » mais dans la région **sans structure**. Il
  est exactement nul jusqu'à 9 transpositions d'un code compositionnel, et nul à
  k = 2 et k = 3 positions propres. Il ne peut donc pas fausser §6.1.
- **§6.2 reçoit son calcul de puissance** : à 100 graines le test distributionnel
  résout un déplacement de 0,0130, quand le seuil retiré exigeait 0,223 sur un
  seul run — dix-sept fois plus sensible, et sur la bonne alternative.

### Corrigé — §6.7 traité : le certificat des optima à égalité ne survit pas

L'étape 3 du plan du test 3, celle dont une réponse négative invalidait tout le
reste. Elle est négative. Carnet §7.15, hypothèses mortes §1.11 et §1.12.

- **Le certificat de §2.2 ne s'applique pas à un jeu à deux agents.** Il exige que
  les objets à égalité soient le support de la loi dont l'entropie est dans
  l'objectif. Au test 2 c'étaient des séquences ; ici ce sont des codes, et aucune
  loi sur les codes n'apparaît dans l'objectif. Mesuré : mélanger K codes fait
  chuter `E[R]` comme 1/K.
- **Le 1,19 × 10⁻²⁵ survit par un argument de symétrie**, qui est plus fort — c'est
  un théorème sur la paramétrisation, valable pour tout algorithme équivariant — et
  plus étroit, puisqu'il ne vaut que pour le cas tabulaire. Vérifié 8 fois sur 8.
- **Le groupe de l'émetteur structuré a pour ordre exactement 1 296**, compté par
  retour arrière et non seulement construit, et **les 1 296 codes compositionnels
  sont exactement l'orbite du code canonique sous ce groupe**.
- **Conséquence sur le plan d'expériences** : §6.1 et §6.2 sur un émetteur
  tabulaire ne peuvent rien découvrir, leur issue est un théorème. Ils deviennent
  des détecteurs de bogue. L'expérience réelle est le contraste tabulaire/structuré.
- **β_c = 1/27 en forme close**, confirmé par le hessien au point de babil
  (croisement en 0,037037037). La bissection sur la montée donnait 0,0381 : elle
  mesurait Adam, dont les pas normalisés ne ralentissent pas là où le gradient
  s'annule. Second seuil à 0,1701, et région **bistable** entre les deux.
- **Résultat de §6.5 obtenu avec deux étapes d'avance** : la montée exacte, sans
  aucun échantillonnage, ne rejoint un code parfait depuis le babil qu'une fois sur
  quarante. Elle se pose sur des codes à 1 à 4 collisions (23/27 à 26/27).
- **Garde ajoutée dans `loi_nulle_longue.py`** : le chemin vectorisé suppose les
  marges uniformes, vrai seulement pour une bijection, et rendait sinon des nombres
  faux sans lever d'erreur (0,110573 au lieu de 0,108071). Les codes émergents
  n'étant pas bijectifs, la loi nulle devra être tirée sur la classe réellement
  atteinte.

### Corrigé — §6.5 traité, et il complète §6.7

Étape 4 du plan du test 3. Carnet §7.16, hypothèse morte §1.13.

- **Les trois réponses sont différentes**, comme au test 2. Représentable : oui
  pour tous, mais **pas pour tous les codes** dès que la paramétrisation est
  structurée. Atteignable : 1 bijection sur 20. Stable : oui partout, sauf un code
  aléatoire sous paramétrisation structurée.
- **§6.7 était incomplet** : `c → c ∘ ρ⁻¹`, le renommage des **référents**, est lui
  aussi transitif sur les 27! bijections, donc l'équivariance d'**un seul des deux
  côtés** suffit à égaliser tous les codes. Une table d'embedding libre par
  référent annule d'avance tout ce que la structure du message pourrait apporter.
  Mesuré : la paramétrisation autorégressive par référent ne préfère rien
  (écart 3,3 × 10⁻¹⁶, puis −0,0013 sur 20 graines).
- **Un émetteur voyant les attributs avec poids partagés déplace la concentration
  de 0,1283 à 0,4233**, soit 7,3 écarts-types, à récompense et objectif identiques.
  Mais il ne peut pas écrire la plupart des bijections (contrôle de capacité à
  20 000 pas), donc c'est une contrainte de capacité et non une émergence, et il
  n'atteint pas le code compositionnel : il s'arrête à 0,4233, en payant 0,067
  d'E[R].
- **Le sommet de l'échelle n'est sûr que pour les bijections.** Le code
  `m₁ = a₁, m₂ = a₁, m₃ = a₂`, qui jette un attribut et n'utilise que 9 messages
  sur 27, obtient **concentration max = 1,000000** contre 0,666667 en apparié. Les
  codes atteints n'étant pas bijectifs, la version appariée devient la seule
  interprétable.

### Corrigé — §6.1 traité, et mon critère de falsification était sous-spécifié

Étape 5 du plan du test 3, précédée de la correction de loi nulle annoncée en
§6.7. Carnet §7.17, §7.18 et **§4.7**.

- **La loi nulle est appariée au profil de fibres**, et c'est exact plutôt
  qu'approché : le groupe `S₂₇ × S₂₇` agissant par `(π, ρ)·c = π ∘ c ∘ ρ⁻¹`, deux
  applications sont dans la même orbite **si et seulement si** elles ont le même
  profil. La sortie tabulaire y est donc uniforme par théorème, et `z = 0` cesse
  d'être une attente.
- **Le théorème tient en distribution** : Kolmogorov-Smirnov sur les centiles des
  runs dans leur propre nulle donne D = 0,090 et p ≈ 0,995 en tabulaire, et
  l'écart-type des z vaut 0,97. La nulle appariée a donc la bonne forme, pas
  seulement la bonne moyenne.
- **Résultats** : `z = −0,12 ± 0,22` en tabulaire, `−0,25 ± 0,25` en factorisé,
  `+9,92 ± 0,78` en structuré, avec 19 runs sur 20 au-delà du quantile 99,9 %. La
  distance de Hamming au compositionnel, qui n'utilise aucune information mutuelle,
  confirme : 21,4 → 15,8.
- **La correction de loi nulle ne change rien**, et il fallait le vérifier pour le
  savoir : sur les onze profils rencontrés elle s'écarte de la bijective de −0,0001
  à +0,0005, quand l'effet vaut 0,30.
- **L'engagement enregistré le 29/07/2026 était sous-spécifié sur trois points** :
  il ne nomme pas la paramétrisation, qui décide ; sa clause d'interprétation
  attribue un dépassement à une faille du raisonnement des optima à égalité, ce qui
  ne suit pas ; et sa première moitié, « des bijections quasi parfaites », est
  fausse — une bijection sur vingt. Enregistrer une prédiction protège de
  l'ajustement après coup, pas d'avoir omis une variable ni d'avoir écrit d'avance
  la mauvaise interprétation.
- **Ce que ça ne prouve pas** : le `z = +9,92` n'est pas une émergence. Cette
  paramétrisation ne peut pas écrire la plupart des bijections, n'atteint pas le
  code compositionnel, et paie sa structure en succès de tâche.

### Corrigé — §6.2 traité à 100 graines, la taille annoncée par le document

Carnet §7.19.

- **20 graines ne suffisaient pas**, et §6.1 avait conclu dessus. Sous le critère
  du document (unilatéral p < 0,001, puissance 80 %), 20 graines ne résolvent que
  0,027 : le scénario « une pression faible soulève tous les runs de 0,02 » y
  serait passé inaperçu.
- **À 100 graines** : `z = −0,01 ± 0,10` en tabulaire, `−0,05 ± 0,10` en factorisé,
  0 run sur 100 au-delà du quantile 99,9 % de chaque côté, KS *p* = 0,386 et 0,613.
  Témoin structuré : `z = +9,01 ± 0,60`, 20 runs sur 20 au-delà.
- **Le négatif est énoncé avec sa borne** : toute sélection résiduelle par la
  dynamique, sur paramétrisation équivariante, est plus petite que **0,0087** de
  concentration (bilatéral p < 0,05, puissance 80 %), ou 0,0123 sous le critère
  strict du document.
- **Balayage en β** sur 0,005 à 0,037, 20 graines chacun : aucun β ne sort. Et une
  observation non cherchée — monter β jusqu'au seuil **améliore** le code, E[R] de
  0,887 à 0,931 et collisions de 2,95 à 1,75.
- **La phrase juste est plus étroite** que « la dynamique tire au hasard » : elle
  tire au hasard **sur l'orbite**, quand la paramétrisation est équivariante. Le
  profil de fibres, lui, est choisi par la dynamique, d'où le conditionnement.

### Corrigé — §6.3 traité : personne n'écrit le code

Carnet §7.20.

- **Ni l'émetteur ni le récepteur.** Geler l'un ou l'autre donne **139 pas dans
  les deux sens** et la même valeur finale **à huit décimales**. Le problème est
  exactement symétrique. Et geler sur le compositionnel ou sur une bijection
  quelconque est le même problème, à 6 × 10⁻⁹ près.
- **Le déficit n'est pas dans l'apprentissage mais dans le code choisi.** Un code
  à *k* collisions plafonne arithmétiquement à (27 − *k*)/27, vérifié et non
  supposé : gelé sur 2 collisions, l'agent libre atteint exactement 25/27. Rapporté
  à ce plafond, la paire libre obtient **E[R]/plafond = 1,0000** dans les deux
  paramétrisations. La coordination coûte en vitesse (260 pas contre 139) et en
  qualité du code atteint, et **rien** en exécution.
- **Deux défauts de mesure corrigés avant publication** : je comparais la paire
  libre à un agent gelé sur une *bijection*, donc deux plafonds et non deux
  apprentissages ; et mon seuil de vitesse (« pas pour atteindre 0,99 ») est
  inatteignable dès la première collision, donc mesurait une capacité en croyant
  mesurer une vitesse. Remplacé par « 99 % de sa propre valeur finale ».
- **Un agent gelé est une matrice fixe, sans paramétrisation**, sans quoi geler sur
  un code aléatoire serait impossible pour la paramétrisation structurée et l'on
  confondrait représentabilité du gelé et apprentissage du libre.

### Corrigé — §6.4 traité : le gradient initial ne voit rien, la préférence naît au pas 30

Carnet §7.21, hypothèse morte §1.14.

- **La prédiction de §4 tient** : coefficient de variation du gradient dans l'espace
  des lois de 1,0 × 10⁻², donc aucune direction préférée à l'initialisation —
  contrairement au test 2 où le déséquilibre du lexique en imposait une dès le pas 1.
- **Ma prédiction supplémentaire est fausse.** J'avais écrit avant mesure que la
  paramétrisation structurée préférerait le code compositionnel dès le premier pas,
  puisqu'elle y va à z = +9,9 à convergence. Mesuré : **z = −0,08 ± 0,24**.
- **La courbe qui répare la réfutation** : z passe de −1,18 au pas 0 à **+4,36 au
  pas 30**, et n'en bouge plus, quand la tabulaire reste à 0 à toute profondeur.
  Près de l'uniforme la contrainte ne mord pas, toute loi étant représentable à
  faible confiance ; elle apparaît quand la loi se concentre.
- **L'empreinte de l'initialisation** : en tabulaire le code atteint est classé
  **premier sur 300** témoins appariés au profil, mais seuls **8,7 %** de ses
  référents sont l'argmax des poids initiaux (hasard 3,7 %). L'initialisation
  biaise fortement en agrégat sans écrire le code. En structuré, l'empreinte est
  exactement nulle.

### Corrigé — §6.6 traité : la seule chose qui a marché est la paramétrisation

Dernière étape du programme du test 3. Carnet §7.22, hypothèses mortes §1.15 et
§1.16.

- **La justification écrite dans ma table de §6.6 est fausse**, et ça se démontre
  sans entraîner : pour un émetteur déterministe et le décodeur optimal,
  `E[R]* = (1/27) Σ_m' max_r C[c(r), m']`, et `c` étant une bijection sur les 27
  messages, ce max ne dépend pas de `c`. Écart compositionnel/aléatoire **≤ 1,1 ×
  10⁻¹⁶ à tout ε**. Perdre « un seul attribut » ne rapporte rien quand le crédit
  est tout-ou-rien.
- **Le canal brise pourtant la symétrie** : écart 0,00e+00 sur le groupe
  structurel, ≥ 0,050 sur 200 permutations quelconques. Donc le certificat des
  optima à égalité tient toujours et le théorème d'équivariance ne s'applique plus.
- **Et il ne se passe rien** : z de −0,44 à +0,38 sur six valeurs d'ε, 0 run sur 90
  au-delà du quantile 99,9 %. Borne à 15 graines : |z| < 0,72, contre +9,9 pour la
  paramétrisation structurée.
- **Briser la symétrie est nécessaire, pas suffisant.** Ça réfute l'hypothèse
  unificatrice que j'avais tirée de §6.7 le matin même.
- **Le renouvellement de population ne fait rien non plus, et c'était prédit** :
  remplacer un agent tabulaire par un neuf est échangeable, donc l'équivariance
  survit. z de −0,34 à +0,33 sur quatre périodes.
- **Ce qu'il aurait fallu, calculé exactement** : une récompense à crédit partiel
  par attribut brise l'égalité (+0,108 à ε = 0,2). Mais ça met la
  compositionnalité dans la spécification.
- **Conclusion** : sur ce banc, la compositionnalité n'a jamais été *sélectionnée*.
  Elle a été soit impossible, soit spécifiée.

### Ajouté

`src/test3_communication/courbe_de_contrainte.py` ·
`src/test3_communication/code_emergent.py` ·
`src/test3_communication/dynamique_uniforme.py` ·
`src/test3_communication/qui_ecrit_le_code.py` ·
`src/test3_communication/gradient_premier_pas.py` ·
`src/test3_communication/loi_nulle_longue.py` ·
`src/test3_communication/variabilite_du_maximum.py` ·
`src/test3_communication/appariement_vs_distance.py` ·
`src/test3_communication/certificat_deux_agents.py` ·
`src/test3_communication/representable_atteignable_stable.py`

## [0.4.0] — 2026-07-31

DOI de version : [10.5281/zenodo.21895365](https://doi.org/10.5281/zenodo.21895365)

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
