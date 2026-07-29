# Test 1 — Copier une phrase fixe en RL pur : résultats et verdict

Expérience du 29/07/2026. Cible `le chat dort`, vocabulaire de 12 caractères,
espace de recherche 12¹² ≈ 8,9 × 10¹². Politique GRU 128, REINFORCE + baseline,
poids initialisés aléatoirement, aucune donnée, aucun pré-entraînement.

## Verdict en une phrase

**L'agent a réussi (copie parfaite en 1 639 épisodes), mais pas pour la raison
qui validerait l'hypothèse de départ : le succès vient de la forme de la
récompense, pas d'une capacité du RL à traverser l'espace de recherche.**

## Le mécanisme

La récompense `positions` est **décomposable** :

```
R = (1/12) · Σ_t  1[a_t = c_t]
```

Chaque position contribue indépendamment. Dans le gradient REINFORCE
`∇log π(a_t) · (R − b)`, seul le t-ième indicateur dépend de `a_t` ; les onze
autres termes sont non corrélés avec `a_t` et s'annulent en espérance — ils
n'ajoutent que de la variance, pas de biais. Le problème à 12¹² se factorise
donc en **12 bandits indépendants à 12 bras**. Chacun se résout en quelques
centaines de tirages, d'où ~1 600 épisodes au total.

Autrement dit, la récompense graduée ne « guide » pas une recherche dans
8,9 × 10¹² : elle supprime la recherche en la remplaçant par 12 problèmes
triviaux résolus en parallèle.

## Preuves

### 1. Ce n'est pas de la force brute, ni de la chance, ni une fuite

| contrôle | résultat |
|---|---|
| épisodes vs espace | 1 639 pour 8,9 × 10¹² → **1,8 × 10⁻⁸ %** de l'espace exploré |
| 4 graines (poids réinitialisés) | 1 360 / 1 502 / 1 639 / 1 702 → moyenne 1 551, **σ = 132** (8,5 %) |
| cible aléatoire ` eaiea innhh` | 1 771 épisodes, même ordre de grandeur |
| baseline | `deque` de 100 récompenses passées, avantage `.detach()` — aucun canal vers la cible hors de la fonction de récompense |

La dispersion serrée sur 4 graines exclut la chance : c'est un mécanisme fiable.
La convergence identique sur une cible aléatoire exclut toute fuite d'information
spécifique à une phrase française.

### 2. La variable causale est la récompense, pas l'algorithme

Contrôle décisif : **même architecture, même algorithme, même graine**, seule la
récompense passe en tout-ou-rien (1 si la phrase est exacte, 0 sinon).

```
[tout_ou_rien] ep  10000 | reward moyen (100) = 0.0000
[tout_ou_rien] ep  20000 | reward moyen (100) = 0.0000
[tout_ou_rien] ep  30000 | reward moyen (100) = 0.0000
               meilleure = 'addahr iddno' (r=0.000) | greedy = 'dddddddddddd'
```

30 000 épisodes, récompense **exactement nulle du début à la fin**, pas un seul
succès, la politique s'effondre sur une sortie dégénérée. C'est exactement le
problème du *sparse reward* décrit dans l'objection initiale, et il est
intégralement confirmé. Le test 1 ne le réfute pas : il le contourne, en
supposant un oracle capable de noter une sortie partiellement correcte.

### 3. Ce que l'agent a appris : une table, avec `h` comme désambiguïsateur

La heatmap ([resultats/heatmap_probabilites.png](resultats/heatmap_probabilites.png))
montre une case à ~1,0 par position, tout le reste à zéro : probabilité moyenne
du bon caractère **0,9923**, minimum 0,9729.

L'ablation de l'état caché donne une survie moyenne de **0,67** (mise à zéro) et
**0,71** (bruit) — l'information n'est donc pas concentrée dans `h`, elle est
largement reconstructible depuis le caractère réinjecté. Les effondrements sont
localisés et interprétables :

```
h→zéro après position 1  →  'le dort dort'   (0.70)
h→zéro après position 2  →  'le dort dort'   (0.67)
h→zéro après positions 0, 4, 6, 7, 9, 10  →  'le chat dort'  (1.00)
```

L'espace apparaît deux fois dans la cible (positions 3 et 8), suivi de `c` puis
de `d`. Privé de `h`, le modèle ne sait plus lequel des deux il vient d'écrire et
rabat sur `d`. L'agent a donc appris une **table caractère→caractère quasi
markovienne**, `h` ne servant qu'à distinguer les occurrences répétées d'un même
caractère.

### 4. Aucune structure réutilisable

Perturbation `le chat dort` → `le chien dort` : transfert 1 602 épisodes contre
2 782 depuis zéro, soit **×1,74**. Pris isolément, ce chiffre suggérerait un
acquis réutilisable. Mais les deux cibles partagent 5 positions sur 13 (`le ch`),
donc l'accélération peut n'être que la réutilisation littérale de ce préfixe.

Contrôle ([test4_controle.py](test4_controle.py)) : même transfert vers une cible
de même longueur **sans aucun caractère commun à la même position**.

| transfert vers | positions partagées | accélération |
|---|---|---|
| `le chien dort` | 5 / 13 | ×1,74 |
| `hclt cncir nd` | 0 / 13 | **×0,91** |

Sans recouvrement, le transfert est **plus lent** que de repartir de zéro. Le
×1,74 s'expliquait entièrement par le préfixe littéral. Il n'y a aucune structure
abstraite réutilisable : c'est de la mémorisation position par position.

## Ce que ce test permet de conclure

1. **La question posée est tranchée dans les deux sens.** Un agent en RL pur,
   depuis des poids aléatoires et sans aucune donnée, peut apprendre à produire
   une séquence cible — mais uniquement si la récompense est décomposable
   position par position. Avec une récompense réellement sparse, l'échec est
   total et immédiat.

2. **L'objection du sparse reward est confirmée, pas réfutée.** Elle portait sur
   le tout-ou-rien ; la récompense graduée la contourne au lieu de la résoudre.

3. **Le passage à l'échelle est bloqué par l'oracle, pas par l'algorithme.** La
   récompense `positions` exige de connaître la cible caractère par caractère.
   Pour du texte réel il n'existe pas d'oracle de ce genre — c'est précisément le
   problème que le pré-entraînement résout, en fournissant un signal dense par
   position via le token suivant.

4. **Rien ici ne ressemble à de l'apprentissage du langage.** L'objet appris est
   une table position→caractère de 12 entrées, sans généralisation (test 4).
   Les tâches plus ambitieuses de la feuille de route — règles logiques,
   généralisation, phrases nouvelles — ne sont pas atteignables par extension
   directe de ce dispositif.

## Piste que ce résultat ouvre quand même

Le point intéressant n'est pas que le RL ait échoué, c'est **où** il a échoué :
uniquement sur l'obtention du signal, jamais sur l'optimisation. Dès qu'un signal
gradué existe, l'optimisation est facile et fiable (σ = 8,5 % sur 4 graines).
La question à instruire ensuite n'est donc pas « comment mieux optimiser » mais
« d'où peut venir un signal gradué sans oracle sur la cible » — c'est là que se
situent les approches par curiosité, modèle du monde, ou auto-supervision, et
c'est le vrai sujet du test 2.

## Fichiers

| fichier | contenu |
|---|---|
| [rl_copie.py](rl_copie.py) | expérience complète, phases 1 et 2 |
| [test4_controle.py](test4_controle.py) | contrôle du transfert sans recouvrement |
| [bench_device.py](bench_device.py) | mesure CPU vs GPU justifiant `--device cpu` |
| `resultats/verdict.txt` | verdict brut généré par le script |
| `resultats/rapport.json` | tous les chiffres, sérialisés |
| `resultats/*.csv` | récompense par épisode pour chaque run |
| `resultats/heatmap_probabilites.png` | distribution apprise position × caractère |
| `resultats/courbe_*.png` | courbes d'apprentissage |
