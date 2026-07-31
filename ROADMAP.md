# Feuille de route — où on en est, et quoi faire ensuite

Fichier de reprise. À lire en premier après une pause ou une compression de
conversation. Le détail conceptuel est dans [docs/TEST3.md](docs/TEST3.md), le
raisonnement dans [docs/CARNET.md](docs/CARNET.md).

---

## État au 29/07/2026

**Tests 1 et 2 : terminés, publiés.**

- Article : https://huggingface.co/blog/RDTvlokip/teaching-a-network-to-write-with-reward-only
- Dépôt : https://github.com/RDTvlokip/RDTRL
- Résultats : [docs/ANALYSE.md](docs/ANALYSE.md) et [docs/ANALYSE_TEST2.md](docs/ANALYSE_TEST2.md)

**Test 3 : étape 1 sur 7 faite.**

---

## Test 3 — ordre d'exécution

⚠️ **Cet ordre n'est pas celui du §6 de TEST3.md.** Le §6 range les questions par
dépendance conceptuelle ; ici c'est l'ordre d'exécution. Ce qui peut invalider le
reste passe en tête, ce qui coûte cher passe en dernier.

### ✅ Étape 1 — le monde et la loi nulle · `src/test3_communication/grammaire3.py`

Fait. 27 référents, 27 messages, comptage vérifié par énumération, matrice
d'information mutuelle attribut × position, statistique de concentration, et la
loi nulle sur 20 000 bijections uniformes.

Acquis :

| | |
|---|---|
| bijections parfaites | 27! = 1,089 × 10²⁸ |
| codes compositionnels | 1 296, soit **1,19 × 10⁻²⁵** |
| concentration d'un code compositionnel | **1,0000** (vérifié sur les 1 296) |
| loi nulle | **0,1273 ± 0,0332**, max observé 0,3305 |
| écart | **26,3 écarts-types** |
| seuil pratique dérivé | ~0,35 |

### ⏳ Étape 2 — LE CERTIFICAT TIENT-IL À DEUX AGENTS ? (§6.7)

**À faire en premier, et c'est la seule qui peut tout casser.**

Le certificat des optima à égalité a été établi au test 2 pour un **agent unique**
optimisant `E[R] + β·H`. Ici l'objectif conjoint est
`(1/27)·tr(S·R) + β·(H(S) + H(R))` : l'entropie porte sur **deux politiques
séparément**, pas sur la distribution jointe des messages. Rien ne garantit que
l'optimum reste une loi de Gibbs, ni que les 27! optima restent équiprobables.

**Si ça casse, le calcul des 10⁻²⁵ est faux et tout TEST3.md est à réécrire.**

À construire : résolution numérique de
`max (1/27)·tr(S·R) + β·(H(S)+H(R))` sur les matrices stochastiques 27 × 27, en
**paramétrisation tabulaire et gradient exact** (aucun réseau, aucun
échantillonnage), puis regarder à quoi ressemble l'optimum et si les solutions
sont équiprobables.

### Étape 3 — représentable / atteignable / stable (§6.5)

Construire à la main le code compositionnel canonique (`S = P`, `R = Pᵀ`), puis
les trois questions **séparément**, parce qu'au test 2 elles avaient trois
réponses différentes et que c'est ce qui a fait basculer le verdict :

1. les réseaux peuvent-ils le **représenter** (ajustement supervisé, diagnostic) ;
2. l'entraînement depuis l'aléatoire peut-il l'**atteindre** ;
3. en démarrant dessus, REINFORCE y **reste**-t-il, ou dérive-t-il vers un
   attracteur comme les 45,3 modes du test 2.

### Étape 4 — quel code émerge, et est-il tiré au hasard ? (§6.1 puis §6.2)

Entraînement multi-graines (50 à 100, le modèle est minuscule). Pour chaque run,
extraire la bijection, calculer sa concentration, comparer à la loi nulle de
l'étape 1.

- indiscernable de la nulle → prédiction confirmée, la récompense décide seule ;
- **significativement au-dessus** → quelque chose sélectionne hors récompense, et
  le raisonnement qui porte tout le projet a une faille. Ce serait le résultat
  majeur de RDTRL.

### Étape 5 — qui écrit le code, émetteur ou récepteur ? (§6.3 puis §6.4)

Geler l'un, entraîner l'autre, dans les deux sens et avec un code compositionnel
puis un code arbitraire. Plus l'analyse du gradient initial, qui est gratuite :
`∂E[R]/∂S[r,m] = R[m,r]/27`.

### Étape 6 — la courbe de contrainte (§6.6)

**En dernier**, parce que c'est la plus coûteuse et qu'elle mesure un déplacement,
donc qu'elle n'a aucun sens avant que la ligne de base des étapes 1 et 4 existe.

Boutons : bruit de canal ε, goulot de vocabulaire, renouvellement de population,
pression de longueur. On trace concentration en fonction du bouton.

---

## Concurrent apparu le 31/07/2026 — le test de renversement du plafond

Le plafond de produit (carnet §7.11) est le résultat le plus solide du projet, et
**une seule expérience décide s'il est publiable** : construire un lexique où
l'ordre des plafonds s'inverse, prédire l'inversion avant de lancer, vérifier.
Quelques heures, tout l'outillage existe (`produit_et_saturation.py`,
`balayage_70_graines.py`).

À arbitrer contre le test 3 : ce test-là finit un résultat acquis, le test 3 en
ouvre un nouveau. Évaluation complète et question d'attribution en §7.12.

---

## Prédiction enregistrée le 29/07/2026

> Les codes émergents seront des bijections quasi parfaites (succès de tâche
> élevé) et **non compositionnels**. Leur concentration sera **statistiquement
> indiscernable** de la loi nulle.

Pas de seuil pass/fail : Théo veut le mécanisme, pas le verdict. Le verdict est de
toute façon déjà démontré par le comptage.

---

## Rappels qui coûtent cher à oublier

- **Modifier avec Edit/Write, jamais par script de remplacement.** Un `str.replace`
  a déjà cassé un import en silence dans ce projet.
- **Langue** : dossiers, README, docs et figures en anglais ; noms de fichiers et
  code (commentaires, identifiants) en français. C'est un choix assumé, écrit dans
  le README. Ne pas le « corriger ».
- **Lancer les scripts depuis leur propre dossier** — ils s'importent mutuellement.
- **Attendre la fin d'un balayage avant d'écrire sa conclusion.** J'ai publié « ce
  n'est pas le bruit, c'est la géométrie » à partir d'un seul β pendant que le
  sweep tournait ; les runs suivants l'ont démentie.
- **Ne jamais rapporter une métrique de diversité sans la masse valide en regard.**
  Le réseau non entraîné a 47,5 modes effectifs sur 48.
- **Le baseline trop bête pour être lancé doit être lancé.** Rejet depuis un réseau
  aléatoire : 100 % de validité, ~47,5 modes, contre 18,6 pour REINFORCE.
