# Test 3 — Efficacité communicative : conception, instruments, seuil

État : **conçu, pas implémenté.** Ce document fixe le dispositif et le critère de
falsification **avant** d'écrire du code, ce que les tests 1 et 2 n'ont jamais eu.

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

## 4. Critère de falsification, fixé à l'avance

Retirer de l'entraînement des **combinaisons** d'attributs, jamais des valeurs
entières — c'est l'erreur du token jamais vu du test 2, où l'embedding non
entraîné rendait le test vide par construction.

Mesures :

| mesure | seuil proposé |
|---|---|
| précision zéro-shot sur les combinaisons retenues | **≥ 60 %** sur au moins **2 graines sur 3** |
| similarité topographique (distance sémantique ↔ distance des messages) | **≥ 0,3** |
| référence hasard | 3,7 % (reconstruction), 6,25 % si variante discrimination à K=16 |

En dessous, l'hypothèse « le RL découvre une représentation linguistique
générale » est abandonnée **pour ce design**, sans renégociation après coup.

**Prédiction au dossier, datée du 29/07/2026** : succès de tâche élevé, code non
compositionnel, zéro-shot proche du hasard.

---

## 5. Instruments à construire avant l'entraînement

1. **Énumération complète** des 27 référents, 27 messages, et de la loi jointe
   émetteur × récepteur.
2. **Comptage exact** des bijections parfaites et compositionnelles (fait, §3).
3. **Optimum de l'objectif** en forme close, comme `optimum_gibbs.py` au test 2.
4. **Sonde de capacité** : construire un code compositionnel à la main, vérifier
   que les deux agents peuvent le représenter, puis s'ils peuvent l'atteindre,
   puis s'il est stable quand on les y démarre. C'est la structure Q-A / Q-D du
   test 2 transposée.
5. **Contrôle tabulaire** : mêmes agents en paramétrisation libre, pour séparer
   l'objectif de la paramétrisation.
6. **Décomposition d'information** : `I(référent ; message)` ventilée par
   attribut. Un code compositionnel porte de l'information attribut par attribut ;
   un code holistique ne porte que l'identité.
7. **Analyse d'ordre 1** : que voit le gradient à l'initialisation, dans un jeu où
   la récompense dépend de la politique du partenaire ?

---

## 6. Ce qui est nouveau par rapport au test 2

La récompense est **non stationnaire** : elle dépend de la politique du partenaire,
qui apprend en même temps. Toutes les analyses du test 2 supposaient une fonction
fixe.

Questions ouvertes que ça crée, reprises en §8 du carnet :

- Quelle est l'ANOVA d'une récompense qui co-évolue ?
- Qui décide de la structure du code, l'émetteur ou le récepteur ? Geler l'un,
  entraîner l'autre. Est-ce la même asymétrie que ordre-de-génération contre
  direction-de-l'accord au test 2 ?
- La mesure induite par la **dynamique** sur l'ensemble des codes parfaits est-elle
  vraiment l'uniforme supposée par le calcul de §3 ? Elle pourrait concentrer sur
  une sous-famille pour des raisons sans rapport avec la récompense, exactement
  comme le biais d'ordre 1 a décidé la branche au test 2.
