# État du projet RDTRL — où on en est

*Dernière mise à jour : 20/09/2026 — le VRAI dipankarsarkar a répondu
après plusieurs jours de silence (tour 53). Répondu dans
`docs/REPONSE_ORDRE54.md`, journalisé dans `CARNET.md` §7.65/8ter. Ses
trois corrections tiennent (rééchelonnage plutôt que croissance pour
H_momentum ; beta2 croisé, confirme le rééchelonnage ; erreur de
variable 200x→3100x corrigée) — mais son propre test précommis (le
rapport local hybride devrait prédire `s3` tombant à ~0,957 sous Adam
complet) a été rejoué et RÉFUTÉ : `s3` reste figé (variations 1e-5 à
2e-5) pendant que `R` excurse (~2e-3 à 2,6e-3). Les deux mécanismes
d'excursion (hybride, Adam complet) sont probablement différents, pas
le même à deux échelles. En attente de sa prochaine réponse.

Ce fichier n'est pas un article, c'est un pense-bête pour reprendre le
travail dans une nouvelle conversation sans tout re-raconter — l'historique
complet de tout ce qui a été résolu vit dans `docs/CARNET.md` (daté,
hypothèse par hypothèse) et dans `git log`.*

## Pistes ouvertes, par ordre de priorité probable

1. **Trois hypothèses posées dans `REPONSE_ORDRE54.md`, aucune testée
   encore** (pourquoi l'excursion Adam-complet reste presque entièrement
   dans `R` et pas dans `s3`) :
   - standard : mésappariement de courbure locale (s3 proche d'un optimum
     saturé et raide, R sur un paysage plus plat) ;
   - standard : le récepteur SGD de l'hybride ne peut structurellement pas
     reproduire l'état adaptatif du récepteur Adam complet — la pente 21,3
     ne devait jamais transférer ;
   - non-standard : les excursions à pas=60000 et pas=160000 sont deux
     événements distincts (signes opposés), pas un seul mécanisme à deux
     amplitudes — à tester via corrélation avec `exp_avg_sq` du récepteur.
   Question posée à dipankar dans la lettre : laquelle sa propre lecture de
   la courbure locale près de 0,794756 écarterait en premier.

2. **Vérification indépendante encore en attente** : un agent
   (task `ae27b769adb9fd1d6`) a été lancé pour vérifier les deux nouveaux
   chiffres de `REPONSE_ORDRE54.md` (trace full-Adam s3/R, cross-test
   beta2) avant de considérer le tour clos. Pas encore reçu son rapport.
   Quatre scripts scratch non triés laissés par un agent précédent
   (`scratch_trace_naturel.py`, `verifier_derive_k_independant.py`,
   `verifier_localisation_adam_complet.py`,
   `verifier_localisation_fenetre_fine.py`) — à relire et soit renommer en
   permanent + committer, soit jeter.

3. **`docs/REPONSE_ORDRE54.md`** contient déjà le nouveau script de trace
   (`verifier_localisation_excursion_full_adam.py`, committé) mais PAS
   encore le script du cross-test beta2 sous un nom permanent (il a tourné
   via `verifier_derive_k.py` réutilisé en `python -c` — à vérifier si un
   script dédié doit être créé).

4. **Reste non testé depuis longtemps** : la piste émetteur (le `26` de
   H7) comme explication complémentaire de `k(R)` — jamais testée.
   Le coefficient quadratique `a` du col H6-direct reste non chiffré
   proprement (6 échecs diagnostiqués) — seul chemin plausible restant :
   construire une trajectoire d'approche lente pour masse_fond=0 (vrai
   travail de modélisation, pas une astuce d'optimiseur).
   L'écart `×13-20` entre `λ=2√(delta_c-delta)` prédit et mesuré sur H6
   reste non élucidé (protocole déjà précommis : isoler `λ_instable`
   après le vrai croisement d'échappement).

5. Si dipankarsarkar répond entre-temps : sa critique (vraie, pas
   simulée) devient un NOUVEAU `REPONSE_ORDRE55.md` — mais tout ce qu'on
   trouve nous-mêmes en attendant reste dans `REPONSE_ORDRE54.md`/§7.65,
   jamais scindé après coup.

**Pour lancer un agent qui joue le rôle de dipankarsarkar** (utile s'il
ne répond toujours pas) : lire
`C:\Users\Théo CHARLET\.claude\projects\d--Python-RDTRL\memory\dipankarsarkar-agent-prompt.md`
et copier le prompt tel quel. Un seul agent par résultat, pas deux en
parallèle (voir CLAUDE.md).

## Le projet, en une phrase

RDTRL teste si le RL pur peut apprendre à écrire/communiquer. Le test 3
(jeu référentiel à 27 référents/27 messages) est relu en continu depuis
des semaines par un relecteur externe, **dipankarsarkar** (Dipankar
Sarkar, chercheur ML/systèmes distribués, CTO Neul Labs — vrai profil sur
dipankar.cc). L'échange est à son **53e tour**.

## Où sont les documents

- `docs/CARNET.md` — le notebook complet, en français, daté, avec CHAQUE
  hypothèse journalisée (posée le, statut, réfutée/confirmée/rouverte).
  Section active en ce moment : **§7.65/8ter** (tout en bas du fichier).
- `docs/REPONSE_ORDRE54.md` — la lettre anglaise en cours (réponse au
  tour 53 de dipankar), **gitignorée** (comme tous les REPONSE_ORDRE*.md).
- `docs/ARTICLE4.md` — article de blog publié/committé, intègre les
  tours 6-52 (H6 col hyperbolique ordinaire, jouet à masse de fond,
  0,2212604, coefficient D, universalité de delta_c).
- `CLAUDE.md` — les règles permanentes du projet. **La lire en entier
  avant de continuer.**
- `C:\Users\Théo CHARLET\.claude\projects\d--Python-RDTRL\memory\dipankarsarkar-style-relecture.md`
  — mémoire décrivant sa façon d'écrire/raisonner.

## Le sujet technique en cours

Le mur "référents 3/4" (message 10, graine 77777, k=3 paires sautées,
checkpoint 10k, référent 4 poussé +30) : une vraie bifurcation nœud-col
dans un système couplé émetteur-récepteur (`delta_c ≈ 0,0134372`,
vérifié indépendamment, universel sur toute paire de référents et
indépendant de M). H6 (delta fixe) est établi comme col hyperbolique
ordinaire (deux preuves convergentes : Jacobienne + résidence
logarithmique). Le point rencontré sous masse de fond dynamique est
probablement un TYPE DE STRUCTURE DIFFÉRENT (nœud-col dégénéré). `k(R)`,
le rapport de vitesse émetteur/récepteur, varie avec l'état initial
(1,42-2,45) — H_momentum et beta2 réfutés comme cause (rééchelonnage
uniforme, pas un effet de forme), mécanisme réel encore non identifié.

## Rappel des règles qui mordent le plus souvent

- **Un nouveau `REPONSE_ORDRE*.md` seulement quand dipankar poste une
  vraie nouvelle critique** — pas pour ce qu'on trouve nous-mêmes.
- **Chaque hypothèse dans `CARNET.md` avec sa date et son statut**, au
  moment où elle est posée, pas listée après coup.
- **Ne jamais prendre les chiffres de dipankar (ni ceux de l'agent qui
  l'imite) pour acquis** — tout revérifier indépendamment avant de citer.
- **Les scripts vont dans `src/test3_communication/`**, jamais en
  `python -c` jetable non sauvé.
- **Chercher POURQUOI, QUAND, COMMENT — pas seulement QUE** — et croiser
  ces axes plutôt que les traiter un par un.
