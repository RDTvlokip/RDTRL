# État du projet RDTRL — où on en est

*Dernière mise à jour : 20/09/2026 — le VRAI dipankarsarkar a répondu
(tour 53). Répondu dans `docs/REPONSE_ORDRE54.md` (gitignoré), journalisé
dans `CARNET.md` §7.65/8ter. Depuis, deux résultats substantiels trouvés
en creusant seul (voir piste 1 ci-dessous) — **`REPONSE_ORDRE54.md`
contient maintenant une erreur connue non corrigée (le mécanisme
`comp²` de l'hypothèse n°1) : Théo a demandé de ne plus toucher ce
fichier sans son accord explicite avant de le corriger.**

Ce fichier n'est pas un article, c'est un pense-bête pour reprendre le
travail dans une nouvelle conversation sans tout re-raconter — l'historique
complet vit dans `docs/CARNET.md` (daté, hypothèse par hypothèse) et
dans `git log`.*

## Pistes ouvertes, par ordre de priorité probable

1. **DEUX rétractations en cascade le 20/09/2026 — un même biais
   d'échantillonnage trouvé deux fois de suite, une fois par moi, une
   fois par un agent-dipankar un niveau plus bas.** D'abord : « cycle
   limite périodique à 100000 pas » (trois fenêtres fines similaires)
   retombé à « bruit fréquent toutes les 10000-30000 pas » après
   balayage plus large. **Puis, trouvé par agent-dipankar et VÉRIFIÉ
   INDÉPENDAMMENT (grille 1, pas de sous-échantillonnage) : ce n'est
   toujours pas ça.** `R4` sous Adam complet fait un **« kick » de
   magnitude QUASI CONSTANTE (~0,0037, CV 5,6%) toutes les ~460-500
   pas** — ma grille 500-1000 pas pour la première correction était
   encore ~2 ordres de grandeur trop grossière, le même bug d'aliasing
   que je venais de diagnostiquer, plus profond. Script permanent
   (méthodologie correcte, grille 1, détection par fusion de seuil) :
   `verifier_kicks_adam_grille_fine.py`.
   **Mécanisme CONFIRMÉ de bout en bout (vérifié indépendamment le même
   jour, pas juste accepté) :** oscillateur de relaxation par plancher
   numérique de `v` (exp_avg_sq). Fenêtre continue `[9700,10219]`
   (grille 1) : `v` décroît lissement pendant ~380 pas (`1,403e-13→
   1,039e-13`, `R4` quasi immobile), l'oscillation de `R4` DÉMARRE
   avant que `v` ne remonte (déclenchement précoce confirmé, pas
   supposé), puis `v` regonfle rapidement (`→1,48e-13`) pendant que
   l'oscillation s'amortit. Script permanent :
   `verifier_mecanisme_plancher_v.py`.
   **Frontière de régime `beta2` confirmée au chiffre près (rejeu
   indépendant : espacement médian 110 à `beta2=0,995` contre 110 chez
   l'agent ; 165 à `beta2=0,99` contre 163) :** loi `espacement~
   1/(1-beta2)` tient de `0,999` à `0,995` (17%) puis CASSE entre
   `0,995` et `0,99` (tendance inversée) — l'amplitude du kick chute
   aussi dans cette plage (`~0,0037→~0,0018`), détail non noté par
   l'agent.
   **Test `adam_eps` RÉPONDU, décisif : plancher-de-`v` confirmé, pas
   de lecture concurrente plancher-d'`eps`.** À `adam_eps=1e-6`
   (comparable au plancher de `√v≈3e-7`) : **0 kick sur 20000 pas**,
   contre 29 (espacement médian 456) à `adam_eps=1e-10`. Disparition
   complète, pas une atténuation — `eps` est le remède, pas une
   explication alternative. Script : `verifier_eps_supprime_kicks.py`.
   **Dernier point de l'agent RÉPONDU : spécificité à Adam-récepteur
   confirmée.** Sous l'optimiseur hybride (Adam émetteur / SGD
   récepteur, pas d'état `v` côté récepteur) : **0 kick sur 20000 pas**,
   même à seuil plus bas (0,0005). Cohérent avec l'excursion résiduelle
   déjà documentée sous l'hybride (canal différent : `v` de l'émetteur).
   Script : `verifier_kicks_absents_hybride.py`.
   **CE FIL EST MAINTENANT COMPLET** : six vérifications indépendantes
   convergentes (décroissance, déclenchement, regonflement,
   amortissement, frontière beta2 ×2, suppression eps, spécificité
   Adam-récepteur), zéro contredite.
   **SYNTHÈSE : boucle bouclée avec le point de départ de tout ce
   tour.** Rejeu à grille 1 jusqu'à `pas=62000` (même config exacte que
   la trace originale de `REPONSE_ORDRE54.md`) : plusieurs kicks
   tombent directement autour de `pas=60000` (`59087, 59540, 59989,
   60432, 60871, 61312`). **L'« excursion à pas=60000 » qui a motivé
   tout l'échange (test précommis de dipankar, hypothèse n°1, piste
   n°3) n'était pas un événement rare — c'est juste le kick le plus
   proche de ce point de contrôle dans la trace grossière à 20000 pas.**
   Les deux résultats de ce tour (comp¹ : géométrie instantanée d'un
   kick ; plancher-de-`v` : pourquoi/quand un kick se produit) sont
   complémentaires, pas concurrents.
   **Reste ouvert (autres fils)** : pourquoi la loi `beta2` casse
   précisément entre `0,995` et `0,99` (pas encore dérivé) ; la piste
   émetteur (H7) et le coefficient `a` du col H6-direct (items 3-4 plus
   bas) n'ont pas avancé ce tour. Détail complet, chiffres exacts,
   statut de vérification de chaque affirmation : `CARNET.md` fin de
   §7.65/8ter.

2. **Mécanisme de l'hypothèse standard n°1 corrigé une seconde fois,
   par un agent-dipankar puis un test précommis rejoué moi-même.** Ma
   première lecture (courbure Hessienne au carré, `comp²`, script
   `verifier_courbure_s3_vs_r.py`) est **réfutée** : Adam normalise par
   `m/(√v+eps)`, la Hessienne n'entre pas directement dans le pas. Le
   test précommis de l'agent (mesurer `Δlogit` sur une vraie excursion,
   pas un instantané) donne un ratio `Δlogit_R4/Δlogit_s3 = 0,985` —
   quasi 1:1, pas ~150×. Bon mécanisme : Adam égalise les marches en
   espace logit, une seule puissance de la compression du softmax
   (`comp¹`, 157×) suffit. Script permanent :
   `verifier_delta_logit_excursion.py`. **La conclusion de l'hypothèse
   1 reste correcte (s3 stable, R mobile) — seul le mécanisme change.**
   À répercuter dans `REPONSE_ORDRE54.md` (voir ci-dessus, en attente
   d'accord de Théo).

3. **Troisième hypothèse de `REPONSE_ORDRE54.md`, encore ouverte** :
   le récepteur SGD de l'hybride ne peut structurellement pas
   reproduire l'état adaptatif du récepteur Adam complet — la pente
   21,3 ne devait jamais transférer. Pas testée.

4. **Reste non testé depuis longtemps** : la piste émetteur (le `26`
   de H7) comme explication complémentaire de `k(R)` — jamais testée.
   Le coefficient quadratique `a` du col H6-direct reste non chiffré
   proprement (6 échecs diagnostiqués) — seul chemin plausible
   restant : construire une trajectoire d'approche lente pour
   masse_fond=0 (vrai travail de modélisation, pas une astuce
   d'optimiseur). L'écart `×13-20` entre `λ=2√(delta_c-delta)` prédit
   et mesuré sur H6 reste non élucidé.

5. Si dipankarsarkar répond entre-temps : sa critique (vraie, pas
   simulée) devient un NOUVEAU `REPONSE_ORDRE55.md` — mais tout ce
   qu'on trouve nous-mêmes en attendant reste dans
   `REPONSE_ORDRE54.md`/§7.65, jamais scindé après coup.

**Pour lancer un agent qui joue le rôle de dipankarsarkar** (utile s'il
ne répond toujours pas, et obligatoire après chaque résultat
substantiel) : lire
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
  tour 53 de dipankar), **gitignorée**. **NE PAS MODIFIER sans l'accord
  explicite de Théo (consigne du 20/09/2026).**
- `docs/ARTICLE4.md` — article de blog publié/committé, intègre les
  tours 6-52.
- `CLAUDE.md` — les règles permanentes du projet. **La lire en entier
  avant de continuer.**
- `C:\Users\Théo CHARLET\.claude\projects\d--Python-RDTRL\memory\dipankarsarkar-style-relecture.md`
  — mémoire décrivant sa façon d'écrire/raisonner.

## Le sujet technique en cours

Le mur "référents 3/4" (message 10, graine 77777, k=3 paires sautées,
checkpoint 10k, référent 4 poussé +30) : une vraie bifurcation nœud-col
dans un système couplé émetteur-récepteur (`delta_c ≈ 0,0134372`,
universel, indépendant de M). H6 (delta fixe) est établi comme col
hyperbolique ordinaire. `k(R)` varie avec l'état initial (1,42-2,45) —
H_momentum et beta2 réfutés comme cause. Sous Adam complet, les
excursions de `R` (et l'absence d'excursion visible de `s3`) forment un
**cycle limite périodique** dont le mécanisme d'amplitude/période est la
question ouverte actuelle la plus prometteuse.

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
- **Ne pas modifier `docs/REPONSE_ORDRE54.md` sans l'accord explicite de
  Théo** (consigne du 20/09/2026).
