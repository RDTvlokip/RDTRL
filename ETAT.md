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

1. **« Cycle limite périodique » RÉTRACTÉ le 20/09/2026, même tour où
   il a été trouvé — biais de sélection, pas un vrai résultat.** En
   balayant `[0,200000]` sans se limiter aux trois fenêtres déjà
   repérées (au lieu de zoomer seulement sur 60000/160000/260000),
   des fluctuations d'amplitude comparable apparaissent PARTOUT,
   toutes les 10000-30000 pas environ — pas un oscillateur à période
   fixe ~100000. **Statut correct, plus modeste** : les excursions de
   `R` sous Adam complet sont des fluctuations fréquentes et
   récurrentes de magnitude caractéristique bornée (~0,002-0,004), pas
   un mode périodique isolé ni deux accidents isolés (l'hypothèse
   non-standard n°3 de `REPONSE_ORDRE54.md` reste réfutée dans sa forme
   originale, mais pas au profit de la périodicité). **Reste ouvert** :
   la distribution complète des amplitudes (bornée ou à queue lourde ?),
   et si la fréquence des fluctuations est stationnaire dans le temps.
   Détail complet et rétractation : `CARNET.md` fin de §7.65/8ter.

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
