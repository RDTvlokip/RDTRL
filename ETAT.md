# État du projet RDTRL — où on en est

*Dernière mise à jour : 17/09/2026 (pistes 1 et 2 résolues dans cette
session, voir ci-dessous). Ce fichier n'est pas un article, c'est un
pense-bête pour reprendre le travail dans une nouvelle conversation sans
tout re-raconter.*

## Pistes concrètes pour la prochaine conversation, par ordre de priorité probable

1. **RÉSOLU le 17/09/2026, EN DEUX PASSES (auto puis agent-dipankar).**
   Refit du coefficient 0,2212 sur 11 points algébriques (au lieu de 3) :
   `0,2212` = écart entre racine stable et racine instable du système
   couplé (`d3_instable - d3_stable`), pas un résidu contre une loi
   molle. **Un agent-dipankar (nouvelle règle CLAUDE.md : challenger
   chaque résultat avant de le clore) a ensuite trouvé deux vraies
   erreurs dans ce premier passage, vérifiées indépendamment en mpmath
   50 chiffres avant d'être acceptées :** (a) mon « plateau » à
   0,221305-0,221372 était un plancher de précision float64, pas la
   vraie asymptote — retraçage à `eps=1,3437e-8` en haute précision + un
   calcul analytique local au pli (Lyapunov-Schmidt) tombent tous les
   deux, indépendamment, sur **0,2212604** (7 chiffres significatifs de
   concordance) ; (b) mon modèle de dérive à 2 termes
   (`gap=C1·√eps+C2·eps`) utilisait une base analytiquement fausse — le
   terme en `eps¹` est structurellement NUL, le vrai terme suivant est
   en `eps^(3/2)` (coefficient empirique `D≈8,0021`, pas encore fermé
   analytiquement). Une 3e correction, plus mineure : mon affirmation
   que les 3 points publiés par dipankar « ne pouvaient pas venir d'un
   entraînement Adam » était surinterprétée — une correspondance
   numérique seule ne distingue pas « jamais entraîné » de « entraîné et
   convergé exactement » ; rétrogradée à indéterminée. Scripts :
   `verifier_refit_gap_hybride.py` (essai raté, documenté),
   `verifier_gap_racines.py` (bonne cible), `verifier_puiseux_gap.py`
   (corrections de l'agent, vérifiées indépendamment). Détail complet :
   `CARNET.md`, fin de §7.65.
2. **RÉSOLU le 17/09/2026.** `delta_c` pour la paire 23/25 (bissection
   empirique par entraînement réel, pas seulement l'algèbre qui le
   prédisait déjà trivialement) : `0,0134370`, contre `0,0134372` pour
   3/4 — écart 0,0015 %, au niveau du bruit de la bissection elle-même.
   **Le SEUIL lui-même est une propriété de l'objectif (N=27, beta=0,02),
   pas de la paire de référents ni de la graine** — troisième
   confirmation indépendante du mécanisme sur 23/25. Script :
   `verifier_delta_c_23_25.py`.
3. **CARACTÉRISÉE le 17/09/2026 — mécanisme confirmé réel, profil à
   seuil+plafond établi, reste à chiffrer contre `k(R)`.** (Une fausse
   conclusion intermédiaire — « effet nul, mécanisme récepteur écarté »
   — a été écrite PUIS corrigée dans la même session ; voir `CARNET.md`
   pour l'historique complet si besoin, mais l'état final ci-dessous est
   le bon.) Jouet à M catégories de fond variables
   (`verifier_jouet_n_variable.py`), bug de convergence corrigé
   (`lr=0,2` au lieu de 0,05). **Grille ciblée finale (masse totale
   fixée, M variable) :**
   ```
   M=25, masse totale=8 %   -> delta_c=0,018672  (= M=8 a 8%, = M=0 : NUL)
   M=15, masse totale=25 %  -> delta_c=0,018516  (= M=25 a 25% : REEL, -0,84%)
   M=25, masse totale=50 %  -> delta_c=0,018516  (IDENTIQUE a 25% — pas plus)
   ```
   **Deux faits établis : (1) c'est la MASSE TOTALE de fond qui compte,
   pas M (confirmé — même masse, M différent, même résultat) ; (2)
   l'effet a un SEUIL (entre 8% et 25%) puis un PLAFOND (25% et 50%
   donnent le même -0,84%, pas de croissance continue).**

   **(a) FAIT — seuil localisé précisément entre 17,7422 % et
   17,7539 % de masse totale** (18 points au total, bissection, jamais
   de valeur intermédiaire entre les deux `delta_c` — un vrai saut
   binaire, pas un dégradé). Hypothèse ouverte : ce saut net suggère une
   bascule qualitative entre deux branches du système, pas un décalage
   continu du même point fixe — pas encore démontré directement.

   **(b)/(c) PAS RÉSOLU — un chiffre précis (`×2,2`) a été mesuré PUIS
   s'est effondré en creusant plus loin, dans la même session. Lire
   `CARNET.md` en entier avant de citer un chiffre de ce fil.**
   Chronologie : (1) mesuré un ralentissement récepteur brut `×8` —
   FAUX, contaminé par un bug d'indexage (`t=0` enregistré après un
   pas d'Adam complet) et par la non-monotonie de `r3` (dépassement de
   62%) ; (2) corrigé et décomposé en `~×3,3` (délai d'évacuation de la
   masse de fond, demi-vie ~2 pas) `×~×2,2` (taux post-évacuation),
   avec ce `2,2` tombant dans `k∈[1,42;2,45]` du tour 52 — vérifié
   deux fois indépendamment (agent + moi) ; (3) tranché le point ouvert
   sur `r_tie` (R_init) seul — trouvé un "pic" non monotone à
   `r_tie=0,75`, ratio `×2,6`, PUIS un agent a montré que `min(r3)` et
   son pas d'apparition sont MONOTONES (le pic n'est visible que sur
   la pente dérivée) — vérifié : en fenêtre tardive, le pic disparaît
   complètement (taux quasi constant, ~6% de variation) — RÉFUTÉ ; (4)
   par prudence, revérifié le `×2,2` de l'étape (2) en fenêtre tardive
   aussi — s'effondre aussi (pente ne se stabilise pas) ; (5)
   **18/09/2026, repris avec le point fixe algébrique du jouet résolu
   (4 équations couplées, itération pure, aucun ralentissement de
   gradient) comme référence FIXE** — révèle un écart de 6 ordres de
   grandeur entre M=0 et M=25 à budget égal (2e-11 contre 7e-5 à 3000
   pas), confirmant que le mécanisme est réel et massif ; mais même à
   20000 pas et avec fenêtre adaptative, `M=25` ne donne jamais une
   pente stable — **trouvé pourquoi : l'écart au point fixe REBONDIT
   (`4,6e-4` à pas=15000 après être descendu à `2e-5`) — c'est le MÊME
   phénomène d'excursions déjà caractérisé ce tour (H15, second moment
   d'Adam), retrouvé indépendamment sur le jouet.** **Conclusion
   méthodologique définitive : aucune mesure de pente sur trajectoire,
   quelle que soit la fenêtre/référence/budget, ne peut donner un `k`
   fiable près d'un pli sous Adam — c'est exactement pourquoi le tour
   52 avait dû utiliser le protocole "épingler-falsifier" (ODE à deux
   échelles de temps, bissection du point de bascule) plutôt qu'une
   lecture directe de taux.** Le mécanisme récepteur est solidement
   établi (via `delta_c`, un seuil discret non affecté par les
   excursions) ; le chiffrer précisément contre `k∈[1,42;2,45]`
   demande de construire l'ODE + protocole pin-and-falsify POUR CE
   JOUET (les fonctions de branche et le point fixe algébrique sont
   déjà dérivés — reste la mise en ODE et la bissection des points de
   bascule) — vrai travail de modélisation pour la prochaine session,
   pas un raccourci qui resterait à trouver. La piste émetteur (le
   `26` de H7) reste une hypothèse complémentaire, jamais testée.
   Scripts : `verifier_evacuation_fond.py`, `verifier_point_fixe_jouet_m.py`
   (point fixe algébrique + fenêtre adaptative — la bonne référence,
   mais la limite d'excursions demeure). Détail complet et hypothèses
   datées : `CARNET.md`, fin de §7.65.
2bis. **Piste 2 : soumise à un agent-dipankar après coup (oubli initial,
   repéré par Théo, corrigé).** Deux corrections mineures confirmées
   (`0,0016%` pas `0,0015%` ; `delta_c(3/4)` tombe à 71% du bracket, pas
   au centre). Une crainte plus sérieuse (valeurs `R` près de `delta_c`
   non convergées à `pas=40000`) testée directement et RÉFUTÉE (`R`
   identique à 6 décimales de 40k à 400k pas). Le résultat original
   (delta_c universel entre 3/4 et 23/25) tient, juste mieux
   caractérisé.
4bis. **Nouveau (17/09/2026, en marge de piste 3) : le coefficient `D`
   du terme `eps^(3/2)` de la forme normale du pli, dérivé à la main
   (Lyapunov-Schmidt à l'ordre suivant), confirmé par un agent-dipankar
   ET vérifié indépendamment.** `D_analytique=8,0021212` matche
   `D_empirique=8,0021` (extrapolation numérique) à 0,000265% près.
   Deux corrections trouvées par l'agent et confirmées : (a) une
   équation intermédiaire mal écrite dans la doc (le CODE, lui, était
   toujours correct) ; (b) `D≈8` n'est PAS une constante structurelle —
   dérive continûment avec `beta` (4,86 à 8,57) et `N` (8,00 à 8,59),
   confirmé par balayage indépendant à 4-5 chiffres significatifs. Une
   question de l'agent (singularité à `beta=0,015`) a été réfutée
   indépendamment (juste un mauvais point de départ de son côté).
   Scripts : `verifier_puiseux_ordre_suivant.py`,
   `verifier_d_structurel.py`, `verifier_c0_vs_beta.py`.
4. **`docs/ARTICLE4.md` n'intègre toujours pas les tours 48-52** — gros
   morceau d'écriture, à faire si ce fil se stabilise assez.
5. Si dipankarsarkar répond entre-temps, reprendre le flux normal (sa
   critique → `REPONSE_ORDRE54.md`, tout ce qu'on a trouvé nous-mêmes
   reste dans `REPONSE_ORDRE53.md`/§7.65 jusqu'à ce moment-là).

**Pour lancer un agent qui joue le rôle de dipankarsarkar** (utile s'il
ne répond toujours pas) : lire
`C:\Users\Théo CHARLET\.claude\projects\d--Python-RDTRL\memory\dipankarsarkar-agent-prompt.md`
et copier le prompt tel quel.

## Le projet, en une phrase

RDTRL teste si le RL pur peut apprendre à écrire/communiquer. Le test 3
(jeu référentiel à 27 référents/27 messages) est relu en continu depuis
des semaines par un relecteur externe, **dipankarsarkar** (Dipankar
Sarkar, chercheur ML/systèmes distribués, CTO Neul Labs — vrai profil sur
dipankar.cc). L'échange est à son **52e tour**. Il n'a pas répondu depuis
2 jours (au 17/09/2026) — on continue à creuser nous-mêmes en attendant,
y compris en simulant sa critique via un agent qui imite son style.

## Où sont les documents

- `docs/CARNET.md` — le notebook complet, en français, daté, avec CHAQUE
  hypothèse journalisée (posée le, statut, réfutée/confirmée/rouverte).
  Section active en ce moment : **§7.65** (tour 52, tout en bas du
  fichier).
- `docs/REPONSE_ORDRE53.md` — la lettre anglaise en cours (réponse au tour
  52 de dipankar), **gitignorée** (comme tous les REPONSE_ORDRE*.md — ne
  pas s'étonner qu'elle ne soit pas dans git log).
- `docs/ARTICLE4.md` — article de blog déjà publié/committé couvrant les
  tours 6-47 (vérifié exhaustivement, 3 erreurs trouvées et corrigées).
  Les tours 48-52 n'y sont PAS encore intégrés.
- `CLAUDE.md` — les règles permanentes du projet, très développées
  maintenant (cycle répondre/expérimenter/hypothèses, journal daté dans
  CARNET, ne jamais prendre les chiffres de dipankar pour acquis, grille
  QUAND/COMMENT/POURQUOI/OÙ/COMBIEN/JUSQU'OÙ/DEPUIS QUAND/SUR COMBIEN +
  ton de méfiance). **La lire en entier avant de continuer.**
- `C:\Users\Théo CHARLET\.claude\projects\d--Python-RDTRL\memory\dipankarsarkar-style-relecture.md`
  — mémoire décrivant précisément sa façon d'écrire/raisonner, à
  réutiliser tant qu'il ne répond pas.

## Le sujet technique en cours (tours 48-52)

Le mur "référents 3/4" (message 10, graine 77777, k=3 paires sautées,
checkpoint 10k, référent 4 poussé +30) — d'abord lu comme un point fixe
dynamique à 0,5, puis comme un posterior bayésien de collision, puis
comme une vraie bifurcation nœud-col dans un système couplé
émetteur-récepteur à deux équations (`delta_c ≈ 0,0134372`, vérifié
indépendamment). Le tour 52 a montré que même cette forme fermée ne
suffit pas : il manque `k`, le rapport de vitesse relatif
émetteur/récepteur, et **k n'est pas constant** (1,42 à 2,45 selon l'état
de départ).

## Ce qui vient d'être trouvé et qui n'est PAS encore complètement résolu

**RÉSOLU le 17/09/2026, ce fil est maintenant clos.** Un agent jouant le
rôle de dipankarsarkar avait trouvé que le test SGD qui avait « réfuté »
H15 avait en réalité le **référent 3 gelé** (gradient brut trop petit
près de la saturation pour bouger sous SGD à taux unique) — vérifié
indépendamment, confirmé, H15 rouverte.

Deux tentatives de correction par SGD à deux taux ont échoué (calibration
sur le mauvais gradient, puis gradient réel trop minuscule pour un `lr`
fixe sans danger — voir `CARNET.md` §7.65 pour le détail). **La
correction qui a marché : un optimiseur HYBRIDE — Adam sur l'émetteur
seul, SGD pur sur le récepteur seul** (`verifier_optimiseur_hybride.py`).
Le référent 3 s'entraîne vraiment cette fois (conforme à la prédiction
fermée de H7), `R` converge vers la vraie valeur de branche (0,794756,
pas l'artefact du référent gelé), et **les excursions sont toujours là**
— plus petites que sous Adam complet mais réelles, aux mêmes pas
approximativement.

**Conclusion : H15 (artefact du second moment d'Adam) est CONFIRMÉE, et
plus précisément qu'avant — les excursions viennent spécifiquement de
l'émetteur, pas du récepteur** (elles survivent à un récepteur non
adaptatif). L'hypothèse concurrente (mode propre du système couplé
complet) est réfutée : un vrai mode dynamique devrait survivre
indépendamment de quel joueur est adaptatif, et ce n'est pas le cas.

Un deuxième point a aussi été trouvé par le même agent : le coefficient
« 0,2212 » cité dans `REPONSE_ORDRE53.md` n'existait dans aucun script du
dépôt — corrigé avec `verifier_coefficient_ralentissement.py` (le calcul
est réel, basé sur les 3 points publiés par dipankar au tour 51, mais
n'avait jamais été sauvé).

## Prochaines étapes concrètes

Le fil H15/excursions/optimiseur hybride est **clos** (17/09/2026).

1. **Ralentissement critique refait avec l'optimiseur hybride, puis à
   résolution fine — FAIT le 17/09, résultat NUANCÉ, considéré assez
   clos pour l'instant.** `verifier_ralentissement_hybride.py` : à
   résolution grossière (pas de 200), 800/1000/1000 — semblait plat. À
   résolution fine (pas de 20) : **760/880/940 — une vraie hausse
   monotone apparaît** (ratios 1,158 puis 1,068), mais bien plus faible
   qu'une loi `(delta_c-delta)^(-1/2)` classique (qui prédirait ~3,16×
   par décade). **Lecture retenue : ni H6 pur ni H11 pur** — un vrai
   ralentissement critique existe, mais amorti, plausiblement par le
   couplage à la dynamique rapide du récepteur (SGD). Laissé comme
   résultat intermédiaire honnête plutôt que forcé vers un camp.
2. **`k(R)` reste sans mécanisme fermé — `beta2` testée et réfutée aussi
   (FAIT le 17/09).** `verifier_k_beta2.py` : k(R_init=0,60) varie de
   1,5676 à 1,5883 sur beta2=0,99→0,9999, un effet minuscule (0,02)
   comparé à la dérive totale avec R_init (~1,03). Ni beta1 ni beta2
   n'expliquent la dérive. **Tentative sur le jouet à 2 référents FAITE
   mais non concluante** (`verifier_k_jouet.py`) : R_init=0,75 semble
   hors de la région graduée du jouet à son propre `delta_c` — le
   jouet a sa propre géométrie, comparer directement les mêmes R_init
   n'est pas valide sans d'abord localiser la coordonnée R de SON
   point selle. **Reste la seule piste vraiment non testée : les 25
   autres lignes du système complet, correctement isolées** — referaire
   le protocole pin-and-falsify sur le système complet en gardant N=27
   dans les formules mais en variant seulement le NOMBRE de référents
   qui participent réellement à la collision (au lieu du jouet à 2
   catégories qui change aussi N).
3. **RÉSOLU le 17/09/2026 — résultat fort.** Deuxième égalité retrouvée
   (`replay_23_25.py`, référents 23/25, message 13, `default_rng(50000)`)
   et le mécanisme prior-asymétrique reproduit dessus
   (`verifier_prior_23_25.py`) : **R identique à 6 chiffres à delta=0,01
   et delta=0,013** par rapport aux résultats sur référents 3/4, même
   effondrement vers 1/27 à delta=0,02. **Le mécanisme n'est pas
   spécifique à une paire — c'est une propriété de l'objectif lui-même
   (N=27, beta=0,02), partagée par toute collision.** Attendu a
   posteriori (la forme fermée de H7 ne mentionne jamais l'identité des
   référents), mais fallait le vérifier plutôt que le supposer.
4. **`docs/ARTICLE4.md` n'intègre pas les tours 48-52** — à faire si/quand
   ce fil se stabilise assez pour être résumé.
5. Si dipankarsarkar répond entre-temps : sa critique (une vraie, pas
   simulée) devient un NOUVEAU fichier `REPONSE_ORDRE54.md` — mais tout
   ce qu'on a trouvé nous-mêmes en attendant reste dans `REPONSE_ORDRE53.md`
   / `§7.65`, jamais scindé après coup.

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
