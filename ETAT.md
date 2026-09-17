# État du projet RDTRL — où on en est

*Dernière mise à jour : 17/09/2026. Ce fichier n'est pas un article, c'est
un pense-bête pour reprendre le travail dans une nouvelle conversation
sans tout re-raconter.*

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
