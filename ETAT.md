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

**Un agent jouant le rôle de dipankarsarkar** (lancé le 17/09 pour
continuer à critiquer en son absence) a trouvé un vrai problème : le test
SGD qui avait « réfuté » H15 (l'hypothèse qu'Adam masque le ralentissement
critique) avait en réalité le **référent 3 gelé** — son gradient brut est
trop petit près de la saturation pour bouger sous SGD à taux unique, donc
le test ne mesurait rien sur le ralentissement, seulement un récepteur
seul convergeant contre un émetteur figé. **Vérifié indépendamment,
confirmé.** Statut H15 remis à "rouverte" dans `CARNET.md` §7.65.

**Correction tentée et PARTIELLEMENT ÉCHOUÉE** (`verifier_sgd_pondere.py`) :
SGD à deux taux, émetteur compensé pour son gradient minuscule. Premier
essai (calibré sur la norme du tenseur émetteur ENTIER, 729 cases) :
raté, dominé par d'autres lignes que [3,10], s3 n'a quasi pas bougé
malgré lr_e=1,5e5. Deuxième essai (gradient précis de la case [3,10]
seule) : le gradient réel est encore plus minuscule (8,65e-14), ce qui
demanderait lr_e≈1,39e11 — beaucoup trop dangereux à appliquer sur tout
le tenseur émetteur d'un coup (risque d'exploser les autres cases).

**Piste retenue pour la suite, pas encore tentée** : optimiseur hybride
— Adam sur l'émetteur SEUL (pour compenser proprement son gradient qui
varie sur des ordres de grandeur pendant l'entraînement), SGD pur sur le
récepteur seul. Teste une question plus précise que "SGD partout" :
est-ce l'adaptativité du RÉCEPTEUR ou celle de l'ÉMETTEUR qui porte les
excursions ? Script à écrire : `verifier_optimiseur_hybride.py` (nom
prévu, pas encore créé).

Un deuxième point a aussi été trouvé par le même agent : le coefficient
« 0,2212 » cité dans `REPONSE_ORDRE53.md` n'existait dans aucun script du
dépôt — corrigé avec `verifier_coefficient_ralentissement.py` (le calcul
est réel, basé sur les 3 points publiés par dipankar au tour 51, mais
n'avait jamais été sauvé).

## Prochaines étapes concrètes

1. Lire le résultat de `verifier_sgd_pondere.py` (voir sa sortie, ou le
   relancer : `python src/test3_communication/verifier_sgd_pondere.py`).
2. Selon le résultat : soit H15 revient réfutée pour de bon (SGD à taux
   corrigé toujours plat), soit un vrai ralentissement critique apparaît
   (le temps de convergence augmente près de `delta_c`) — dans ce cas H6
   (nœud-col) redevient plus solide et H11 (crise de bord) pourrait
   revenir en jeu.
3. Mettre à jour `docs/CARNET.md` §7.65 et `docs/REPONSE_ORDRE53.md` avec
   le résultat, en respectant le format habituel (journal d'hypothèses
   daté, scripts commités).
4. `git commit` + `git push` (le carnet se pousse toujours immédiatement
   après une modification, sans demander).
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
