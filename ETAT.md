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

Le fil H15/excursions/optimiseur hybride est **clos** (17/09/2026). Ce qui
reste ouvert pour continuer, par ordre de priorité probable :

1. **Refaire le test de ralentissement critique correctement**, maintenant
   qu'on a un optimiseur qui entraîne vraiment le référent 3 (l'hybride
   Adam-émetteur/SGD-récepteur, ou directement Adam des deux côtés avec
   des points mieux espacés) — le test §7.63/§7.64 original utilisait le
   même référent gelé par endroits, donc le résultat "pas de
   ralentissement critique" qui avait fait pencher pour H6 (nœud-col)
   mérite d'être revérifié avec le montage propre.
2. **`k(R)` reste sans mécanisme fermé** — H_chemin et H_momentum
   réfutées, il reste "fonction de l'état, probablement liée à `beta2`
   ou aux 25 autres lignes du système complet à 27 référents" — aucune
   des deux pistes testée directement.
3. **Aucune reproduction sur une autre collision/graine** — tout ce
   travail (tours 48-52) repose sur UNE SEULE paire (référents 3/4,
   graine 77777, k=3). Le mur idx5 (référents 0, graine 999) est déjà
   reconstruit (`replay_idx5.py`) et pourrait servir de deuxième cas.
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
