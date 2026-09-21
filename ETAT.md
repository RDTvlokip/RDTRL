# État du projet RDTRL — où on en est

## 20 questions inhabituelles mais logiques, posées le 21/09/2026 —
## à répondre UNE PAR UNE après la prochaine compaction, chacune avec
## son propre commit (test si possible, sinon argumentation fermée)

Demandées par Théo : des questions que « personne n'aime » dans le
monde du ML/LLM/RL — pas des questions philosophiques, des questions
qui découlent logiquement de ce qui a déjà été trouvé et vérifié dans
CE projet, mais qui dérangent parce qu'elles remettent en cause une
pratique ou une explication reçue. Chacune se teste ou s'argumente
individuellement — ne pas répondre en bloc, une réponse = un commit,
avec le cycle habituel (hypothèse, test si possible, agent-dipankar
si le résultat est substantiel, journal daté dans `CARNET.md`).

**Consigne explicite de Théo (21/09/2026) : une fois les 20 questions
toutes répondues, lancer un agent style dipankar (worktree isolé) pour
tout vérifier/challenger indépendamment avant de considérer ce fil
clos.** Ne pas oublier cette étape finale.

1. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 1 des
   20 »] Les « loss spikes » périodiques des gros entraînements ont-ils
   une période caractéristique gouvernée par la décroissance de `v` —
   et si oui, baisser `lr` réduit-il vraiment leur fréquence, ou
   seulement leur amplitude ?** Réponse : PARTIELLEMENT connu, pas
   « une question que personne ne pose » comme formulé initialement
   (correction assumée) — le mécanisme qualitatif (décroissance de `v`
   → spike → regonflement → cycle) est déjà publié indépendamment
   (Bai et al., arXiv:2506.04805, ICML 2026, cité et vérifié à
   l'abstract). Ce qui semble rester non publié : la loi quantitative
   précise trouvée ici (période quasi-invariante à `lr`, amplitude
   linéaire jusqu'à rupture entre `lr∈[0,1;0,2]`) et le mécanisme
   `v_pic∝lr²`.
2. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 2 des
   20 »] Le « seuil critique de batch size / learning rate » que
   rapportent certains papiers est-il une vraie bifurcation dynamique,
   ou un artefact de seuil discret (le mécanisme `ratio(K)` trouvé
   ici) ?** Réponse, en deux niveaux : NON pour le papier fondateur
   (McCandlish et al., arXiv:1812.06162 — la « noise scale » est déjà
   traitée comme continue, pas comme bifurcation, vérifié directement
   à l'abstract) — mon hypothèse de départ était trop sévère pour cette
   source précise. PLAUSIBLE mais NON VÉRIFIÉ pour l'usage pratique en
   aval (citer un chiffre unique de « batch size critique » sans
   rappeler le seuil de lecture choisi) — pas de recherche dédiée faite
   sur ce point.
3. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 3 des
   20 »] Le clipping de gradient déplace-t-il les kicks du
   plancher-de-`v` au lieu de les éliminer, puisqu'il ne change rien à
   la récurrence de `v` elle-même ?** Réponse, testée directement sur
   le vrai système (`verifier_clip_gradient_deplace_kicks.py`) : ni
   l'un ni l'autre. Le régime stationnaire de kicks est intact (même
   période ~456-473, même amplitude ~0,0034-0,0037) — ma prédiction
   précommise (écrêtage du regonflement, période raccourcie) est
   réfutée : le clip (à seuil réaliste) n'engage QUE dans les 40
   premiers pas d'entraînement (gradients initiaux plus grands),
   jamais pendant le régime stationnaire. Effet réel trouvé à la
   place : le clip supprime le TRANSITOIRE de rampe initial (~19
   événements avant stabilisation dans le baseline) en empêchant `v`
   de s'inflater tôt — le régime stationnaire s'installe ~2700 pas
   plus tôt (premier kick à **2881** au lieu de 7869, chiffres
   **corrigés** le 21/09 par l'audit interne — mon `pas_min=5000`
   d'origine cachait 5 événements réels, vrais chiffres `+9`
   événements et `~4988` pas d'accélération, pas `+4`/`~2700` comme
   d'abord publié ; mécanisme qualitatif inchangé, même renforcé).
4. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 4 des
   20 »] Le « reward hacking » en RLHF est-il parfois un franchissement
   de séparatrice avec sensibilité aux conditions initiales (comme
   K=12,80 ici) ?** Réponse : le PHÉNOMÈNE (deux runs à config
   identique divergeant qualitativement) est documenté (Anthropic,
   arXiv:2511.18397, existence vérifiée directement par moi) mais le
   MÉCANISME séparatrice-chaotique proposé ici ne semble pas publié —
   ni l'agent ni ma vérification n'ont trouvé de travail reliant
   bifurcation/chaos au reward hacking en RLHF. Hypothèse de recherche
   motivée par l'analogie, PAS un résultat établi — aucun test direct
   possible (pas de système RLHF dans ce projet).
5. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 5 des
   20 »] La correction de biais d'Adam sature-t-elle trop tôt pour
   expliquer l'instabilité de début d'entraînement, et le folklore
   « warmup aide » masque-t-il le régime plancher-de-`v` ?** Réponse en
   deux parties : OUI, saturation confirmée analytiquement (`bias1` en
   `t≈356`, `bias2` en `t≈37412` pas à la précision machine — chiffres
   **corrigés** le 21/09 par l'audit interne, ma première version
   utilisait le mauvais seuil d'arrondi float64 (`2^-52` au lieu de
   `2^-54`), sous-estimant les deux de ~4% ; déjà <1% d'effet dès
   `t≈4600` — hors-jeu bien avant la fin d'un entraînement typique,
   argument inchangé). NON pour la seconde partie au sens direct : la référence
   standard sur le warmup (RAdam, arXiv:1908.03265, vérifié directement
   à l'abstract) n'invoque déjà pas la correction de biais stricte,
   mais la variance de `v` — un concept proche mais distinct de notre
   plancher-de-`v`. Confusion bias-correction/fiabilité-de-v confirmée
   dans la littérature (arXiv:2511.20516), mais pas avec NOTRE
   mécanisme spécifique.
6. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 6 des
   20 »] Confond-on « représentable » avec « atteignable par
   descente de gradient » ?** Réponse : question PAS originale — cette
   distinction est déjà bien établie et nommée (« expressivity vs
   trainability », Zhang et al. NeurIPS 2021 arXiv:2210.12001, Nguyen
   ICML 2018), avec un cas canonique documenté (apprentissage de la
   parité, Shalev-Shwartz et al. ICML 2017). Un écart trouvé et
   corrigé : l'agent a mal attribué à un papier sur le grokking
   (arXiv:2606.17120) le vocabulaire « bifurcation nœud-col » —
   vérifié directement, le papier utilise en fait un cadre
   thermodynamique (transition de phase, hystérésis, échappement
   Arrhenius), distinct de notre H6 (bifurcation déterministe).
7. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 7 des
   20 »] Compare-t-on l'importance de coordonnées SANS corriger par le
   Jacobien de saturation (mécanisme `comp¹`) ?** Réponse : question
   PAS originale — problème déjà nommé « gradient saturation »,
   motivation fondatrice d'Integrated Gradients (Sundararajan et al.,
   ICML 2017, citation vérifiée mot pour mot par moi sur
   `ar5iv.labs.arxiv.org`), avec un exemple quasi-identique à notre
   `comp¹` (`f(x)=1-ReLU(1-x)`, gradient nul malgré effet réel). Reste
   une pratique répandue hors du sous-domaine attribution formel, mais
   non quantifiée par la recherche faite.
8. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 8 des
   20 ». PREMIER ANGLE POTENTIELLEMENT ORIGINAL trouvé sur les 8
   premières questions.] « Le modèle a appris X » est-il parfois
   « la structure de la récompense force X » (point fixe structurel,
   comme H7) ?** Réponse, prudente : le résultat formel le plus proche
   (Skalse et al., NeurIPS 2022, arXiv:2209.13085, théorème de
   constance vérifié mot pour mot par moi : « two reward functions can
   only be unhackable if one of them is constant ») porte sur la
   hackabilité proxy/vraie récompense, PAS sur l'axe précis
   « appris vs forcé structurellement » que H7 illustre. Aucun nom
   établi trouvé pour cette distinction précise — angle
   potentiellement original, mais confiance modérée seulement (une
   seule passe de recherche), à ne pas sur-vendre.
9. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 9 des
   20 ». Deuxième angle potentiellement original.] Le « ralentissement
   critique » du grokking souffre-t-il du même artefact de seuil
   discret que `ratio(K)` ?** Réponse : les papiers fondateurs mesurent
   effectivement via seuil discret (arXiv:2410.04489, existence/sujet
   vérifiés directement par moi ; arXiv:2607.06639, rapport agent). Une
   réaction partielle à la dépendance au seuil existe (ratio de temps
   indépendant du seuil), mais PAS le test précis fait ici (corrélation
   réelle avec la proximité d'une bifurcation) — semble non publié,
   avec la réserve que deux sources pertinentes n'ont pas été lues en
   détail par l'agent (limite signalée, pas un vide confirmé).
10. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 10
    des 20 ». Troisième angle potentiellement non couvert (numérotation
    corrigée le 21/09 par l'audit interne — ce label manquait,
    décalant tous les suivants d'un cran). INCIDENT : confabulation
    moteur de recherche détectée et écartée (agent + moi).] Le bruit
    fédéré est-il un kick
    plancher-de-`v` amplifié par des syncs peu fréquentes ?** Réponse :
    angle non couvert dans la littérature atteinte (FedAdam, FedAdamW,
    DES-LOC, Q-LocalAdam — textes vérifiés par l'agent, un point
    revérifié directement par moi) — l'instabilité y est toujours
    attribuée à l'hétérogénéité non-IID ou à la réinitialisation de `v`
    à zéro, jamais à une décroissance vers un plancher numérique. Une
    première recherche avait produit une phrase quasi-identique à
    l'hypothèse posée, présentée comme publiée — vérifiée fausse
    (absente du texte source) avant d'être acceptée.
11. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 11
    des 20 ». Quatrième angle potentiellement non couvert.] Le
    temperature scaling corrige-t-il l'artefact `comp¹` plutôt qu'un
    vrai mécalibrage ?** Réponse : Guo et al. (ICML 2017,
    arXiv:1706.04599, abstract vérifié directement par moi) cadre la
    calibration comme un problème d'entraînement (profondeur, largeur,
    weight decay, BatchNorm), pas comme un artefact géométrique du
    softmax — aucune distinction formelle trouvée entre les deux
    causes dans la littérature atteinte. Limite assumée : vérification
    arrêtée au niveau abstract, pas de lecture complète de la section
    « Why Does Temperature Scaling Work? ».
12. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 12
    des 20 ». Cinquième angle potentiellement non couvert — voir aussi
    question 15, traitée avec la même recherche.] « L'effet rétrécit à
    plus haute précision » signifie-t-il toujours « c'était du bruit »,
    ou peut-on être sur une vraie frontière chaotique ?** Réponse : le
    chaos formel en entraînement ML est déjà bien établi (Wang et al.,
    NeurIPS 2023, arXiv:2310.15418, existence vérifiée directement par
    moi — Lyapunov/Hölder sur l'objectif, pas directement sur la
    trajectoire, correction mineure au rapport agent) — PAS original en
    soi. Mais la mise en garde méthodologique précise (« plus de
    précision peut ne pas résoudre un désaccord sur une vraie
    frontière ») ne semble pas formulée explicitement dans la
    littérature ML d'entraînement — 5 requêtes agent sans résultat,
    seul un lemme classique de dynamique générale (shadowing lemma) s'en
    approche.
13. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 13
    des 20 ».] L'arrêt anticipé est-il vulnérable au hasard de timing
    d'un cycle plancher-de-`v` ?** Réponse : OUI, confirmé
    empiriquement sur le vrai système
    (`verifier_arret_anticipe_vs_phase_kick.py`) — rapport max/min de
    la durée jusqu'à arrêt jusqu'à 1,89× selon la phase de départ.
    Prédiction précommise (sauts discontinus par cycle) réfutée —
    variation continue. Mécanisme plus riche que prévu : à seuil
    d'amélioration fin (sensible au bruit de ringing sous-kick), la
    vulnérabilité CROÎT avec la patience au lieu de se diluer (effet
    inverse de l'intuition). Bug de calibration initial trouvé et
    corrigé (seuil 4 ordres de grandeur trop grand, test vide).
14. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 14
    des 20 ». Sixième angle potentiellement non couvert.] Compare-t-on
    graines-qui-généralisent-mieux en testant si c'est une séparatrice
    précoce plutôt qu'un vrai apprentissage différent ?** Réponse :
    l'outil existe déjà (Frankle et al., ICML 2020, arXiv:1912.05671,
    « instability analysis », abstract vérifié directement par moi)
    mais ne semble pas mobilisé systématiquement pour CETTE distinction
    précise — la pratique standard (Henderson et al.) traite la
    variance inter-graines comme du bruit à moyenner, pas comme un
    événement de bifurcation à localiser. Une source récente et
    proche (arXiv:2510.05606) pourrait combler ce vide mais n'a été lue
    qu'en abstract par l'agent — non confirmé.
15. **[RÉPONDUE le 21/09/2026 avec la question 12 (même recherche,
    quasi-doublon), voir `CARNET.md` section « Question 12 des 20 ».]
    Le réflexe « augmente la précision pour trancher un désaccord »
    suppose une convergence monotone vers une « vraie » réponse —
    conseil parfois trompeur ?** Réponse : identique à la question 12
    — mise en garde méthodologique précise non trouvée formulée
    explicitement pour l'entraînement ML, seul un lemme classique de
    dynamique générale (shadowing lemma) s'en approche formellement.
16. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 16
    des 20 ». Tension réelle trouvée avec notre propre résultat, pas
    une simple confirmation.] Différence sous-estimée entre
    « convergence apparente » et « vrai point fixe » ?** Réponse :
    le risque est réel (base interne du 18/09 : le flot de gradient
    pur avait besoin de `lr=5,0`, pas `0,02`, pour matcher Adam à 7
    chiffres — un test précommis avait d'abord « échoué » à cause d'un
    budget insuffisant, pas d'un mécanisme manquant), et partiellement
    documenté (formellement en calcul évolutionnaire, arXiv:2505.01036 ;
    informellement en DL). MAIS une théorie récente (arXiv:2511.04622,
    citation vérifiée mot pour mot par moi : « if Adam converges, the
    limit must be a zero of the Adam vector field, rather than a local
    minimizer or critical point ») montre que le point fixe d'Adam
    n'est PAS garanti d'être celui de l'objectif en général — notre
    accord à 7 chiffres est vraisemblablement un cas spécial de notre
    système, pas une propriété générale à sur-généraliser.
17. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 17
    des 20 ». Septième angle potentiellement non couvert.] Le reward
    hacking par quasi-égalités est-il prévisible à l'avance depuis la
    structure de récompense (comme H7) ?** Réponse : le bris de
    symétrie prédictif existe (Soudry et al., arXiv:1710.10345,
    citation vérifiée mot pour mot par moi — GD converge vers la
    direction max-margin, prédictible depuis la structure du problème
    seule), mais côté géométrie d'optimisation, pas côté RL/récompense
    — rien trouvé qui prédise à l'avance quelle branche d'une
    quasi-égalité RL sera choisie depuis la structure de récompense
    seule, seulement des diagnostics après coup. H7 semble à
    l'intersection de deux littératures qui ne se citent pas. Une
    source proche (DPO, biais structurel) vient d'un domaine déjà
    signalé comme moins fiable (question 5) — traitée avec méfiance.
18. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 18
    des 20 ». Huitième angle potentiellement non couvert — la
    synthèse, pas les pièces individuelles.] Les ablations
    d'hyperparamètres sont-elles structurellement aveugles aux
    sensibilités proches d'un seuil ?** Réponse : chaque pièce existe
    déjà séparément — critique du maillage grossier (Bergstra & Bengio
    2012, abstract vérifié via voie de contournement après échec
    d'accès PDF direct), méthodes de sensibilité formelles (HSIC,
    Sobol), et Edge of Stability (Cohen et al. 2021) comme cas précis
    du même phénomène — mais aucune source trouvée ne les relie comme
    trois instances d'un même problème générique. Confiance
    moyenne-haute sur l'absence de synthèse, recherche web seule.
19. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 19
    des 20 ». Neuvième angle potentiellement non couvert, confiance
    faible explicitement assumée. DEUXIÈME incident de confabulation
    détecté (par l'agent lui-même cette fois).] Combien de résultats
    « d'équité émergente » en RL multi-agents sont en fait un point
    fixe forcé par symétrie (comme `R=1/2` ici) ?** Réponse : aucun
    papier trouvé qui revendique une équité/coopération émergente en
    cadre symétrique SANS vérifier si la symétrie force déjà le
    résultat (deux papiers Minority Game présentent la symétrie comme
    LE résultat sans ce contrôle). La critique la plus proche trouvée
    (arXiv:2602.15407, citation vérifiée mot pour mot par moi après
    que l'agent a lui-même détecté et corrigé une confabulation sur ce
    même papier) est adjacente, pas identique — porte sur l'échec du
    mécanisme d'aversion à l'inéquité sous asymétrie, pas sur le point
    fixe combinatoire lui-même. Confiance globale FAIBLE, explicitement
    signalée.
20. **[RÉPONDUE le 21/09/2026, voir `CARNET.md` section « Question 20
    des 20 ». DERNIÈRE des 20, résultat le plus net.] L'ablation qui
    « ne change pas grand-chose » teste-t-elle l'atteignabilité d'une
    route de secours plutôt que l'importance d'origine (mécanisme
    `masse_fond`) ?** Réponse : le mécanisme EXACT existe déjà publié
    sous un autre nom — McGrath et al., « The Hydra Effect: Emergent
    Self-repair in Language Model Computations » (arXiv:2307.15771,
    2023), citation vérifiée mot pour mot par moi : « ablations of one
    attention layer... cause another layer to compensate (which we
    term the Hydra effect) ». Retrouvé indépendamment sur un système
    jouet complètement différent. Ce qui reste non couvert : la
    littérature d'élagage classique (Michel et al. 2019, cadrage
    « redondance » statique) et celle de l'effet Hydra/backup-heads
    restent largement séparées, sans principe méthodologique général
    explicite reliant les deux.

## LES 20 QUESTIONS SONT TOUTES RÉPONDUES (21/09/2026)

Chacune avec son commit dédié (`git log`), détail complet dans
`CARNET.md`. Prochaine étape obligatoire (consigne de Théo,
21/09/2026) : lancer un agent style dipankarsarkar (worktree isolé)
pour un audit global des 20 réponses avant de clore définitivement ce
fil.

---

*Dernière mise à jour : 20/09/2026 — le VRAI dipankarsarkar a répondu
une SECONDE fois (tour 54, sur `REPONSE_ORDRE54.md`). Répondu dans
`docs/REPONSE_ORDRE55.md` (gitignoré), journalisé dans `CARNET.md`
(nouvelle section « VRAIE CRITIQUE DE DIPANKARSARKAR, tour 54 », tout
en bas du fichier). Ses trois points : H1 (courbure→Jacobien) et H3
(abandonner, un seul mécanisme) convergent avec ce qu'on avait déjà
trouvé nous-mêmes ce tour, avant sa lettre. H2 (sa prédiction chiffrée
sur `s[4,10]`) est RÉFUTÉE par 7 ordres de grandeur. **Ses trois tests
précommis ont TOUS été exécutés le même jour** (voir piste 0) — sa
défense "mauvais instant" est close, le battement partagé n'est pas le
planning de biais d'Adam, et le contrôle `delta=0` a donné une
troisième réponse plus fine que sa dichotomie. **En attente de sa
prochaine réponse.**

Ce fichier n'est pas un article, c'est un pense-bête pour reprendre le
travail dans une nouvelle conversation sans tout re-raconter — l'historique
complet vit dans `docs/CARNET.md` (daté, hypothèse par hypothèse) et
dans `git log`.*

## Pistes ouvertes, par ordre de priorité probable

0. **Tour 54 de dipankar — ses trois tests précommis, TOUS exécutés le
   20/09/2026 (plus deux trous trouvés et corrigés dans la lettre
   après que Théo a demandé "tu as répondu à toutes ses questions ?").**
   Test 1 (grille-1 sur pas=60000) : `1-s[4,10]∈[4,66e-15;4,89e-15]`,
   trois ordres de grandeur sous son seuil de réouverture (`5e-12`) —
   sa défense "mauvais instant" est close. Test 2 (facteur de
   correction de biais d'Adam) : saturé à la précision machine dès
   pas=60000 (calcul direct) — écarte le "battement partagé = planning
   Adam brut". Test 3 (`delta=0`) : le kick de `R` survit (optimiseur
   pur, confirmé), mais le co-timing avec `s3` ne survit PAS —
   troisième réponse : le kick est indépendant de `delta`, sa
   TRANSMISSION vers `s3` nécessite l'asymétrie de récompense comme
   canal. Plus la trace logit 10 décimales qu'il avait implicitement
   demandée (`verifier_logits_s3_s4_r4_60000.py`) : `logit_s4` lisse,
   aucun accroc ; `logit_s3`/`logit_r4` plongent et récupèrent
   ensemble. **Scripts permanents** :
   `verifier_precommis_dipankar_grille1_s410.py`,
   `verifier_precommis_dipankar_delta0_controle.py`,
   `verifier_logits_s3_s4_r4_60000.py`,
   `verifier_split_s4_r4_hybride.py`. Détail complet : `CARNET.md`,
   section « VRAIE CRITIQUE DE DIPANKARSARKAR, tour 54 ».

0bis. **JOUET À K VARIABLE — fil clos avec un mécanisme complet le
   21/09/2026, après six couches de correction successives.** Le vrai
   mécanisme (trouvé par un agent-dipankar, vérifié indépendamment) :
   chaque trajectoire dépasse un pic vers `t≈50-60` puis redescend et
   remonte lentement ; `ratio(K)=pic/équilibre` croît de façon LISSE
   et MONOTONE avec K (`0,9808` à K=1 → `0,9940` à K=26) — le seuil
   discret `premier_pas_1pct` ne mesure pas un taux de relaxation, il
   mesure si ce ratio dépasse `0,99`, ce qui se produit quelque part
   entre K=10 et K=12,5. **Toute la série de localisations
   successives (« falaise entre K=8/K=20 » → « K=10/K=14 » →
   « K=12/K=13 » → « coin net K=12,80/K=12,95 ») était construite sur
   ce même artefact de seuil discret** — pas une vraie transition
   dynamique nouvelle. Détail complet des six couches de correction et
   du mécanisme final : `CARNET.md`, section « SYNTHÈSE FINALE, jouet
   à K variable, 21/09/2026 » (tout en bas du fichier). Script :
   `verifier_ratio_pic_equilibre_mecanisme.py`. **Reste ouvert** :
   `ratio(K)` n'a pas été resserré finement autour de sa traversée de
   `0,99` (K=10-12,5 seulement, pas de balayage fin) ; le taux de
   relaxation lui-même n'est pas purement exponentiel (change de pente
   selon la fenêtre, mais reste K-invariant à chaque échelle) ; jamais
   inspecté côté émetteur (`s3`,`s4`), seulement `r4`.

0ter. **Généralisation ML au sens large — loi d'échelle période/amplitude
   du mécanisme plancher-de-`v` vs `lr`, testée sur le VRAI système.**
   Prédiction fermée confirmée puis affinée par un agent-dipankar,
   revérifiée indépendamment (chiffres identiques) : la période du
   cycle est quasi indépendante de `lr` (encore PLUS vrai en régime
   stationnaire que mes premiers chiffres, contaminés par un
   transitoire ramping sur ~19 événements) ; l'amplitude croît
   linéairement avec `lr` mais **cette loi CASSE dès `lr∈[0,1;0,2]`**
   (pas à un `lr` lointain comme je le supposais). Mécanisme du pic
   post-kick de `v` trouvé et confirmé (`v_pic∝lr²`, exact à 5
   chiffres). Ma propre hypothèse pour la cassure (saturation du pas
   normalisé d'Adam) est RÉFUTÉE par mesure directe. Hypothèse
   alternative de l'agent (compression softmax en aval, mécanisme
   comp¹) précommise, **pas encore testée** : mesurer `Δlogit` plutôt
   que `ΔR4` au pic du kick pour `lr∈{0,025;0,1;0,2;0,3}`. Mise en
   garde méthodologique notée mais PAS vérifiée : « loss spikes » et
   « augmenter `adam_eps` » sont peut-être déjà des phénomènes/remèdes
   connus en ML — à vérifier par une vraie recherche avant de
   revendiquer une nouveauté. Détail complet : `CARNET.md`, section
   « Généralisation au ML au sens large ». Scripts :
   `verifier_kick_periode_amplitude_vs_lr.py`,
   `verifier_kick_cassure_loi_lr.py`.

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
   **Cassure `beta2` ÉLUCIDÉE (reprise après 18h) : ce n'est pas un
   déplacement de moyenne, c'est un mélange de deux populations de
   cycles.** Agent-dipankar a dérivé la vraie récurrence de `v` avec
   terme de gradient de fond `G`, s'est lui-même corrigé d'un biais de
   sélection (transitoire de démarrage donnant un faux "match parfait"
   au premier passage), puis a montré sur les kicks stationnaires
   (15-23, n=8/beta2) que le cycle "propre" continue de décroître
   monotonement (`t*≈437→84→68`), mais qu'une fraction croissante des
   intervalles devient "coincée" (`0/8→1/8→5/8`) — une fois cette
   fraction >50%, la MÉDIANE bascule sur la population coincée.
   **Vérifié indépendamment sans nouveau calcul** : le rapport
   max/médiane de l'espacement (déjà mesuré ce tour) passe de `1,14×`
   (beta2=0,999, spread étroit) à `8-10×` (0,995/0,99, spread massif)
   — signature exacte du mélange. **Reste ouvert** : test précommis à
   `beta2=0,985` (n≥40 kicks) pas encore exécuté ; le coefficient `a`
   du col H6-direct (item 4 plus bas) n'a pas avancé.
   **Jouet à K variable côté émetteur** (`verifier_jouet_k_emetteur_variable.py`,
   généralise le "26" de H7) — **validation forte, revue par
   agent-dipankar, une erreur trouvée et corrigée** : collapse exact à
   `1/(K+1)` pour K=1,8,26 ; `delta_c(K=1)=0,018699` (reproduit le
   bracket H13) ; `delta_c(K=26)=0,013438` contre le vrai système
   `0,0134372` — écart réel **0,006%** (j'avais écrit ~0,03%, faux
   d'un facteur 5, corrigé). Pli d'équilibre confirmé indépendamment
   (Newton, `det(J)→0`, bistabilité directe). **Puzzle non résolu
   trouvé et vérifié indépendamment par moi (chiffres identiques)** :
   K=26 (pli collé au seuil) montre un ralentissement critique PLAT
   (40/60/60 pas), K=1 (pli 12× plus loin) ralentit ×4,25 (80/300/340)
   — la proximité du pli ne prédit PAS la signature dynamique, sens
   inverse de l'attendu. Courbe `flip_s3(R_init)` mesurée (monotone,
   confirmée deux fois), conversion en `k_fit` comparable au vrai
   système pas encore faite. Test précommis à K=8/K=20 pas encore
   exécuté. Scripts permanents : `verifier_meanfield_fold_toy_k.py`,
   `verifier_meanfield_fold_diag_toy_k.py`,
   `verifier_ralentissement_toy_k26.py`,
   `verifier_ralentissement_toy_k1_controle.py`.
   Détail complet, chiffres exacts, statut de vérification de chaque
   affirmation : `CARNET.md`, section « Piste émetteur (le « 26 » de
   H7) — jouet à K variable ».

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
dipankar.cc). L'échange est à son **55e tour** (sa dernière critique,
tour 54, a reçu réponse dans `REPONSE_ORDRE55.md` ; en attente de sa
prochaine réponse).

## Où sont les documents

- `docs/CARNET.md` — le notebook complet, en français, daté, avec CHAQUE
  hypothèse journalisée (posée le, statut, réfutée/confirmée/rouverte).
  Sections actives en ce moment (tout en bas du fichier, dans l'ordre) :
  « VRAIE CRITIQUE DE DIPANKARSARKAR, tour 54 », « Piste émetteur (le
  « 26 » de H7) — jouet à K variable », « SYNTHÈSE FINALE, jouet à K
  variable », « Généralisation au ML au sens large ».
- `docs/REPONSE_ORDRE55.md` — la lettre anglaise en cours (réponse au
  tour 54 de dipankar), **gitignorée**. **NE PAS MODIFIER sans l'accord
  explicite de Théo (consigne du 20/09/2026).** (`REPONSE_ORDRE54.md`
  a déjà été envoyée et a reçu sa réponse — c'est de l'historique,
  plus la lettre active.)
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
excursions de `R` (et l'absence d'excursion visible de `s3`) sont des
**« kicks » d'un oscillateur de relaxation par plancher numérique de
`v` (exp_avg_sq)**, confirmé de bout en bout (six vérifications
indépendantes, piste 1 de `ETAT.md`) — pas un cycle périodique à
période fixe, un processus récurrent d'amplitude quasi constante
(~0,0037) toutes les ~460-500 pas dont la loi de dépendance à `beta2`
reste partiellement non élucidée entre `0,995` et `0,99`.

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
- **Ne pas modifier `docs/REPONSE_ORDRE55.md` sans l'accord explicite de
  Théo** (consigne du 20/09/2026).
