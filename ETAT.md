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
   plus tôt (premier kick à 5153 au lieu de 7869), d'où le nombre
   d'événements 29→33 sur une fenêtre fixe, pas 4 nouveaux kicks
   stationnaires.
4. **Le « reward hacking » en RLHF est-il parfois un franchissement de
   séparatrice avec sensibilité aux conditions initiales (comme K=12,80
   ici) — deux graines identiques en hyperparamètres tombant de part et
   d'autre d'une frontière chaotique, pas deux graines qui « apprennent »
   des choses différentes ?**
5. **La correction de biais d'Adam (`bias1`/`bias2`) est-elle vraiment
   responsable de l'instabilité de début d'entraînement qu'on lui
   attribue, ou sature-t-elle à 1 si tôt (dès quelques milliers de pas,
   vérifié ici) qu'elle est hors-jeu pour presque tout le reste de
   l'entraînement — et le folklore du « warmup aide » masquerait-il en
   fait le régime plancher-de-`v` plutôt que la correction de biais ?**
6. **Confond-on systématiquement « le modèle peut représenter la
   solution optimale » avec « le modèle peut l'ATTEINDRE depuis son
   init typique par descente de gradient » — un problème de bassin
   d'attraction habillé en problème de capacité ?**
7. **Compare-t-on l'importance de coordonnées dans un modèle
   softmax/sigmoïde SANS corriger par le Jacobien de la saturation
   (le mécanisme `comp¹` établi ici) — créditant systématiquement trop
   d'importance aux coordonnées non saturées et trop peu aux
   saturées ?**
8. **« Le modèle a appris X » est-il parfois en réalité « la structure
   de la récompense force X quel que soit ce qui est appris » —
   attribue-t-on à l'apprentissage ce qui est en fait un point fixe
   structurel de l'objectif (comme la formule fermée H7 de ce
   projet) ?**
9. **Les preuves de « ralentissement critique » utilisées pour
   diagnostiquer des transitions de phase dans l'entraînement (grokking
   compris) sont-elles vérifiées comme un vrai ralentissement
   dynamique, ou souffrent-elles du même artefact de seuil discret
   trouvé ici pour `ratio(K)` ?**
10. **En entraînement distribué/fédéré, le « bruit des rounds de
    communication » est-il parfois un kick périodique plancher-de-`v`
    d'Adam, amplifié par des synchronisations peu fréquentes qui
    allongent la phase calme avant que `v` touche son plancher ?**
11. **Le travail de calibration de confiance (temperature scaling)
    corrige-t-il parfois exactement l'artefact de compression `comp¹`
    du softmax plutôt qu'un vrai mécalibrage des logits sous-jacents ?**
12. **Suppose-t-on toujours que « l'effet rétrécit à plus haute
    précision » signifie « c'était du bruit » — alors que ça peut
    vouloir dire, comme ici (la sensibilité à `1e-15` près de K=12,80),
    qu'on est authentiquement SUR une frontière mathématique où la
    distinction bruit/signal classique s'effondre (du vrai chaos, pas
    une erreur de mesure) ?**
13. **L'arrêt anticipé sur « la perte n'a pas progressé depuis N pas »
    est-il vulnérable au même hasard de timing qu'un cycle
    plancher-de-`v` (~500 pas ici) — un run arrêté trop tôt ou trop
    tard selon where dans le cycle la fenêtre de patience est tombée ?**
14. **Quand on compare deux graines dont une « généralise mieux »,
    vérifie-t-on si c'est un vrai apprentissage différent, ou
    simplement de quel côté d'une séparatrice précoce chaque bruit
    d'initialisation est tombé ?**
15. **Le réflexe « augmente la précision pour trancher un désaccord »
    suppose une convergence monotone vers une « vraie » réponse —
    alors qu'augmenter la précision peut révéler qu'il n'existe PAS de
    réponse stable à ce point d'opération (un vrai régime chaotique,
    trouvé ici), rendant « prends juste plus de décimales » un conseil
    activement trompeur dans certains cas ?**
16. **Y a-t-il une différence sous-estimée entre « l'optimiseur a
    convergé » (les paramètres ont arrêté de bouger visiblement) et
    « le système a atteint son vrai point fixe » — étant donné qu'ici,
    Adam et le flot de gradient réel n'ont convergé vers EXACTEMENT le
    même point fixe qu'une fois `lr` assez grand, ce qui suggère que
    beaucoup de runs « convergés » à `lr` typique pourraient être
    bloqués sur un plateau d'artefact, pas le vrai optimum ?**
17. **Le « reward hacking par quasi-égalités » en RL est-il parfois
    prévisible à l'AVANCE depuis la seule structure de la récompense
    (comme la formule H7 de ce projet, où l'identité des référents
    n'apparaît jamais) — c'est-à-dire que la « préférence » apparente
    du modèle est un artefact arbitraire de quelle égalité
    l'optimiseur a fini par trancher, pas une préférence apprise ?**
18. **Teste-t-on les ablations d'hyperparamètres à un écart/une
    précision comparable à la vraie sensibilité du système — ou la
    pratique courante « balaie X, si la métrique ne bouge pas
    visiblement, X n'a pas d'importance » est-elle structurellement
    aveugle aux sensibilités proches d'un seuil (comme le cas limite
    de K=12,5 ici) ?**
19. **Combien de résultats « d'équité/coopération émergente » en RL
    multi-agents sont en fait le même genre de point fixe forcé par la
    symétrie (comme `R=1/2` à `delta=0` ici, une numérologie
    indépendante de N, pas un apprentissage) habillé en comportement
    social appris ?**
20. **« Ablater un composant ne change pas beaucoup le comportement »
    (utilisé pour argumenter qu'un composant n'a pas d'importance, ex.
    élagage de têtes d'attention) reflète-t-il parfois une
    redistribution de masse vers une route alternative (le mécanisme
    masse_fond de ce projet, où retirer 25 lignes ne fait que
    déplacer où va la masse, pas si elle bouge) — c'est-à-dire que
    l'ablation teste l'ATTEIGNABILITÉ d'une route de secours, pas
    l'importance de la route d'origine ?**

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
