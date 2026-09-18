# État du projet RDTRL — où on en est

*Dernière mise à jour : 18/09/2026 (mécanisme du chaos H6-direct
tranché : artefact de démarrage à froid d'Adam sur un vrai col
hyperbolique, pas une différence de géométrie — voir piste 3c).
Ce fichier n'est pas un article, c'est un pense-bête pour reprendre le
travail dans une nouvelle conversation sans tout re-raconter.*

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
   M=15, masse totale=25 %  -> delta_c=0,018516  (= M=25 a 25% : REEL, -0,84% a tol=3e-4)
   M=25, masse totale=50 %  -> delta_c=0,018516  (IDENTIQUE a 25% — pas plus)
   ```
   **CORRIGÉ le 18/09/2026 (piste 3a ci-dessous) : le -0,84% ci-dessus
   était mesuré à `tol=3e-4`, un seuil de bissection du même ordre que
   l'effet lui-même (`ε_tol=0,80%`) — recalculé à `tol=1e-5`, le vrai
   chiffre est `-0,6275%`, revérifié indépendamment. L'effet reste
   réel (23× au-dessus du bruit à cette résolution plus fine), juste
   plus petit de ~25% que publié ici. Les deux faits établis
   ci-dessous (seuil, plafond) restent qualitativement valides —
   seule la GRANDEUR du décalage à 25% de masse est corrigée.**
   **Deux faits établis : (1) c'est la MASSE TOTALE de fond qui compte,
   pas M (confirmé — même masse, M différent, même résultat) ; (2)
   l'effet a un SEUIL (entre 8% et 25%) puis un PLAFOND (25% et 50%
   donnent le même décalage, pas de croissance continue).**

   **(a) FAIT — seuil localisé précisément entre 17,7422 % et
   17,7539 % de masse totale** (18 points au total, bissection, jamais
   de valeur intermédiaire entre les deux `delta_c` — un vrai saut
   binaire, pas un dégradé). Hypothèse ouverte : ce saut net suggère une
   bascule qualitative entre deux branches du système, pas un décalage
   continu du même point fixe — pas encore démontré directement.

   **(b)/(c) PAS RÉSOLU, mais désormais bien compris pourquoi c'est
   dur — ne pas chercher de raccourci, l'historique complet (5 essais
   de chiffrage tous tombés puis expliqués) est dans `CARNET.md`, fin
   de §7.65. Résumé de l'état final (18/09/2026) :**
   - Le mécanisme récepteur/masse de fond est **réel et solidement
     établi** : seuil net sur `delta_c` (17,74%→17,75%), ET écart de
     **6,5 ordres de grandeur** de vitesse de convergence entre M=0 et
     M=25 à budget égal, mesuré contre le point fixe ALGÉBRIQUE du
     jouet (4 équations couplées, résolu par itération pure — script
     `verifier_point_fixe_jouet_m.py`).
   - **Aucun chiffre de taux précis (`×2,2`, `×2,6`, `×8`, `×39`) n'a
     résisté à la vérification** — chacun s'est effondré en creusant
     plus loin, pour des raisons maintenant identifiées et distinctes :
     bug d'indexage, non-monotonie, cible mobile (référence non
     convergée), et surtout **deux phénomènes réels qui empêchent
     toute pente de trajectoire d'être fiable près d'un pli** : (i) un
     « fantôme » déterministe du pli (décroissance en `1/t`, pas
     exponentielle, prédiction théorique standard des plis, confirmée
     par un agent) — indépendant d'Adam ; (ii) les excursions H15
     (second moment d'Adam) — **confirmées sur ce jouet aussi**, via
     variation de `beta2` (`verifier_h15_jouet_beta2.py` :
     `beta2=0,9` oscille en continu, `beta2=0,999` a de longues
     périodes stables — signature nette, pas ambiguë).
   - **Nouvelle question ouverte, non résolue, trouvée en vérifiant
     autre chose** : M=0 et M=25 sont à la MÊME distance de leur pli
     respectif (0,84% d'écart) mais convergent avec 6,5 décades
     d'écart — `M` change donc la GÉOMÉTRIE LOCALE du pli, pas
     seulement sa position. Pourquoi, pas encore exploré.
   - Le protocole ODE+pin-and-falsify (déjà utilisé au tour 52) reste
     la bonne voie pour chiffrer le lien à `k∈[1,42;2,45]`, mais
     **n'est pas totalement immunisé non plus** (bande de
     classification irréductible `~1/T²` près du pli, argument
     théorique accepté) — à construire en rapportant un INTERVALLE
     pour `delta_c`, pas un point, avec budget adaptatif. La piste
     émetteur (le `26` de H7) reste une hypothèse complémentaire,
     jamais testée.

   **PIVOT décidé le 18/09/2026 (Théo : « on n'arrête pas ») — noté
   explicitement pour la reprise, trois pistes distinctes :**

   **3a. FAITE le 18/09/2026 (Théo : « commence par 3a »).** Fonctions
   de branche dérivées, ODE à deux échelles de temps construite
   (`verifier_ode_jouet_m.py`). **Deux corrections en cours de route,
   toutes deux trouvées puis vérifiées indépendamment (un bug de
   grille faux de 40%, un chiffre de sensibilité d'agent lui-même faux
   de 10 ordres de grandeur — corrigé à 8e-21, cohérent avec la borne
   théorique).** Résultat central : **M est négligeable sur le pli
   quasi-statique** (~21 ordres de grandeur sous `e3+e4`, confirmé
   deux fois). Ce résultat a ensuite servi à corriger un chiffre déjà
   publié : le décalage `-0,84%` de `delta_c(M)` (ligne ci-dessus,
   piste 3) était **gonflé de ~25% par la résolution de la bissection
   elle-même** (`ε_tol=(tol/2)/delta_c=0,80%` à `tol=3e-4`, du même
   ordre que l'effet rapporté) — recalculé à `tol=1e-5` :
   **-0,6275%**, revérifié indépendamment au 4e chiffre significatif.
   L'effet SURVIT à la correction (23× au-dessus du plancher de bruit
   à cette résolution) — un vrai canal manquant existe dans la
   réduction quasi-statique, mais **l'hypothèse `s4` non saturé est
   réfutée** (`s4=1,0000000000` à 10 décimales, M=0 et M=25, de 90% à
   99% de `delta_c`, vérifié indépendamment). Protocole précommis non
   encore lancé (coûteux, ~15 min/point) : comparer le décalage à
   `lr=0,2` contre `lr=0,05` pour trancher artefact-Adam vs vrai canal
   manquant indépendant de l'optimiseur. Détail complet : `CARNET.md`
   fin de §7.65.

   **Protocole lr exécuté le 18/09/2026 (Théo : « continue à
   chercher »).** `lr=0,05` donne un décalage de **-1,0447%** (contre
   -0,6275% à `lr=0,2`) — **le décalage dépend bien significativement
   de `lr`.** Un agent a réfuté mon explication naïve pour ce
   résultat (résidu de convergence) et proposé un mécanisme plus fin
   (temps de franchissement du col, pas résidu final) — **mais SON
   PROPRE tableau de vérification contenait une vraie erreur sur
   M=25, trouvée en le rejouant deux fois moi-même** (`s3=0,4998`
   reproductible bit-à-bit, pas `0,9973` comme rapporté). En
   corrigeant sa fenêtre de mesure (le vrai col est à `s3≈0,9958`,
   pas `0,5`), confirmé : **M=25 passe 11-13× moins de temps dans la
   région lente du VRAI pli que M=0** — un effet dynamique réel,
   distinct de la géométrie statique (identique). Mais M=25 finit
   ensuite bloqué à `s3=0,5`, un attracteur DÉJÀ DOCUMENTÉ dans le
   docstring de `bissecter_delta_c` (pas une découverte de cette
   session) — dont l'interaction précise avec M reste non élucidée.

   **MÉCANISME FINAL ÉLUCIDÉ le 18/09/2026 (Théo : « on continue, je
   n'aime pas rester sans réponse »).** Trace dense `s3(t)` pour M=0
   et M=25 à leur propre `delta_c` : les deux atteignent la région du
   pli vite (`t≈300-500`) puis divergent — M=0 s'y fixe et y reste
   (`s3=0,99643232` jusqu'à t=40000) ; M=25 la traverse puis tombe à
   `s3=0,5` et s'y fixe. **`s3=0,5` n'est PAS un attracteur séparé
   mystérieux — vérifié : `x_br(r4)` colle exactement à `s3` mesuré à
   chaque instant, et `s3=0,5=x_br(R=1)` exactement.** Le mécanisme
   complet : **pour M=25, le récepteur file jusqu'à `R=1` (saturation
   totale, committement au challenger) au lieu de se stabiliser sur la
   valeur intermédiaire de la branche graduée stable comme le fait
   M=0.** Cohérent avec la construction du jouet (M est strictement
   côté récepteur) : ce n'est pas la fonction de branche de l'émetteur
   ni la géométrie du pli qui changent avec M — c'est LA STABILITÉ DE
   LA BRANCHE GRADUÉE côté récepteur, qui se déstabilise vers une
   saturation complète plutôt que de rester sur son équilibre
   intermédiaire. **Fil de la piste 3 (masse de fond) refermé avec un
   mécanisme complet, pas juste un constat.**

   **POURQUOI la réduction (x,R) ne voit pas ce canal — élucidé
   structurellement.** Dérivées de branche recalculées à la valeur R
   de la branche graduée STABLE (pas seulement au pli) : identiques à
   15 chiffres pour tout M. **La réduction (x,R) est donc
   M-indépendante PARTOUT, pas seulement au pli — le canal manquant
   ne peut structurellement pas y être.** Candidat bien étayé : la
   masse de fond elle-même reste non négligeable jusqu'à `t≈5000-8000`
   (mesuré), exactement la fenêtre où `R` évolue vers sa valeur
   finale — l'hypothèse de quasi-stationnarité sous-jacente à
   `R_br(x;M)` (vraie à l'équilibre, vérifiée plusieurs fois) ne
   tient pas assez longtemps pour M>0 pendant que le sort de la
   branche se joue. Piste complète pour une session future : ajouter
   une 3e équation (`d(masse_autres)/dt`) et refaire l'analyse en 3D.

   **Épisode du 18/09/2026 (Théo : « continue, pose des hypothèses
   inconnues ») — une alerte sérieuse levée, puis RÉTRACTÉE après
   vérification plus poussée.** Gradient naturel implémenté pour
   l'émetteur (divise par `s3(1-s3)`, compense l'aplatissement du
   sigmoïde) a d'abord semblé montrer une contradiction de signe entre
   la réduction (x,R) et la simulation directe (le plateau Adam à
   `s3=0,9964` confirmé robuste sur 400000 pas entre-temps). **En
   creusant, la contradiction s'est révélée être une comparaison à la
   mauvaise variable** (R algébrique au lieu de R réellement mesuré,
   qui n'est pas à l'équilibre pendant le transitoire — déjà su) —
   **aucune vraie contradiction.** Ce qui restait (un facteur
   d'échelle ~3000-6000×) n'était pas non plus une simple constante
   manquante, mais un **mauvais choix de coordonnée** : la dynamique
   est linéaire et EXACTE en espace LOGIT (`z3`), pas en espace
   probabilité (`x`) — `dz3/dt = lr*(poids3*r3-(β/N)*z3)`, vérifié à
   ratio=1,0000 exact (pas approximatif) à deux `lr` différents. La
   compression exponentielle du sigmoïde masquait, en espace `x`, un
   écart réel de 17,6 unités en espace logit derrière une différence
   de probabilité de ~2,5e-4. **Le facteur ~2× en plus à M=25
   s'explique entièrement par l'approximation déjà connue `r3≈1-R`**
   (fausse de 67% à M=25), pas un second phénomène. La réduction
   (x,R) n'était donc pas cassée — juste mal exprimée pour analyser la
   dynamique loin du pli.

   **CLÔTURE FINALE le 18/09/2026 (goal actif : « continue, ne
   t'arrête pas »).** Le plateau Adam s'explique bien par la même
   relation exacte — et son jumeau côté récepteur, jamais écrit avant
   ce tour, a été dérivé par un troisième agent :
   `u*=(1/beta)*[(1-delta)*s3-(1+delta)*s4]`, `r4*=1-sigmoid(u*)`.
   **Vérifié indépendamment : `r4*=0,8852106009` contre le plateau
   Adam observé `0,8852130348` — écart 2,4e-6.** Test précommis
   (système couplé émetteur+récepteur en gradient naturel, doit
   converger vers `r4*` en un budget comparable à Adam) : échoué à
   `lr=0,02` (encore loin à t=20000), **confirmé à `lr=5,0`** — écart
   final à la prédiction de **1,845e-7**, et `s3` correspond EXACTEMENT
   au plateau Adam à 7 chiffres significatifs. **Conclusion définitive :
   Adam et le flot de gradient réel convergent vers exactement le
   même point fixe — Adam n'est qu'un accélérateur, confirmé
   analytiquement des deux côtés (émetteur et récepteur), vérifié
   trois fois indépendamment.** Détail complet (y compris la fausse
   alerte initiale et son échec-puis-succès du test précommis, laissés
   dans le carnet plutôt qu'effacés) : `CARNET.md` fin de §7.65.

   **3b. DISSOUTE le 18/09/2026 (Théo : « 3b »).** La question elle-même
   n'avait plus d'objet : le coefficient de courbure `a` du pli
   (`a=F''(x*)/2`, calculé directement avec les fonctions de branche
   de 3a) est **identique (99,064645) pour tout M∈{0,1,3,8,25}
   testé** — stable en h, confirmé par un agent. Et le chiffre "6,5
   décades" qui motivait la question **ne se reproduit pas** : rejouer
   `verifier_point_fixe_jouet_m.py` tel quel donne `39,50` (1,60
   décade), pas 6,5 — vérifié indépendamment. Le commit d'origine
   (`fe737ef`) portait déjà sa propre mise en garde dans son message
   (« fenêtre adaptative de M=25 pas encore propre, relance plus
   longue »), perdue en route jusqu'ici. **Mécanisme exact** : la
   fenêtre adaptative de M=25 ne se ferme JAMAIS avant la fin du
   budget (3000 pas) — la "pente" mesurée est une moyenne prise EN
   PLEIN TRANSITOIRE D'ÉVACUATION DE MASSE (`masse_autres` encore à
   4,8e-6 à t=3000, contre ~4e-11 à l'équilibre), pas une vitesse de
   relaxation asymptotique. **Conclusion 3a+3b réunies : le pli
   algébrique quasi-statique (position ET courbure) ne dépend pas de
   M — chaque différence empirique attribuée à M jusqu'ici (seuil,
   vitesse) était un artefact de budget/résolution de mesure.** Ce
   qui reste réel : un effet résiduel de -0,6275% sur `delta_c`
   (piste 3a) et un temps d'évacuation de fond visiblement
   multi-échelle, pas encore caractérisé proprement (fenêtre trop
   courte pour le voir converger) — les deux pointent vers un
   mécanisme hors de la réduction (x,R), probablement dans la
   dynamique du fond lui-même. Détail complet : `CARNET.md` fin de
   §7.65.

   **3c. FAIT le 18/09/2026 — test direct sur le VRAI système à 27
   référents. Résultat DÉCISIF et inattendu, plus riche que la question
   posée, mais le mécanisme précis reste ouvert.** Plutôt que de
   construire une ODE pour un modèle réduit, applique le protocole
   pin-and-falsify du tour 52 DIRECTEMENT sur le système complet :
   perturbe la masse initiale de 10 « autres » référents (6 à 15) sur
   le message 10 à 25 % de masse totale, bissecte le point de bascule à
   `R_init=0,60`, `delta=0,013`, avec et sans cette perturbation.
   - **Baseline reproduite avec succès** : `flip=0,979616`, `k≈1,613`
     (dans la fourchette `[1,42;2,45]` déjà connue).
   - **Avec 25 % de masse de fond : LE BASSIN S'INVERSE.** `s3_init`
     BAS (0,90-0,999) → gradué ; `s3_init` HAUT (≥0,9995) → effondré —
     l'exact inverse du cas non perturbé. Seuil localisé précisément :
     entre `0,9994521` et `0,9994526`.
   - **Ce n'est PAS universel** : à `R_init=0,75` (plus proche de la
     branche `≈0,794`), aucun renversement, tout reste gradué —
     spécifique à `R_init=0,60` (zone déjà sensible de la séparatrice
     d'origine).
   - **Soumis à un agent-dipankar, plusieurs vraies failles trouvées et
     corrigées** : bissection ne prouve pas une vraie discontinuité
     (juste son critère d'arrêt) ; grille de gradient 100× trop
     grossière ; surtout — **H-course (les deux gradients d'émetteur se
     croisent) RÉFUTÉE** : `grad[4,10]` est constant et négligeable
     partout (référent 4 déjà saturé), `grad3-grad4` ne croise jamais
     zéro. **Testé aussi côté récepteur (`grad_r3-grad_r4`) : même
     verdict, lisse, pas de croisement.**
   - **Conclusion robuste par élimination : le renversement de bassin
     n'est visible dans AUCUN instantané statique (émetteur ou
     récepteur, à `t=0`) — c'est un phénomène authentiquement
     DYNAMIQUE, qui se joue au fil des 40000 pas d'entraînement.**
   - **MÉCANISME ENTIÈREMENT ÉLUCIDÉ en suivant la trajectoire
     complète (`verifier_trajectoire_renversement.py`).** Deux configs
     à `5e-6` d'écart seulement (`s3_init=0,999450` vs `0,999455`)
     restent quasi confondues pendant ~600 pas (approche lente
     commune), puis divergent PRÉCISÉMENT à `pas=600,283`
     (interpolation) : le gradient de l'émetteur change de signe pour
     la config qui survit (rebondit vers la branche graduée), reste
     toujours positif pour celle qui s'effondre (chute catastrophique
     en ~25 pas, `pas≈745-770`). **Cas manuel, textbook, de fantôme de
     nœud-col (`dx/dt=μ+x²`, temps de passage `τ=C/√μ`)** — confirmé
     par un 3e round d'agent-dipankar via un préfacteur `C` d'ordre 1
     et cohérent entre les deux configs (`0,920` et `0,977`, à 6%
     près) — **une vraie confirmation de structure, pas juste une
     analogie.** Risque d'artefact d'optimiseur (état Adam résiduel
     via `fixer_s3`) **définitivement écarté par lecture de code**
     (l'optimiseur est toujours reconstruit neuf APRÈS les `fixer_*`).
   - **L'IDENTITÉ du point selle (même H6, ou différent ?) : creusée à
     fond, TOUJOURS OUVERTE — avec une leçon méthodologique importante
     en prime.** Protocole 3 (différentielle de masse) exécuté :
     `diff_s3` toujours 2-3 ordres de grandeur plus grand que
     `diff_masse_fond` pendant la divergence — penche pour « même
     famille, pas un canal causal séparé ». Protocole 1 (superposition
     temporelle avec H6 d'origine) exécuté : structure qualitative
     identique (fantôme à deux phases), mais taux local apparemment
     `×6-17` plus rapide pour H6-direct — **puis un agent a montré que
     ce `×17` était en grande partie un ARTEFACT DE DÉFINITION de `τ`
     (600 pas écoulés vs 210 pas réellement dans la fenêtre de mesure)
     — corrigé à `×5,94`, cohérent à 4,3% près avec un calcul
     indépendant (`×6,20`). Vérifié moi-même, confirmé au chiffre
     près.** **Conclusion honnête : NI confirmé NI réfuté** — j'avais
     conclu trop vite à « point selle différent » sur un chiffre qui
     s'est effondré à la vérification ; la parenté structurelle
     (fantôme de nœud-col) est solide, mais l'identité exacte demande
     un travail plus rigoureux (ajuster le coefficient quadratique `a`
     directement sur les données, pas le supposer à 1 ; la distance
     statique `mu_H6=5e-6` s'est révélée être une coordonnée
     probablement incorrecte, `×2058` d'écart avec le `mu` implicite du
     taux mesuré).
   - **Étape 2 du protocole tentée (ajuster le coefficient quadratique
     `a` directement) — 3 méthodes essayées, 3 échecs DIAGNOSTIQUÉS,
     aucune n'a produit de chiffre fiable.** (i) Fit sur trajectoire
     unique bien localisée par indice de pas : `a_H6direct=-98,5`, mais
     INSTABLE (`×5,6` selon la fenêtre, trop peu de points) contre
     `a_delayed=-9,7`, LUI stable (`×1,56` seulement) → écart minimum
     robuste `~×8,4`, mais pas de chiffre précis. (ii) Un seul pas Adam
     depuis 21 points frais : sommet absurde — **le premier pas Adam à
     état frais a TOUJOURS une magnitude ≈lr complet (correction de
     biais), écrasant l'info de gradient — piège méthodologique réel,
     pas un bug.** (iii) Regrouper 10 trajectoires des deux côtés du
     seuil : sommet `x0=1,005`, physiquement impossible — **mélanger
     des trajectoires qui récupèrent avec des trajectoires qui
     s'effondrent viole la forme normale locale.** Piste non essayée :
     fit sur UN SEUL côté du seuil, en sautant les premiers pas
     biaisés, avec un point de départ encore plus proche du seuil.
   - **MÉCANISME DU CHAOS ENFIN TRANCHÉ le 18/09/2026 (étapes 4/5 du
     protocole exécutées, puis un agent-dipankar a affiné et un test
     précommis rejoué moi-même a tranché net).** Étape 5 (biais Adam
     seul) réfutée comme cause principale. Étape 4 a trouvé que `R`
     dérive énormément (×67,5/pas, chiffre exact, pas juste "~100×")
     dans la fenêtre H6-direct — première lecture (« R n'est pas
     asservi ») CORRECTE EN CORRÉLATION mais PAS EN CAUSALITÉ : un
     agent a montré, via une jacobienne indépendante au point H6, que
     le champ continu (avant Adam) a une vraie structure de col
     hyperbolique (valeurs propres réelles opposées, `+3,2e-6`/
     `-1,1e-4`, séparation ×35) — **confirmé par ma propre méthode
     indépendante** (différences finies SGD brutes en espace `(s3,R)`,
     méthodologie différente de celle de l'agent, signal net aux
     réglages fiables). Le chaos vient donc d'un **artefact de
     démarrage à froid de l'optimiseur** (m=0,v=0), pas d'un régime
     intrinsèquement non-1D. **Tranché par un test PRÉCOMMIS que j'ai
     rejoué moi-même** (`verifier_injection_moments_adam.py`) : aux
     coordonnées EXACTES de H6-direct, injecter un état Adam interne
     cohérent (sans bouger `s3` ni `R` d'un iota) fait passer les
     changements de signe de `Δs3` de **5 (froid) à 0 (moments
     injectés)** — net, décisif, pas ambigu. Corrections en cours de
     route notées honnêtement : ma lecture « spécifique à
     l'effondrement » réfutée (la trajectoire miroir est tout aussi
     chaotique) ; le ratio `|ΔR/Δs3|` seul n'est pas un bon diagnostic
     (explose même dans des régimes par ailleurs lisses) — le nombre
     de changements de signe de `Δs3` est la métrique robuste ; le
     chiffre annexe de l'agent (séparation ×114 côté cas retardé)
     lui-même signalé par l'agent comme convention-dépendant (×114 ou
     ×255 selon comment les référents de fond sont gelés), pas cité
     avec la même confiance que le ×35 côté H6.
   - **Conséquence actionnable pour l'identité du point selle : la
     comparaison `a_H6direct` vs `a_delayed` n'était pas condamnée par
     une différence de géométrie — elle était condamnée par un
     artefact d'optimiseur, mais PAS aussi facilement corrigible
     qu'espéré.** Recette testée le 18/09/2026 (même tour) : (1)
     fenêtre tardive (attendre que le chaos brut s'éteigne tout seul,
     pas 18+) — marche pour le chaos (0 changement de signe) mais la
     trajectoire a alors QUITTÉ la zone locale du col (`x0` dérive à
     0,925-0,938, loin de 0,994300) ; (2) fit direct sur la trajectoire
     à moments injectés — `a` dépend fortement et continûment de
     `t_injecte` (`+25` à `t=10`, `+4` à `t=40`, pas de plateau), et
     instable même en sous-fenêtre (`-18,7` puis `+28,2`, changement de
     signe). **Les deux échouent pour la MÊME raison de fond : le
     temps qu'il faut pour "nettoyer" l'optimiseur est du même ordre
     que le temps que la trajectoire met à quitter le voisinage du
     col — contrairement au cas retardé qui a ~400 pas de marge avant
     son propre goulot.**
   - **Essai 3, PISTE FERMÉE le 18/09/2026 (Théo : « on continue une
     dernière fois ») — la piste "la plus prometteuse" (vrai
     réchauffement multi-pas sur gradients réels, pas une injection à
     un seul gradient) a été essayée et a ÉCHOUÉ, pire que les deux
     précédentes sur son propre critère.** Mécanisme compris : réchauffer
     sur un point FIXE apprend "ce gradient ne change jamais" (Adam se
     met à appliquer un pas de magnitude `lr` complète à chaque pas de
     réchauffement) plutôt que la vraie courbure locale, qui ne peut
     s'observer qu'en laissant la position réellement bouger — ce que
     seul le cas retardé fait, via 400 pas de vraie trajectoire.
     **Six échecs distincts et diagnostiqués au total pour
     `a_H6direct`. Le seul chemin plausible restant est un vrai travail
     de modélisation (construire une trajectoire d'approche lente pour
     masse_fond=0, pas une astuce d'optimiseur) — mis de côté comme
     travail futur, pas retenté sans fin dans cette session.**
   - **Réponse à la question initiale (« k se déplace-t-il comme
     prévu ? ») : ni oui ni non simplement — le résultat est
     QUALITATIF (réorganisation de la structure de bassin), pas une
     simple lecture d'un `k` différent sur la même table.** Décisif,
     positif ET négatif à la fois (Théo, 18/09 : « il nous faut des
     réponses même négatives ou positives »).
   Scripts : `verifier_masse_fond_systeme_reel.py`,
   `verifier_trajectoire_renversement.py`. Détail complet, avec tous
   les chiffres et le cycle QUAND/COMMENT/POURQUOI complet :
   `CARNET.md`, fin de §7.65 (section « Piste 3c »).

   **3d. Piste 4 — RÉSOLUE le 18/09/2026.** `docs/ARTICLE4.md` intègre
   maintenant les tours 48-52 (test point-fixe-vs-posterior, mécanisme
   H13 du mur, H-course réfutée deux fois, forme fermée du pli et
   `delta_c`, `k` non constant) ET le travail auto-dirigé après le
   tour 52 (0,2212604 + coefficient `D`, universalité de `delta_c`,
   renversement de bassin et fantôme de nœud-col, six échecs
   diagnostiqués sur le coefficient `a`) — nouvelle « Part three »,
   titre/intro/crédit/Summary/Q&A/tableau "what died" tous mis à jour
   pour refléter la troisième ère (avec lecteur, tours 6-52, puis sans
   lecteur, auto-vérification assumée et documentée comme telle).
   Poussé sur main (commit `6201bb4`).
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
4. **RÉSOLU le 18/09/2026.** `docs/ARTICLE4.md` intègre maintenant les
   tours 48-52 et le travail auto-dirigé qui a suivi (voir piste 3d
   ci-dessus pour le détail).
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
4. **RÉSOLU le 18/09/2026** — voir piste 3d/4 plus haut dans ce fichier.
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
