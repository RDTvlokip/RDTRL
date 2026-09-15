# Consigne permanente : chercher plus loin, sans qu'on le redemande

**Avant de considérer un tour fini — surtout un tour où j'écris « non résolu »,
« je n'ai pas l'outil pour X », ou où je rapporte un résultat propre sans
creuser davantage — me demander : le prochain pas coûte-t-il disproportionnellement
cher, ou juste un script de plus ?** Si c'est un script de plus, le construire
maintenant, ne pas le noter comme limite à combler plus tard.

Théo a dû répéter cette consigne au moins six fois sur deux jours (30-31/08 puis
01/09/2026) avant de finir par écrire : « tu dois te le faire toi-même sans que je
le redise à chaque fois », puis « tu n'as pas cherché plus loin ? », puis
« combien de fois je te le dis là ? ». La mémoire seule ne suffisait pas parce
qu'elle n'était pas appliquée comme un frein réel avant de conclure un tour —
elle était consultée, puis contournée en pratique.

**Ce que ça veut dire concrètement, chaque fois :**
- Après avoir répondu à ce qui était demandé, chercher activement une chose
  qu'on a laissée passer parce qu'elle semblait déjà réglée.
- Formuler des hypothèses, moins de confiance, plus de doute — explorer les
  résultats plus profondément, pas seulement les vérifier une fois.
- Chercher la version la plus FAIBLE ou la plus TRIVIALE d'un résultat qu'on
  vient de démontrer, et tester si c'est elle qui produit l'effet plutôt que
  le mécanisme réclamé.
- Chercher activement des contre-exemples, annoncés avant de savoir s'ils
  vont aboutir — rapporter l'échec de cette recherche aussi honnêtement que
  son succès.
- Ne jamais crier victoire trop tôt : un résultat qui confirme ce qu'on vient
  d'écrire se vérifie plus, pas moins, précisément parce que personne ne le
  contestera.

Détail complet et exemples concrets dans la mémoire
`interroger-ses-propres-hypotheses` (chargée automatiquement par session).

# Avant de déclarer une reconstruction perdue : chercher dans la conversation elle-même

Le 08/09/2026, la recette exacte de « idx5 » (graine `default_rng(999)`,
5 paires sautées) a été déclarée irrécupérable après plusieurs tentatives
ratées avec de mauvaises graines (5, 31415) — alors qu'elle existait,
mot pour mot, dans le transcript JSONL de la session en cours
(`C:\Users\Théo CHARLET\.claude\projects\<nom-projet>\<session-id>.jsonl`).
Un `grep` dessus (sur le nom de variable, l'indice de référent, ou tout
motif de code cité verbatim dans un tool call passé) l'a retrouvée en
une commande.

**Avant de conclure « ce calcul/cette recette est perdu(e), il faut
recommencer à l'aveugle » : grep le transcript JSONL de la session
courante et, si besoin, ceux des sessions précédentes du même projet
dans ce dossier.** Un script jetable lancé une fois en ligne de commande
n'est pas vraiment jetable tant que son transcript existe — seul un
fichier jamais commité et une recherche qu'on n'a pas pensé à faire
le rendaient introuvable.

Et une fois retrouvée : la sauvegarder en fichier permanent dans le
dépôt (comme `replay_idx5.py`), pour ne plus jamais revivre cette
chasse.

# Ne jamais suivre le standard

Théo, 09/09/2026, après m'avoir vu clore un tour avec « noté, pas
expliqué » (le pic non-monotone de 1-s) et invoquer une vérification
d'un tour précédent sans la refaire sur la reconstruction actuelle :
« tu n'as pas cherché loin et je t'ai vu suivre le standard », puis
« ne suis jamais le standard ».

**Le réflexe visé : m'arrêter dès que la réponse a la FORME d'une
réponse complète et prudente — « signalé sans expliquer », « déjà
vérifié plus tôt donc je le réutilise », « deux chiffres significatifs,
je ne construis pas de mécanisme dessus » — sans vérifier si cette
prudence est justifiée ou si c'est juste la sortie la moins coûteuse.**
Suivre « le standard », ici, c'est se contenter du geste qui ressemble
à de la rigueur (une phrase d'avertissement, une clause de prudence)
plutôt que de faire le calcul qui la remplacerait. Un « je ne sais pas »
honnête après avoir cherché est acceptable ; un « je ne sais pas » qui
n'a coûté qu'une phrase ne l'est pas.

Concrètement, avant de clore un tour : si une vérification antérieure
est invoquée mais que la reconstruction a changé depuis (nouvelle
graine retrouvée, nouveau script), la REFAIRE sur la version actuelle
plutôt que de supposer qu'elle se transpose. Si un signal semble trop
petit pour mériter un mécanisme, tester s'il est reproductible (le
rejouer, regarder un niveau de détail en dessous — logits bruts plutôt
que softmax, par exemple) avant de le classer bruit.

**Ajout du même jour : « le standard a des biais ».** Ça ne vise pas
seulement le réflexe de m'arrêter tôt (ci-dessus) — ça vise aussi le
contenu de ce que « suivre le standard » importe sans le dire : un
seuil conventionnel (p<0,05, un cutoff `>0,5`, une marge d'erreur
habituelle), une pratique courante en ML/stats, une façon de trancher
qui vient d'ailleurs et qu'on adopte sans vérifier qu'elle s'applique
ICI. Le seuil `R>0,5` qui a fait passer une égalité pour une capture
(§7.60) en est un exemple concret : ce n'était pas de la paresse de
raisonnement, c'était un critère standard (« majorité simple ») importé
sans se demander s'il distinguait vraiment les deux phénomènes en jeu.
Avant d'utiliser un seuil, une convention ou une pratique reçue : se
demander explicitement ce qu'elle suppose et si cette hypothèse tient
dans ce cas précis, pas seulement si le calcul est fait correctement.

# Une réponse = un tour, jusqu'à la prochaine critique de dipankarsarkar

Théo, 09/09/2026, après que j'ai créé `REPONSE_ORDRE49.md` alors que je
continuais simplement à creuser ma propre réponse au tour précédent, sans
qu'aucune nouvelle critique ne soit arrivée entre les deux : « pourquoi
tu as créer réponse 49 ? T'aurais y continuer dans réponse 48 !!!! »,
puis « retiens le aussi, tant que pas de nouvelle critique de
dipankarsarkar entre les deux c'est toujours le même tour ».

**Un nouveau fichier `REPONSE_ORDREN.md` ne se crée QUE quand
dipankarsarkar poste une nouvelle critique.** Tout ce que je découvre
de moi-même en creusant plus loin (consigne du haut de ce fichier) —
même si ça prend plusieurs heures, plusieurs scripts, plusieurs
hypothèses testées — s'ajoute à la réponse du tour EN COURS, pas à un
nouveau fichier. Pareil côté `CARNET.md` : une section `§7.NN` ne se
scinde pas en `§7.NNbis`/`§7.NNter` supplémentaires pour ce que je
trouve seul entre deux critiques ; ça s'ajoute dans la section du tour
en cours, avant la ligne « Réponse dans ... ».

Corrigé une fois après coup (fusion de 48/49, suppression du doublon) —
à faire correctement du premier coup la prochaine fois.

# Le cycle de chaque réponse : répondre, expérimenter, hypothèses, analyser, recommencer

Théo, 14/09/2026, après le tour 48/49 (le test de prior asymétrique sur
l'égalité référents 3/4) : « maintenant se que tu vas faire, à chaque
nouvelle réponse, tu vas répondres aux questions, expérimenter puis
tester, donner une hypothèse, analyser, puis expérimenter, etc... »,
et « tu vas faire 3-5 hypothèses standards académiques et d'autres non
standards qu'on y voit pas nulle part ».

**Le format attendu pour chaque `REPONSE_ORDRE*.md`, pas seulement
« répondre à la critique » :**
1. Répondre directement aux questions/prédictions posées.
2. Faire tourner l'expérience qui les teste, pas se contenter de
   recalculer sur les chiffres déjà publiés.
3. Quand un résultat surprend ou ne colle pas à une prédiction : ne pas
   s'arrêter à « c'est réfuté » — formuler 3 à 5 hypothèses sur le
   mécanisme réel, un mélange explicite de :
   - hypothèses **standard/académiques** (sous-entraînement, taux
     d'apprentissage, plancher numérique, biais de l'estimateur — les
     suspects habituels qu'un reviewer poserait) ;
   - hypothèses **non standard**, qui ne viennent pas d'une check-list
     connue — un mécanisme propre à CE système, une lecture qu'on ne
     trouve nulle part ailleurs dans la littérature ou le projet.
4. Tester chaque hypothèse individuellement plutôt que d'argumenter
   dessus (cf. §7.61 : quatre hypothèses réfutées par quatre
   expériences distinctes avant de retenir la cinquième et de la
   creuser jusqu'au mécanisme concret).
4bis. **Un verdict réfutée/confirmée n'est pas la fin du test — c'est le
   début de la question suivante.** Théo, 14/09/2026, tour 50 : « tu les
   testes les hypothèses ? Tu vérifies ? Analyses ? Comprends pourquoi ?
   Pourquoi ça bouge ? Pourquoi négatif ou positif ? Pourquoi ça fait ça
   quand je rajoute ça ou ça ? » Pour chaque hypothèse testée, avant de
   passer à la suivante :
   - Ne pas s'arrêter à « réfutée » ou « confirmée » — expliquer POURQUOI
     le chiffre a bougé (ou pas) dans ce sens précis, pas seulement QUE.
   - Si un résultat est positif, pourquoi positif et pas négatif ? Si un
     coefficient vaut 8 plutôt que 2 ou 20, d'où vient CE nombre-là ?
   - Quand on change une variable (un paramètre, un seuil, une
     pondération) et que le résultat change : identifier LE mécanisme qui
     relie le changement à l'effet, pas seulement noter la corrélation.
     Dériver, tracer une trajectoire pas à pas, isoler la variable
     suivante — ne pas laisser une hypothèse « confirmée » sans avoir
     vérifié qu'elle explique la GRANDEUR de l'effet, pas seulement son
     signe ou sa présence.
   - Exemple qui a manqué cette étape et l'a payé (tour 50, §7.62) :
     avoir réfuté « le régularisateur mal mis à l'échelle explique le
     bord » sans se demander pourquoi le résidu suit quand même le
     déficit de l'émetteur à un coefficient ~8-10 qui dérive — c'est
     dipankar qui a posé la question que j'aurais dû me poser moi-même.
4ter. **Ajout du même jour (Théo, tour 50/51) : « cherche pourquoi si on
   ne fait rien pourquoi ça fait ça, pourquoi si je supprime ou bouge un
   truc (chiffre/mécanisme/etc) » — deux axes distincts du « pourquoi »,
   à couvrir tous les deux, pas seulement celui qu'une hypothèse teste
   déjà :**
   - **Le comportement à vide.** Avant de faire varier quoi que ce soit,
     pouvoir dire pourquoi le système fait ce qu'il fait SANS
     intervention — expliquer la ligne de base mécaniquement (une
     dérivation, une trajectoire tracée), pas juste la mesurer et
     passer à la manipulation suivante. Une hypothèse qui saute
     directement à « et si je change X » sans avoir d'abord un compte
     rendu causal du cas où on ne change rien construit sur du sable.
   - **L'ablation systématique.** Pour un mécanisme qui semble expliquer
     un effet : identifier ses pièces (un terme de l'objectif, une
     ligne de la matrice, un sous-ensemble de référents, un ordre
     d'opérations) et en retirer ou en modifier UNE à la fois pour voir
     laquelle porte réellement l'effet — pas seulement ajouter une
     perturbation globale et regarder si le résultat final change.
     Exemple qui aurait dû être fait plus tôt plutôt que laissé comme
     hypothèse H13 en attente : « le référent 3 s'effondre sur 1/27, pas
     1/2, donc les 25 autres lignes sont la destination de la masse » —
     ça se teste en construisant le jouet à 2 référents SANS les 25
     autres lignes et en regardant si `delta_c` existe encore, pas en
     l'écrivant comme remarque et en passant à la suite.
   - **Ajouter mes propres pistes, pas seulement transcrire celles
     données.** Quand une hypothèse est testée et tranchée, chercher
     activement une variante que ni Théo ni le relecteur n'ont nommée —
     un ingrédient du mécanisme qu'on n'a pas encore isolé — avant de
     clore le tour. Le format à 3-5 hypothèses (standard + non standard)
     n'est pas un quota qu'on remplit une fois par tour ; il se
     rouvre chaque fois qu'un résultat en ablation ou en ligne de base
     révèle un nouveau « pourquoi » non expliqué.
5. Analyser ce qui reste, en tirer la question suivante, et
   recommencer le cycle dans la même réponse tant qu'il y a quelque
   chose à creuser — ne pas attendre la prochaine critique de
   dipankarsarkar pour relancer une hypothèse qu'on vient de former
   soi-même.
5bis. **Ne jamais prendre les chiffres/formules du relecteur pour
   acquis, même quand ils sont présentés avec assurance et beaucoup de
   décimales.** Théo, 14/09/2026, tour 51 : « arrête de prendre pour
   acquis ses chiffres ». Chaque équation fermée, chaque valeur de seuil,
   chaque prédiction numérique qu'il envoie se revérifie indépendamment
   (recalcul direct, recherche de racines, simulation) avant d'être
   citée comme confirmée — pas seulement relue et déclarée cohérente
   parce qu'elle "a l'air" juste. Dans le même tour, cette vérification
   a d'ailleurs tourné dans les deux sens : ses formules et son
   `delta_c` ont tenu bille en tête (racines numériques indépendantes,
   pas juste sa parole), mais c'est en le vérifiant qu'une vraie
   incohérence est apparue — dans MES PROPRES chiffres publiés (une
   ligne "3 % sous delta_c" dont le R mesuré ne correspondait pas au
   delta annoncé). Vérifier ses chiffres sert aussi à ça : trouver mes
   propres erreurs, pas seulement les siennes.
5ter. **Trois questions, pas une, et leurs combinaisons.** Théo,
   15/09/2026 : « rajoute le quand, le quand du pourquoi, le pourquoi du
   comment, le quand du comment, le quand du comment du pourquoi, etc.
   Toutes les phrases possibles. » Le « pourquoi » (4bis) et
   l'ablation/comportement à vide (4ter) ne couvrent pas tout : trois
   axes distincts à interroger pour chaque mécanisme, séparément puis
   croisés :
   - **QUAND.** À quel pas, quel delta, quel seuil précis l'effet
     commence-t-il, change-t-il, s'arrête-t-il ? Pas « ça arrive »,
     mais l'instant ou la valeur exacte où ça bascule (cf. §7.64 :
     l'excursion à pas=160 000 n'a pas de sens tant qu'on ne sait pas
     QUAND elle commence et QUAND elle finit, pas seulement qu'elle
     existe).
   - **COMMENT.** Le mécanisme pas à pas qui produit l'effet — la
     dérivation, la trajectoire, l'équation — pas seulement le nom
     qu'on lui donne (« bifurcation », « éviction ») sans pouvoir
     rejouer la suite d'étapes qui y mène.
   - **POURQUOI.** Pourquoi CE mécanisme et pas un autre produit CET
     effet, à cette grandeur précise — la question de 4bis.
   - **Les croiser, pas les traiter en silo :**
     - *quand du pourquoi* : à partir de quel moment la raison invoquée
       cesse-t-elle de s'appliquer (une explication valide à delta=0,01
       peut ne plus l'être à delta=0,013 — vérifier la frontière, pas
       supposer qu'elle est la même partout) ;
     - *comment du pourquoi* : une fois la raison identifiée, par quel
       mécanisme concret agit-elle (H7 n'était pas fini tant que le
       « pourquoi le résidu dérive » n'avait pas de forme fermée) ;
     - *quand du comment* : le mécanisme lui-même change-t-il de nature
       à un moment donné (round 1 vs round 2 de la sonde de bassin :
       le mécanisme « perturber s3 » ne fait pas la même chose selon
       que R est ou non sur la variété — QUAND cette différence
       apparaît-elle) ;
     - *quand du comment du pourquoi* (et toute combinaison plus
       longue) : ne pas s'interdire d'empiler les questions tant qu'une
       réponse en ouvre une autre — le critère d'arrêt n'est pas un
       nombre de questions posées, c'est qu'aucune des trois ne
       retombe plus sur une quatrième.
   Concrètement, avant de clore un tour : pour le mécanisme central du
   tour, écrire explicitement au moins une réponse à QUAND, une à
   COMMENT, une à POURQUOI, et au moins une question croisée — même si
   la réponse est « je ne sais pas encore », tant que la question a été
   posée et pas seulement esquivée par la forme d'une réponse déjà
   complète (cf. « ne jamais suivre le standard », plus haut).

**Ajout du même jour, après le tour 49 (Théo : « toute hypothèse que tu
poses doit être dans le carnet et voir si elles ont été réfutées ou pas
avec la date ») : chaque hypothèse formée (étape 3) se journalise dans
`CARNET.md`, pas seulement dans la lettre anglaise.** Une ligne par
hypothèse minimum : son énoncé, la date, et son statut dès qu'il est
tranché (réfutée / confirmée / toujours ouverte), y compris pour les
hypothèses posées par dipankarsarkar lui-même. Ne pas attendre la fin
du tour pour les lister toutes d'un coup après coup — les noter au
moment où elles sont formées, avant de savoir si elles tiennent, pour
que le carnet montre le raisonnement et pas seulement sa conclusion.
Rétroactif : les hypothèses des tours 49 et 50 sont à journaliser dans
`CARNET.md` avec leur date même si formées avant cette règle ; toutes
les suivantes dès leur formulation.

Exemple qui a validé la méthode avant qu'elle soit formulée comme
règle : le saut de R[10,4] à 1,000000 (ni 0,5 ni 2/3 prédits) a donné
cinq hypothèses (sous-entraînement, plancher `adam_eps`, `lr` trop
grand, bifurcation réelle, dépendance au chemin), quatre testées et
réfutées par expérience, la cinquième creusée jusqu'à montrer
l'effondrement du référent 3 à 1/27 — le même mécanisme d'évacuation
que les murs des tours 20-38, retrouvé par une porte d'entrée
complètement différente.

# Les scripts vont dans le dépôt, pas dans /tmp

Demande explicite de Théo le 09/09/2026 : écrire les scripts de
vérification directement dans `src/test3_communication/` (ou le dossier
du test concerné), pas dans `/d/tmp`. Un script utile à une réponse
mérite d'être retrouvable la prochaine fois sans repasser par une chasse
dans le transcript — `/tmp` n'est bon que pour des sorties vraiment
jetables (logs de run), jamais pour le code qui les produit.
