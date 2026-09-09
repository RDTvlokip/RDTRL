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

# Les scripts vont dans le dépôt, pas dans /tmp

Demande explicite de Théo le 09/09/2026 : écrire les scripts de
vérification directement dans `src/test3_communication/` (ou le dossier
du test concerné), pas dans `/d/tmp`. Un script utile à une réponse
mérite d'être retrouvable la prochaine fois sans repasser par une chasse
dans le transcript — `/tmp` n'est bon que pour des sorties vraiment
jetables (logs de run), jamais pour le code qui les produit.
