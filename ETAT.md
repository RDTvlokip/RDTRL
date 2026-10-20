# État du projet RDTRL — où on en est

## Tour 79 (vraie critique de dipankarsarkar, 24/10/2026) — LETTRE PRÊTE

**Lettre** : `docs/REPONSE_ORDRE79.md` (gitignorée), PRÊTE À ENVOYER, première personne. La lettre 78
est partie (sa critique du tour 79 y répond) et reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 24/10/2026 (tour 79, PAS simulée) », avant « ## 9. ».

**Sa critique** : il perd son pari (conditionné sur 4e-8 au lieu de 1,811e-8) ; oui, eps/lr est l'invariant du bloc à 2 % de lr 0,05 à 0,3, avec le périmètre (réseau à lr 0,05 et 0,1, réduction au-dessus, moments mis à l'échelle ; c'est un invariant du franchissement) ; la dérive de 1,7 % est dans √v_e3/√v_l3 (0,0544 → 0,0533), qui dépend de x (pente
−0,36 entre 1e-4 et 4,106e-4, −0,035 entre 1e-6 et 1e-5) : x effectif +6 % ; test : à x = 1e-6, eps/lr = 2e-9, partenaire/niveau dérive de moins de 0,5 % de lr 0,05 à 0,3 (50 %) ; question : lancer la réduction à x = 1e-6 sur les quatre lr ?
**Acquis (prédictions poussées avant les runs)** : ses chiffres ✓ ; **x = 1e-6 : la dérive est plus grande** (rapport −3,7 %, partenaire/niveau +5,1 % relatif, d_s et poids +3,3 %, lr p_z +2,6 %), x = 1e-4 : rapport −2,7 %, x = 1e-5 (lr 0,05 → 0,1) −1,2 % ; **le motif du bloc (+,+,+,−) est celui d'une hausse de lr p_z (gain du sender)**, pas d'un
décalage de x (test direct à 1,06 × x, lr 0,05 : partenaire +1,9 %, poids −3,2 %, d_s −0,4 %, rapport −3,3 %) ; rapport des dérives d_s/niveau sur rapport = −0,89 à −0,93 à quatre x ; √v_e3 ∝ lr^0,986 (x = 1e-6) ; les salves : le rapport √v_e3/√v_l3 est invariant par salve (remplissages égaux à 1e-4 près) ; le cycle change énormément
(période 453 → 389 → 206 pas, remplissage 17,5 → 13 → 2 %, quasi-continu à lr 0,3) alors que le bloc au franchissement ne bouge que de 2–5 %.
**Mes ratés** : rapport à 1,06 × x (−3,3 contre −1,5 à −2,5 %) ; amplitudes des dérives à x = 1e-6 ; amplitude de salve ∝ lr^0,69 et non lr ; remplissage de √v_l3 non constant. H79-4 (A_sat ∝ lr) fausse.
**Limites** : réduction seule (lr ≥ 0,2) ; origine de la dépendance en lr du rapport des gradients de e3 et l3 pendant la salve non trouvée ; x = 1e-5 à lr 0,2 et 0,3 : « aucun franchissement » non examiné ; régime quasi continu à lr 0,3, x = 1e-6.
Écart à tes règles : un `python -c` pour refaire des pentes avant de les mettre dans un script.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 79 ; question : le bloc dépend-il du franchissement S = 38 et du gain du sender seulement, pas du cycle ? (oui/non).
2. Chercher pourquoi le rapport des gradients de e3 et l3 en salve dépend de lr (nonlinéarités en l'amplitude ? durée en ln lr ?) ; regarder x = 1e-5 à lr 0,2 et 0,3.
3. Réseau complet à lr 0,2 (moments à l'échelle) et b1 = 0,8 / 0,95.
4. `.gitignore` modifié par l'utilisateur, non commité.

## Tour 78 (vraie critique de dipankarsarkar, 23/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE78.md` (gitignorée), partie, figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 23/10/2026 (tour 78, PAS simulée) », avant « ## 9. ».

**Sa critique** : l'accord à eps 1e-9 compte (quatre accords libres à deux eps) ; sur lr il ne s'attend pas à ce que eps/lr soit l'invariant : lr n'entre pas dans p_z, 38 ne bouge pas, √v_e3 reste ~4e-8 donc eps 2e-10 est un eps plus gros (p_z −3 %, pente log 1,4 → partenaire/niveau ≈ 0,056) ;
pari 0,056 ± 0,002 à 55 % ; question : que vaut √v_e3 au franchissement à lr 0,1 ?
**Acquis (prédictions poussées avant les runs)** : ses chiffres ✓ (4,84e7 ; 4,70e7 ; 2,9 % ; 18 % ; 25 % ; 1,396 ; 0,0561) ; **√v_e3 au franchissement : 3,585e-8 (réduction), 3,605e-8 (réseau complet) à lr 0,1, eps 2e-10, soit 1,98–1,99 fois 1,81e-8 à lr 0,05** ; partenaire/niveau 0,05874 (réseau) et 0,05875 (réduction), poids 0,04874 ;
√v_l3 proportionnel à lr à 0,03 % (invariance d'échelle du bord S_c = 38) ; lr 0,2 et 0,3 avec moments à l'échelle : 0,05937 et 0,05951 (dérive +1,7 %, dans le rapport √v_e3/√v_l3 : 0,0544 → 0,0533) ; son 0,056 est **exact à lr fixé** : lr 0,05, eps 2e-10 : 0,05572 (réseau), 0,05571 (réduction) ; à eps/lr = 1e-9 : 0,06041 ;
√v_e3 vaut ~4e-8 pour x ≤ 1e-6 et 1,8e-8 au delta réel (rapport 0,134 → 0,054). **Mes ratés** : lr 0,2 effondré sans moments à l'échelle ; je n'avais pas prédit la valeur absolue de √v_e3. Son pari raté de 0,0007.
**Limites** : b1 = 0,9 ; le réseau complet jusqu'à lr 0,1 seulement ; l'origine de la dérive de 2 % du rapport √v_e3/√v_l3 non trouvée.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 78 ; question : eps/lr est-il l'invariant du bloc à 2 % de lr 0,05 à 0,3 ? (oui/non).
2. Expliquer la dérive du rapport √v_e3/√v_l3 avec lr (0,0544 → 0,0533) et sa chute avec x (0,134 → 0,054).
3. Réseau complet à lr 0,2–0,3 (moments à l'échelle) ; b1 = 0,8 et 0,95 : le bord (18, 78) et le bloc.
4. `.gitignore` modifié par l'utilisateur, non commité.

## Tour 77 (vraie critique de dipankarsarkar, 22/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE77.md` (gitignorée), partie, figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 22/10/2026 (tour 77, PAS simulée) », avant « ## 9. ».

**Sa critique** : mes cinq accords du tour 76 sont deux nombres libres (d_gap/S_max, d_s/S_max), le reste en découle (arithmétique 2 × 2) ; l'échelle (38,85 contre 38,02) a raté ; à ce x, eps déplace le bloc de 33 % (réduction : partenaire/S_max 0,0441 à eps 1e-9, 0,0585 à 1e-10) et le réseau n'a été lancé qu'à 1e-10 ;
question : à eps 1e-9, même delta, même lr, une salve, le réseau complet tombe-t-il sur 0,0441 ?
**Acquis (prédictions poussées avant les runs)** : son arithmétique ✓ (1,00003 ; 0,05845 ; 0,0484 ; couplage 0,2021) et son compte (deux nombres libres) ✓ ; `verifier_tour77_bloc_eps.py reseau` : **réseau reconstruit nativement à eps 1e-9 : partenaire/niveau 0,04412 sur deux salves (pas 59009 et 59462), poids 0,03499, d_s/niveau 0,07756,
d_gap/niveau 0,96656, fermeture 0,0000** ; eps 1e-10 au franchissement du niveau 38,0 (pas 59456) : 0,05852 ; 0,04851 ; 0,10419 (même niveau des deux côtés, ce que le tour 76 n'avait pas : grandeurs absolues à 0,05–0,5 %) ; rapport des eps 1,326 (réduction 1,327) ; autres x à eps 1e-9 : x = 1e-4 partenaire/niveau 0,02743, poids 0,06269,
d_s 0,08840 ; x = 1e-5 : 0,00982 ; 0,08591 ; 0,09489, identiques à la réduction à 1e-4 près (trois franchissements chacun) ; balayage d'eps de la réduction à x = 4,1e-4 : partenaire/S 0,0616 ; 0,0585 ; 0,0441 ; 0,0242 pour eps 1e-11, 1e-10, 1e-9, 1e-8.
**Mes ratés** : P77-5 (borne ≤ 0,02 à eps 1e-8 : 0,0242). Mes mises (35–45 %) étaient basses et ont toutes tenu. Écart à tes règles : `python -c` et un heredoc vide pour refaire de l'arithmétique avant de la mettre dans le script.
**Limites** : un lr (0,05), un b1 ; une construction du réseau par eps ; seul le bloc (gap, ligne 3) est testé (les lignes 5 et 8 ne sont pas dans le bloc) ; s non comparée à la direction z3.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 77 ; question : le bloc suit-il la réduction quand lr change à eps/lr fixé (lr 0,1, eps 2e-10) ? (oui/non).
2. Tester lr 0,1 (module `reduction_numba_tour71_lr10`/lrvar) contre le réseau complet à lr 0,1 (reconstruction native avec lr changé).
3. Les lignes 5 et 8 de l'émetteur (S ≈ 38 en permanence à eps 1e-10) : même bloc ?
4. Modifications locales non commitées par moi : `.gitignore` (ajout `.tmp.driveupload`), de l'utilisateur.

## Tour 76 (vraie critique de dipankarsarkar, 21/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE76.md` (gitignorée), partie, figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 21/10/2026 (tour 76, PAS simulée) », avant « ## 9. ».

**Sa critique** : mes 13,4 % contre 4,7 % ne sont pas une dilution ; dans la trace du 23/09 (bloc gap-émetteur, symétrique, poids 4,9 % dans P^(1/2) H P^(1/2)) : (S_max − d_gap)/(S_max − d_s) = 0,0515 impose d_s = 4,22, partenaire 2,39,
couplage 8,06 : répulsion de niveaux, pas somme (somme : d_s = 1,95, S_gap = 37,79) ; ce qui tranche : la diagonale de la ligne émetteur et le second niveau ; question : poids pris en coordonnées brutes ou dans P^(1/2) H P^(1/2) ?
**Acquis (prédictions poussées avant les runs, 6 sur 6 tenues)** : poids dans P^(−1/2) H P^(−1/2) (code de `verifier_tour58_bord_stabilite_salve.py`, lignes 63–66, 87) ; son arithmétique ✓ ; rejeu du réseau complet (`verifier_tour76_bloc_2x2.py reseau 59500 59542`) :
au pas 59536 S_gap 37,912, S_max 39,739, part e3 0,0491 (identiques au journal) et bloc (gap, projection de M g sur la ligne 3) **d_gap 37,912, d_s 4,346, couplage 8,041, niveaux 39,739 et 2,519**, fermeture 0,000 : **répulsion de niveaux, son arithmétique exacte** ;
réduction à x = 4,106e-4 (eps 1e-10) : d_s 3,961, couplage 7,691, niveaux 38,024 et 2,2247, poids du sender 0,0485 ; quantités normalisées par S_max identiques au réseau complet à 0,1 % (partenaire 0,0585, d_s 0,1042, couplage 0,2023, poids 0,0485) ; **le partenaire est le mode mou du pli** :
partenaire = 0,92–0,97 × lr p_z |λ_s,nœud(x)| de x = 1e-8 à 4,1e-4 ; partenaire/S_max de 1e-4 (x = 1e-13) à 5,9 % ; ma forme fermée du tour 75 est la TRACE du bloc (exacte à 0,04 % pour x ≤ 1e-8, 6 % trop haute à 4,1e-4). **Mon « dilution par les 26 colonnes » (tour 75) est réfuté.**
**Mes ratés** : aucun sur les six prédictions ; l'erreur est de lecture, du tour 75. J'ai lancé sept processus d'un coup (plafond six).
**Limites** : un delta, un lr, un eps, une salve du réseau complet ; la direction s (projection de M g sur la ligne 3) non comparée explicitement à la direction z3 ; pas de rejeu à delta = 0.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 76 ; question : l'accord à 0,1 % du bloc normalisé entre le réseau complet et la réduction valide-t-il la réduction à cette distance du pli ? (oui/non).
2. Rejeu du réseau complet à d'autres delta (x = 1e-4, 1e-5) pour voir partenaire/S_max suivre √x ; comparer s à la direction z3.
3. Dériver les 3–8 % entre le partenaire et lr p_z |λ_s| (facteur de gain 0,916 du tour 74 ?).
4. Revoir les tours 58–70 où S = lr 2|J_uu|/(√v_l3+eps) ou S_gap sont employés à x ≥ 1e-6.

## Tour 75 (vraie critique de dipankarsarkar, 20/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE75.md` (gitignorée), partie, figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 20/10/2026 (tour 75, PAS simulée) », avant « ## 9. ».

**Sa critique** : oui, déplacer x seul ne sépare pas terme suivant du pli et momentum (les deux en √x) ; équation caractéristique redérivée à la main ; au second ordre μ − 1 = cJ (1 − b1 cJ/(1 − b1)), levier 4, 9, 19 ; mon S à
0,87–0,965 du seuil est peut-être une dent de scie (trace du 23/09 : S_gap 32,4 après une salve, 38 vers 59040, 34,4 à la fin) ; deux lectures : S au démarrage des salves (38), profondeur de la dent contre eps/(√v+eps) ; question :
à b1 = 0,8 et 0,95, les salves démarrent-elles à 18 et 78 ?
**Acquis (prédictions poussées avant les runs)** : ses chiffres ✓ (je corrige mon « ordre cJ/(1−b1) » : le paramètre est b1 cJ/(1−b1), 0,825 contre 0,870 exact à b1 = 0,9) ; `verifier_tour75_salves_depart.py`, réduction lr 0,05, six
états, 1 485–1 965 salves par état : S_c exact (valeur propre raide de lr P J6, seuil exact 2(1+b1)/(1−b1) pour v figé) ; **la croissance de l'amplitude suit l'application avec momentum à 0,01–0,03 % du seuil : les salves
démarrent à 38, 18 et 78** ; rejeu exact à 2e-4 ; la médiane de S_c est la moyenne géométrique creux × après à 0,6–1,8 % ; profondeur de la dent 4,83 → 3,96 quand eps/(√v+eps) passe de 0,003 à 0,329 (mais égale entre 0,003 et 0,032) ;
**mon S du tour 66 omettait un terme** : S = lr [2|J_uu| p_u + k3 p_z], k3 = 2,006e-6, p_z = 1/(√v_e3+eps) + 1/(√v_e3+26 eps), forme fermée à 0,04 % de S_c, terme de 1 à 10 % (13,4 % quand eps → 0) ; le manque 0,87–0,965 du tour 66 =
(S_66/S_c : 0,90–0,99) × (dent : 0,932–0,978).
**Mes ratés** : la sonde E = |Δ²u|/4 n'isole pas le mode raide (argmin de E biaisé de 0,3 à 0,8 %, pente à b1 = 0,8 : 0,642 de ½ ln b1), avec E_alt = |Δ²_t − Δ²_{t−1}|/8 : 0,975 ; mais pente 3–7 % plus lente que ½ ln b1 (0,933 à b1 = 0,95) :
non expliqué ; P75-5 (égalité des deux premiers états), P75-6, P75-7 (b1 = 0,8), P75-10a (0,95), P75-10b.
**Limites** : réduction seule ; pas de réseau complet ; le terme en z3 vaut 13,4 % (eps → 0) dans la réduction contre 4,7 % (S_max 39,74 / S_gap 37,91) dans la trace du 23/09 au delta réel (les 26 colonnes de l'émetteur fusionnées en o3 ?).

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 75 ; question : les 4,7 % entre S_gap et S_max (trace du 23/09, delta réel) sont-ils le même couplage que mon terme k3 p_z, dilué par les 26 colonnes de l'émetteur ? (oui/non).
2. Reproduire dans le réseau complet (torch, `verifier_tour58_bord_stabilite_salve.py`) le terme k3 p_z : S_max − S_gap contre eps ; ou construire une réduction à 26 colonnes d'émetteur.
3. Expliquer la pente de décroissance entre salves (0,95–0,975 de ½ ln b1) : battements de E_alt pour S loin du seuil (θ ≠ π) ou vrai effet.
4. Revoir les tours 58–70 qui utilisent S = lr 2|J_uu|/(√v_l3+eps) (S/seuil, 0,92 à lr 0,3, etc.) avec le terme k3 p_z.

## Tour 74 (vraie critique de dipankarsarkar, 20/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE74.md` (gitignorée), partie, figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 20/10/2026 (tour 74, PAS simulée) », avant « ## 9. ».

**Sa critique** : le 0,84 n'est pas dû à l'échelle seule ; deux tests : (1) valeur propre molle de la forme fermée à x = 1e-6 / 7,30e-8 (0,84 : courbure ; 1,00 : le gain) ; (2) déplacer x
(un terme suivant donne 0,95 à 1e-7, 0,98 à 1e-8). Question : « 0,84 ou 1,00 ? ».
**Acquis (prédictions poussées avant les runs, sauf le test (1), signalé)** : ses chiffres ✓ ; valeur propre molle fermée à 1e-6 : col +7,70e-8 (**1,055**), nœud −6,97e-8 (0,955) → pas 0,84 ;
le dénominateur diagonal d'Adam non plus (A = I + lr P J6 donne 0,985 du modèle) ; **le 0,84–0,87 est le MOMENTUM** : équation caractéristique (μ−b1)(μ−1) = (1−b1) cJ μ, cJ = A−1, reproduit le
jacobien 18-dim à 2e-5 en ln sur tout le séjour (0,01678 à t = 200 … 0,02635 à t = 920) ; l'application à 12 dimensions (x, m) à 5 décimales ; le facteur va de 0,871 à 0,813 ; b1 = 0,8 et 0,95 sur le
même état : 0,9343 et 0,7810 prédits 0,934 et 0,781 ; x = 1e-7 et 1e-8 : lr\* existe (0,305065 ; 0,304070), ln18/ln A = 0,951 et 0,983 (ses 0,95 et 0,98, mais par le momentum, qui va aussi en √x :
son test (2) ne tranche pas) ; modèle corrigé hors échantillon : t(δ) à −0,2 % (x = 1e-6), −0,9 % (x = 1e-7), contre −3,3 à −7,7 % avant. Décomposition à 1e-6, t = 200 : 0,858 = 1,055 × 1,019 × 0,916 × 0,871.
**Mes ratés** : A reproduit mon modèle (0,985), pas le jacobien (P74-1, P74-2) ; plafond d'ablation 1,075 (P74-3) ; rapport à 1e-7 et 1e-8 hors de [0,78 ; 0,90] (0,934 ; 1,020) ; coût de la décade à
1e-7 +10 % au lieu de +15 % (P74-5) ; ln A/modèle monte vers 1,27 à t = 920 au lieu de baisser (P74-13). Retiré : « le taux est gain × λ_s » (lettres 72 et 73) : c'est cJ, le taux en est la racine.
**Limites** : réduction seule ; le dwell n'existe pas à b1 = 0,8 et 0,95 (test sur un même état, pas sur un séjour) ; le facteur 0,916 est numérique ; réseau complet non testé.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 74 ; question : le déplacement de x seul ne tranche pas entre un terme suivant du pli et le momentum, b1 à état fixé le fait (oui/non).
2. Dériver 0,916 en forme fermée (vecteur nul gauche avec les gains) ; pourquoi A reste à 1,5 % du modèle.
3. Revoir les tours 59–72 où le facteur de momentum (cJ/(1−b1), ~13 %) a pu être confondu avec autre chose (« taux 16–20 % trop haut »).
4. Où les salves atteignent leur bord de stabilité (le mode raide de u est stable sur le séjour avec v figé).

## Tour 73 (vraie critique de dipankarsarkar, 19/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE73.md` (gitignorée), partie, figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 19/10/2026 (tour 73, PAS simulée) », avant « ## 9. ».

**Sa critique** : « oui » sur le taux (valeur propre du pas qu'Adam fait, préconditionneur qui bouge pendant le séjour) ; mon écart de 12–15 % venait de la fenêtre
d'ajustement (pente 1e-7 à 1e-4 : −130,8, taux 0,01760 contre ln 1,01757 = 0,01742, 1 %) ; décades sous lr\* : 67, 133, 132, 127 ; au-dessus : 133, 129, 124 ; prédiction : δ = 1e-8
coûte moins de 124 pas.
**Acquis (prédictions poussées avant les runs)** : tous ses chiffres ✓ ; lr\* à 1e-15 = 0,314002321343052 ; décades au-dessus (tol 0,02) 133, 129, 124, **119**, 114, 111 jusqu'à δ = 1e-10
(sa prédiction tenue) ; sous lr\* (tol 0,005) 133, 132, 127, 121, 116, 113 ; jacobien en fin de séjour (δ = 1e-10) 1,01855 (500), 1,01987 (700), 1,02195 (850), 1,02670 (920) ; √v_e3 décroît en
exp(−t/2000) (1985 mesuré) ; modèle gain × λ_s : forme bonne, taux 16–20 % trop haut (décades 104, 101, 97, 93 contre 124, 119, 114, 111, rapport 0,84–0,86 constant).
**Mes ratés** : modèle à 5 % (3 sur 4 hors), lead ≥ 1,0200 à t = 700 (1,01987), monotonie de la série sous lr\* à tol 0,02 (une excursion de 4641).
**Limites** : réduction seule, un code, l'origine du facteur 0,84 inconnue (correction en x de λ_s ? terme 26 eps ?), réseau complet non testé.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 73 ; question : le 0,84 est-il la correction d'ordre x de λ_s à x = 1e-6 ? (oui/non).
2. Calculer λ_s(x) exact au col (jacobien de la réduction à x = 1e-6, avec point fixe g_l3 = g_l4) ; vérifier le vecteur propre de 1,0177.
3. Refaire x = 1e-5 avec le col de la réduction au lieu de la forme fermée.

## Tour 72 (vraie critique de dipankarsarkar, 18/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE72.md` (gitignorée), partie, figée. La lettre 71
est partie (sa critique du tour 72 y répond) et reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 18/10/2026 (tour 72, PAS simulée) », avant « ## 9. ».

**Sa critique** : le taux de la loi log (125 pas/décade → 0,0184, multiplicateur 1,0186) est celui d'un col ; mais le col appartient à
(x, lr\*(x)), un balayage à un x ne les sépare pas ; deux vérifications : (1) la loi des deux côtés (mes 17 pas ne comptent que rb > 0,9) ; (2) la valeur propre
de l'application d'un pas à mi-séjour (1,0186 attendu). Question : sous lr\*, le temps de retour croît-il comme 125 log10(1/δ) ?
**Acquis (prédictions poussées avant les runs)** : son arithmétique ✓ ; mon horloge (rb) était aveugle ; **le séjour est sur le col (jumeau) de la forme fermée :
d3 = 2,837195e-3 = nœud + 0,2213√x à x = 1e-6** ; départ du col : −122 pas/décade au-dessus, −130 (tol 0,02) / −113 (tol 0,005) au-dessous : **oui** ; valeur
propre 1,0177 (t = 350), 1,0172 (mi-séjour) : taux 4–8 % sous 1,0186, dérive le long du séjour 1,0169 → 1,0197 ≈ gain Adam ×1,12 ; pentes : peu de q (−14 %, −24 %),
beaucoup de x (×2,4 à x ÷ 10, ×0,27 à x × 10) ; **formule : taux = lr\* λ_s(x) [1/(s+eps) + 1/(s+26 eps)], λ_s = 7,30e-5√x, dans 0,83–1,07 du taux mesuré sur quatre
configurations**. **Mes ratés** : métrique rb de P72-1 ; q = 9e-7 à −24 % (limite 20) ; valeur propre à 15–19 % de la formule (limite 15) ; ma bissection sur [0,2 ; 0,45] a échoué à q = 5,95e-7
(plafond non monotone).
**Limites** : réduction seule, un code ; x = 1e-5 non testé (séjour décalé) ; vecteur propre de 1,0177 non vérifié ; l'état de base vient de lr 0,05.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 72 ; question : le taux du col est-il gain Adam × λ_s(x) ? (oui/non).
2. x = 1e-5 : définir le col dans la réduction (point fixe g_l3 = g_l4 + z4 du run) au lieu de la forme fermée ; vérifier le vecteur propre.
3. Reprendre les conclusions de lr 0,1–0,3 (tours 69–71) en tenant compte que le démarrage passe par le col.

## Tour 71 (vraie critique de dipankarsarkar, 17/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE71.md` (gitignorée), partie, figée. La lettre 70
est partie (sa critique du tour 71 y répond) et reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 17/10/2026 (tour 71, PAS simulée) », avant « ## 9. ».

**Sa critique** : le bord de stabilité sur u ne met pas de plafond à 0,35 (S = 0,92 de 38 à lr 0,3, atteint 38 vers 0,9 ; écart/sd
2,4 → 1,6) ; 0/16 → 16/16 est une marche ; deux formes (pli en lr, crise H11) ; test : le temps d'effondrement croît-il en
1/√(lr − lr_c) entre 0,31 et 0,35 ?
**Acquis (prédictions poussées avant les runs)** : ses chiffres ✓ (S → 38 à 0,86, écart/sd → 0 à 0,52 ; mon « 4,9 » à x = 1e-7 était 5,2) ;
**le plafond du tour 70 est un effet de démarrage** : l'état de base a ses moments de lr 0,05, lr 0,35 reçoit un premier pas ~6× trop grand,
l'effondrement est installé dans les 100–700 premiers pas ; **lr* = 0,314002321** (état de base brut) ; temps d'installation au-dessus
logarithmique (−125 pas par décade de lr − lr*, e-fold 54 pas), pas en 1/√ ; temps de retour en dessous constant (17 pas) ; **avec m × lr/0,05
et v × (lr/0,05)² : aucun effondrement de lr 0,3 à 2,0** (lr 3,0 s'effondre au pas 4611) ; zéros mis à l'échelle : lr 0,4 : 5,8870 ; lr 0,5 :
5,8165 ; collapse à eps/lr = 6,0e-7 pour lr ≥ 0,4 ; la série paire du tour 70 ne s'extrapole pas (5,955 prédit à lr 0,4, 5,887 mesuré ; lr³ :
5,896) : parité ouverte. **Mes ratés** : P71-3, P71-5, P71-7 (0,003), P71-8, moitié de P71-1.
**Limites** : réduction seule, un code, la loi logarithmique n'a pas d'origine établie, mécanisme de lr 3,0 inconnu, les résultats des
tours 59–70 à lr ≠ 0,05 démarraient tous d'un état de base fait à lr 0,05 (la chauffe de 20 000 pas absorbe le transitoire sauf s'il
franchit le bassin : à vérifier pour lr 0,1 et 0,2).

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 71 ; question : le taux de la loi log est-il le taux instable du col, fonction de x et pas de lr ? (oui/non).
2. Refaire les zéros de lr 0,1 à 0,3 avec moments mis à l'échelle (cohérence avec le tour 70) ; la parité à 0,4–0,5 (nouvelles séries).
3. Regarder le taux 0,0184/pas contre la valeur propre instable du col de la réduction à x = 1e-6.

## Tour 70 (vraie critique de dipankarsarkar, 16/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE70.md` (gitignorée), partie, figée. La lettre 69
est partie (sa critique du tour 70 y répond) et reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 16/10/2026 (tour 70, PAS simulée) », avant « ## 9. ».

**Sa critique** : lr 0,2 seul point > 0,1 : c de a + b lr² + c lr⁴ = −19,6 (0,01–0,1), +17,1 (0,01–0,2), +13,8
(0,005–0,2) ; d lr³ ajuste aussi (d = +5,1) ; x confondant (c = 38 à 1e-7, 19 à 1e-8) ; deux runs : lr 0,3 (lr⁴ : 5,976 ;
lr³ : 5,944 ; pur : 5,853) et lr 0,15 (6,142 contre 6,137) ; pari 55 % pour la série paire ; question : lr 0,3 hors effondrement ?
**Acquis (prédictions poussées avant les runs)** : tous ses ajustements reproduits ; lr 0,15 = 6,1413 (x = 1e-8), 6,1383
(1e-7) ; **lr 0,3 = 5,9721 (1e-8), 5,9579 (1e-7), 0/16 effondrées : son pari tenu** ; série paire : c = +16,50 sur 0,01–0,3 (résidu
max 0,0006 contre 0,0016 pour lr³ et 0,0212 pour le quadratique pur) ; c est de même valeur à x = 1e-7 (15,9, cinq points) : son
c = 38 venait d'un lr 0,1 bimodal ; **plafond entre lr 0,3 et 0,35 à eps/lr 6e-7 : lr 0,35 et 0,4 s'effondrent à 16/16**
(marche, cause inconnue). **Mes ratés** : lr 0,4 et 0,35 à x = 1e-7 (je prédisais un zéro).
**Limites** : réduction seule, un code, la parité tranchée sur un jeu de sept zéros, le plafond non expliqué, l'absence du
terme impair non dérivée.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 70 ; question : son image du bord de stabilité sur u prédit-elle le plafond 0,3–0,35 ? (oui/non).
2. Encadrer le plafond (lr 0,32 ; 0,33) et regarder ce qui change (S, sd, gap/sd, bimodalité) juste avant.
3. Le même coefficient c dans le réseau complet (hors portée : runs longs).

## Tour 69 (vraie critique de dipankarsarkar, 15/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE69.md` (gitignorée), partie, figée. La lettre 68
est partie (sa critique du tour 69 y répond) et reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 15/10/2026 (tour 69, PAS simulée) », avant « ## 9. ».

**Sa critique** : le zéro de D_d est libre de lr dans G(eps/lr) seul, mais mes cinq zéros choisissent la
correction : zéro = 6,230e-7 (1 − 0,63 lr²) (ajustement, p = 2 ; lr⁴ dans D_d, pas de lr³) ; lr 0,005 rate de 0,010 (run non
réglé, selon lui) ; prédictions : lr 0,2 à 6,07 (80 %), runs longs à lr 0,005 remontent vers 6,23 (65 %) ; question :
lr 0,2 est-il stable ? Le tableau de la queue vivante est « le vrai résultat » du tour 68.
**Acquis (prédictions poussées avant les runs)** : son tableau reproduit au chiffre ; zéros affinés (pas 1e-9) :
6,2234 / 6,2295 / 6,2302 / 6,2214 / 6,1896 (lr 0,005 → 0,1) ; **lr 0,2 stable, zéro 6,0838e-7 (x = 1e-8) et 6,0752e-7
(x = 1e-7), sa prédiction tenue à 0,3 %** ; run long à lr 0,005 : le zéro descend à 6,2142 (il prédisait une
remontée) ; **cause : dérive de z4 (d4 ~ 2e-12 dans la force à côté de g_l = 3,5e-13), z4 moyen du run 30,056 (court) /
30,117 (long) contre 30,0231 à la base ; avec la référence à z4 du run : lr 0,005 court 6,2285, long 6,2287 ; p = 2
résidu max 0,0005 (quatre points), 0,0022 (cinq), p libre 2,07** ; lr 0,2 à +0,017 au-dessus du quadratique.
**Mes ratés** : P69-2 (6,220 ± 0,003 : 6,2142 long), P69-3 (lr 0,2 s'effondre : non), P69-7 (écart des zéros < 0,025 :
0,041), P69-8 (p ≠ 2 : p = 2,07).
**Limites** : réduction seule, un code, pas de dérivation de l'absence du terme impair, le z4 du réseau complet n'est
pas la dérive de la réduction, l'excès à lr 0,2 testé sur un seul lr.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 69 ; question : l'excès à lr 0,2 est-il le terme suivant c lr⁴ avec c positif ? (oui/non).
2. lr 0,3 / 0,4 pour contraindre c, si stables (collapse à surveiller).
3. Refaire les tours 59–66 sensibles au gel de la queue et à z4 avec la référence « z4 du run » (D0 corrigé).

## Tour 68 (vraie critique de dipankarsarkar, 15/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE68.md` (gitignorée), partie, figée. La lettre 67
est partie (sa critique du tour 68 y répond) et reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 15/10/2026 (tour 68, PAS simulée) », avant « ## 9. ».

**Sa critique** : D0 n'est ni dans le gradient ni dans Adam, il est dans la condition de point fixe : u est
immobile à g_l3 = g_l4 (pas g_l3 = 0), donc la force de queue gelée +6,99e-13 est partagée en deux,
g_l3 = g_l4 = +3,495e-13, D0 = (g_queue/2)/λ ; le pli suit la condition (g_l3 = g_l4 : PLI − 1,9e-11) donc la
réduction n'a pas de nœud sous x = 1,9e-11 ; question : somme de g sur les 25 références de la ligne 10 en fin
de phase 1 ?
**Acquis (prédictions poussées avant les runs)** : tous ses nombres reproduits à 4 chiffres (racines 60 chiffres,
tangence PLI + 1,2171e-11 / PLI − 1,8691e-11) ; D_d − 10,31e-9 s'annule à 6,19–6,23e-7 pour lr 0,005 → 0,1 ; état
réglé : m4 = m5 = +3,495e-13, dérive de l3 + l4 conforme, d3 à 2e-12 de la racine ; échappements à eps 1e-6 :
règle à x ≥ 1,9e-11, s'effondre à ≤ 1,5e-11. **Checkpoint de phase 1 : somme de la queue = 6,9586e-13 contre
6,9828e-13 gelé (0,35 %). Mais dans le réseau complet à eps 1e-10 la queue n'est PAS gelée : 1/(1 + 1,4e-5 t),
moitié à ~71 000 pas, 40,7 % à 1e5, donc D0 et le pli de la réduction dérivent ; à eps 1e-6 gel exact.** Les états
des tours 59–66 à eps 1e-10 et x = 1e-13 sont sans nœud dans la réduction (tenus par les salves).
**Limites** : réseau complet à eps ~3e-8 (zéro de D_d) non testé ; G inconnue ; dérive de 0,5 % du zéro à lr 0,1
inexpliquée.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 68 ; question : sa loi prédit-elle la chute de 0,5 % du zéro à lr 0,1 ? (oui/non).
2. Rendre la queue vivante dans la réduction (une coordonnée de plus avec Adam à eps) et revalider contre le
   réseau complet à eps 1e-10 (D_c, sd, D_d) ; ou valider à eps 3e-8 où le gel tient.
3. Revoir les conclusions des tours 59–66 sensibles à D0 et à la position du pli à eps 1e-10 (D_d, tilt, seuil).

## Tour 67 (vraie critique de dipankarsarkar, 14/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE67.md` (gitignorée), partie, figée. La lettre 66
est partie (sa critique du tour 67 y répond) et reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 14/10/2026 (tour 67, PAS simulée) », avant « ## 9. ».

**Sa critique** : la raison est dans mon code : `g[1] = -ge3/C`, donc le pas sur z3 est
lr m0 [1/(s+eps) + 1/(s+26 eps)] et D_d = lr² G(eps/lr)/λ, zéro de D_d à eps/lr fixe indépendant de lr ;
3 prédictions (zéro à 5 % de 6,15e-7 pour lr 0,01 et 0,1 ; |D_d| lr 0,1/0,05 dans [3,6 ; 4,8] ;
eps[1] = eps[0]/26 déplace le zéro, a priori 40 %). Il ne pouvait pas me rejouer (état de base hors dépôt).
**Acquis (prédictions poussées avant les runs)** : sa lecture du code exacte (1e-16, 1e-11/1e-13) ;
état de base exporté en JSON dans le dépôt (`etat_base_reduction_mur23.json`) ; pente de D_d en eps/lr ∝ lr²
à 5 % ; zéros 5,757 / 6,104 / 6,199 / 6,186e-7 (lr 0,01 / 0,02 / 0,05 / 0,1), sa prédiction ✓ sauf lr 0,01 ;
**D_d contient un terme lr-indépendant D0 ≈ +1,0 à +1,6e-8 à x = 1e-8 (∝ 1/√x, présent sans salves, pas un
transitoire, pas une erreur de référence : le point fixe propre est SOUS le nœud fermé de 6,06e-13/√x)** ;
avec D0 = +1,0e-8 retiré le zéro est fixe à 1,1 % ; eps[1] = eps/26 supprime le zéro (D_d < 0 partout,
sd ×4) ; lr 0,1 s'effondre à eps/lr 1e-6, x = 1e-8. **Mes ratés** : P67-2 (test impossible), ordre monotone
du zéro, D0 = erreur de référence (P67-8, P67-9), S inchangé sous eps[1].
**Limites** : réduction seule (o3 = moyenne des 26 concurrents avec un seul v ; le réseau complet en a 26),
un code, origine de D0 inconnue, G inconnue.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 67 ; question : D0 est-il une force résiduelle du code du gradient (oui/non).
2. Trouver D0 : résidu de gradient de la réduction (comparer ge3, ge4, g[4] aux dérivées exactes de
   l'objectif complet au point fixe ; Sed/Se constants dans la réduction contre l'objectif réel).
3. Tester l'asymétrie d'eps (o3) dans le réseau complet avec AdamMasque (26 coordonnées).
4. Séparer eps/lr fixe de r fixe (eps[1] ≠ eps[0] et √v_e3 déplacé).

## Tour 66 (vraie critique de dipankarsarkar, 13/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE66.md` (gitignorée), partie, figée. La lettre 65
est partie (sa critique du tour 66 y répond) et reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 13/10/2026 (tour 66, PAS simulée) », avant « ## 9. ».

**Sa critique** : symétrie exacte du pli (lr, eps, x, d) → (c lr, c² eps, c² x, c d) ; √v_e3 ∝ lr et
l'ordre en x contredisent un pli 1-D ; la coordonnée des salves voit une force linéaire indépendante
de x. Test : lr 0,02, eps et x × 0,16 ; a priori 25 % que √v lise 6,25. Question : quelle coordonnée
porte le pli ?
**Acquis (prédictions poussées avant chaque run)** : symétrie vérifiée sur la forme normale (5e-16),
ses 54 % / 58 % non reproductibles sans ses paramètres. **Jacobien de la réduction : pli = mode mou
sur z3 (99,1 %), ∝ √x ; mode raide sur u (99,1 %), J_uu = −1,12e-4 indépendant de x.** Paire du pli :
√v 1,89–2,45 (pas 6,25), sd, |D_d| et coupure faux, tilt inchangé. **Les salves = bord de stabilité
d'Adam sur u** : √v_e3/√v_l3 = 0,1329–0,1339 = J_zu/|J_uu| (0,1341) ; S = lr·2|J_uu|/(√v_l3+eps) =
31,5–35,2 pour 38 ; à β1 = 0,8 S = 16,0 (seuil 18), à 0,95 S = 75,3 (seuil 78). Signe de D_d : fixé par
eps/lr (zéro à 6,20e-7 et 6,11e-7 à x = 1e-8 = seuil d'effondrement à x = 1e-13) ; écart nœud–col
0,2213 √x. **Mes ratés** : P66-1 (A, C sous 2,2), P66-7 (τ ne fixe pas le signe), P66-9 (critère
écart < k·sd faux hors de l'eps/lr où il a été lu).
**Limites** : réduction seule, un code, S/seuil ≠ 1 non dérivé, mécanisme du déplacement non dérivé,
effondrements à β1 0,8 / 0,95 non expliqués.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 66 ; question : le zéro de D_d est-il à eps/lr fixe indépendant de lr dans son
   image du pli ? (oui/non).
2. Mesurer D_d au x de la coupure pour un critère d'effondrement qui inclut D_d ; dériver S/38 ;
   vérifier le bord de stabilité sur u dans le réseau complet (√v des logits récepteurs).

## Tour 65 (vraie critique de dipankarsarkar, 13/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE65.md` (gitignorée), partie, figée. La lettre 64
est partie (sa critique du tour 65 y répond) et reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 13/10/2026 (tour 65, PAS simulée) », avant « ## 9. ».

**Sa critique** : « oui sur la tendance, non sur la raison » : invariance d'échelle d'Adam (lr, eps,
déplacement) → c·(…) avec gradient linéaire ; √v ∝ lr en est la trace ; à eps/lr égal tous les
rapports valent 2,5 et L est identique ; prédit que l'effondrement à x = 1e-13 est au même eps/lr
(eps_on(0,02) = 0,4·eps_on(0,05), 60 %, ±20 %). **Tous ses chiffres vérifiés** ; symétrie exacte sur
jouet 2D (5e-16), cassée par un terme quadratique. Ses « cinq paires » ne sont pas à eps/lr égal
(1,04 à 2,5×) : mesuré en vraies paires à eps/lr égal.
**Acquis (prédictions poussées avant les runs)** : seuil d'effondrement au même eps/lr à x = 1e-13 et
1e-11 (6,0–6,3e-7 pour les deux lr : il avait raison, mon intervalle 0,5–1 est manqué) ; à x = 1e-9 faux
(lr 0,05 : 7e-7 ; lr 0,02 : aucun jusqu'à 5e-6) ; pas d'état r > 1 à x = 1e-13 (r max 0,78–0,88), mais
nombreux à x ≥ 1e-8 (r = 1,36 / 3,43 / 6,92, 0/16 effondrées) ; à eps/lr égal et r > 1 : sd ratio
2,9–3,9, |D_d| (nœud) ratio 7,0–7,7 (x fixé) ou 4,1–4,5 (x mis à l'échelle), pas 2,5 ; le tilt
D_r/D_d (nœud) est le même pour les deux lr à 1–3 % ; paires à x = 1e-13 : sd 2,56 et |D_d| 2,504 seulement
au sommet (eps/lr 5,2e-7), tilt faux d'un facteur 2,3, sd 3,4–3,9 en dessous ; coupure en x à eps/lr 8e-7 :
rapport 0,10–0,20 (pas 0,4), POST HOC compatible avec lr². **Mes ratés** : P65-4, 5, 6, 8, 9, 11, 12, 15 ;
P65-3 à moitié.
**Limites** : réduction seule, un code (mur 23), un seul eps/lr pour la coupure en x, mécanisme non
dérivé.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 65 ; attendre sa réponse (question : coupure en x ∝ lr² ? oui/non).
2. Pourquoi sd ∝ lr^1,2–1,5 à eps/lr égal pour r > 1 alors que √v ∝ lr exactement ; pourquoi
   |D_d| ∝ lr² à x fixé ; autre eps/lr pour x_c ; réseau complet.

## Tour 64 (vraie critique de dipankarsarkar, 08/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE64.md` (gitignorée), PRÊTE À ENVOYER, première
personne. La lettre 63 est partie (sa critique du tour 64 y répond) et reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 08/10/2026 (tour 64, PAS simulée) »,
avant « ## 9. ».

**Sa critique** : à ma question « L = F(eps/√v_ligne3) seul ? » il répond NON (deux boutons : r
et le gain lr/√v ; à lr 0,02 L baisse (r monte) et |D_d| baisse (r baisse) : contradictoire) et
demande le rapport des médianes de √v sur (e3, o3) entre (lr 0,05, eps 1e-9) et (lr 0,02, eps
1e-10) : r seul exige > 10. **Tous ses chiffres vérifiés** (eps équivalent 1,09e-9, 5,0e-9, facteur
10,9, q = 2,32 ; ses L prédits 0,96708 / 0,96632 contre mes 0,96719 / 0,96639, écarts d'interpolation).
**Acquis (prédictions poussées avant les runs)** : R = 2,436 (< 10 : r seul rejeté, 4,3 à 4,7× trop
petit) ; √v ∝ lr (√v/lr = 8,2 à 9,5e-7 sur 10 états, gain 1,05 à 1,23e6, 16 %) : ses deux boutons
n'en font presque qu'un ; à r égal ΔL jusqu'à +0,0158, Δ|D_d| jusqu'à −44 % ; lr 0,05 à r = 0,64
(eps 2,6e-8) donne L = 0,9724, pas ≈ 0,987 ; ni sd ni eps/lr ne rangent L. **Observation POST HOC**
: |D_d| ∝ sd^a avec a = 0,03 / 0,15 / 0,29 / 0,52 / 0,88 pour r = 2e-3 … 0,64 (son « ∝ sd » est vrai
à r ~ 0,6, faux à r ~ 2e-3). **Mes ratés** : fenêtre R ∈ [2,5 ; 9] manquée par le bas (2,44) ; paire
« à r 20 % » la plus proche à 1,21.
**Limites** : réduction seule (validée au réseau complet à eps 1e-10, lr 0,05) ; un seul code ; aucun
état à r > 1 ; mécanisme non dérivé.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 64 ; attendre sa réponse.
2. Ma question à lui : a(r) atteint-il 1 et y reste-t-il pour r > 1 ? (lr 0,02, eps 3e-8 donne
   r ≈ 1,8 ; attention : le seuil Δ(eps) peut faire s'effondrer l'état à x = 1e-13).
3. Un deuxième code (c814, c7771) pour la loi a(r), et le réseau complet à lr 0,02.
4. Pourquoi √v ∝ lr à ~16 % (bord de stabilité adaptatif, tour 58) : dérivation non faite.

## Tour 63 (vraie critique de dipankarsarkar, 08/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE63.md` (gitignorée), PRÊTE À ENVOYER, première
personne. La lettre 62 est partie (sa critique du tour 63 y répond) et reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 08/10/2026 (tour 63, PAS simulée) »,
avant « ## 9. ».

**Sa critique** : il reproduit mes ajustements et ma table D_c, répond « oui » à ma question
(l'inclinaison de 3,2 % ∝ sd) et prédit à eps 1e-8 : D_d 3× plus petit (limite du rapport ≈
0,990) ou 9× (inclinaison immobile) ; il note qu'eps est un confus. Son « 0,47 sd » reprend le
sd NON saturé (1,3e-5) ; saturé (2,2e-5), D_d = 0,277 sd (même erreur que le 0,63 du tour 61).
**Acquis (vérifiés, prédictions poussées avant les runs)** : à eps 1e-8 sd saturé ÷ 3,45 (6,37e-6)
mais |D_d| × 1,42 (8,63e-6) et L = 0,9499 (inclinaison 3,2 → 5,0 %) : NI l'une NI l'autre de ses
branches. À eps fixe (lr 0,02) : sd ÷ 3,6, D_d −4 %, L 0,9654 : le déplacement et l'inclinaison
ne sont PAS pilotés par l'amplitude des salves. À eps 1e-6 (sans salves) la médiane est sur le
nœud (L = 0,9998). Par coordonnée (réduction) : eps 1e-8 sur la ligne 3 seule reproduit tout
eps 1e-8 (D_d −8,74e-6, L 0,9494, sd 6,27e-6), sur le récepteur seul reproduit eps 1e-10 :
le déplacement, l'inclinaison et l'amplitude des salves suivent le gain d'Adam de la ligne 3.
**Mon erreur corrigée** : l'argument de la lettre 61 (« pas le nœud d'un δ décalé car rapport
δ' ≠ 1 ») était invalide hors réponse linéaire ; le bon critère (courbe des nœuds à même D_d)
donne un écart de 3,3 à 5,2 % : conclusion maintenue, argument corrigé.
**Limites** : tout hors eps 1e-10 vient de la réduction (validée au réseau complet à eps 1e-10
seulement) ; réduction instable à eps 1e-6 et x ≤ 1e-11 (cause non élucidée) ; aucune
dérivation de D_d ni de l'inclinaison.

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 63 ; attendre sa réponse.
2. Ma question à lui : l'inclinaison est-elle fonction de eps/√v_ligne3 seul ? Test : lr 0,035 ×
   eps ajusté, ou eps 3e-10 / 3e-9 à lr 0,02 (modules à lr différent : `reduction_numba_tour63_lr002.py`).
3. Les masques d'eps sur le réseau complet (E3 du tour 59 a été fait sur le réseau complet pour
   le seuil, pas pour D_d / l'inclinaison).
4. Pourquoi la réduction s'effondre à eps 1e-6 et x ≤ 1e-11 (non élucidé).

## Tour 62 (vraie critique de dipankarsarkar, 07/10/2026) — LETTRE PARTIE

**Lettre** : `docs/REPONSE_ORDRE62.md` (gitignorée), PRÊTE À ENVOYER, première
personne. Les lettres 60 et 61 sont DÉJÀ ENVOYÉES (Théo, 07/10 : sa critique du tour 62
répond à la 61, celle du tour 61 répondait à la 60) et restent figées. **Carnet** : section « VRAIE
CRITIQUE DE DIPANKARSARKAR, 07/10/2026 (tour 62, PAS simulée) », avant « ## 9. ».

**Sa critique** : il reconnaît son 0,63 ; pas de 1 % sans Σ ; ajustements 1 - rapport =
c0 + c1 x^q sur mes six lignes (q libre 0,25 / limite 0,970 ; q = 0,47 / limite 0,967) ;
question : à pli-1e-13 le rapport est-il plus près de 0,9687 ou de 0,9668 ?
**Acquis (vérifiés, prédictions poussées avant les runs)** : réponse = NI L'UN NI L'AUTRE :
0,96782 ± 0,00001 ; ses lignes sont mal étiquetées (0,955…0,967 = pli-3e-10…-1e-12) et ses
ajustements ne se reproduisent pas (q = 0,564 / limite 0,9679 ; sa prédiction 0,9668 est
INFÉRIEURE à la valeur déjà mesurée à -1e-12, 0,9674 : arrondi à 3 chiffres). Seul modèle qui
tient sur 11 lignes : c0 + c1√x + c2x (χ² 12,2/8), limite 0,96799. Mécanisme confirmé :
déplacement des médianes constant (D_d = -6,081e-6, D_r = -4,391e-5 à 0,1 % pour x ≤ 1e-11 ;
D_r/D_d = 7,2210), 2 constantes + fermeture exacte du nœud prédisent le rapport à 1e-5-4e-5
près ; limite = (D_r/D_d)/ρ = 7,2210/7,459415 = 0,96804 (ρ forme fermée du mode mou).
Le rapport de la réduction = celui du réseau complet à ≤ 1 ES (jusqu'à pli-1e-10).
**Mes erreurs** : test du pli dynamique par échappements (mal conçu) ; prédiction fausse que le
départ de d4 change s de > 10 % (0,03 %) ; un premier ajustement non convergé (borne).

**Pistes concrètes pour la suite :**
1. Envoyer la lettre 62 (seule pas encore partie) ; attendre sa réponse.
2. Sa question de retour (ma question à lui) : l'inclinaison de 3,2 % entre D_r/D_d et ρ
   diminue-t-elle quand les salves sont plus petites (eps 1e-8) ? Je peux le tester dans
   la réduction en passant `eps` à `warmed`/`avancer_obs` (non fait).
3. Σ et le vecteur rigide pour DÉRIVER 7,2210 : non faits.
4. Le mode conjoint (0,79 / 0,82) contre les médianes (0,968) : cause non élucidée.

## Tour 61 (vraie critique de dipankarsarkar, 29/09/2026, traitée le 07/10) — LETTRE PRÊTE

**Lettre** : `docs/REPONSE_ORDRE61.md` (gitignorée), PRÊTE À ENVOYER, première
personne. La lettre 60 reste figée (et contient l'erreur « sd(d3) = 1,27e-5 » corrigée
dans la 61). **Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 29/09/2026
(tour 61, PAS simulée) », avant « ## 9. ».

**Sa critique** : biais des salves = distance FIXE en d3, mais à l'OPPOSÉ du jumeau (ma
formulation « vers » était fausse) ; décalage = mean-median (fixe, -7,99e-6) + median-nœud
(grandit, 39 % à pli-1e-10) ; ajustement B - c x^0,47 ; 2 questions (a) plateau sous
pli-1e-10 ? (b) médiane déplacée = nœud de l'objectif moyenné plutôt que d'un δ décalé ?
**Acquis (vérifiés, prédictions poussées avant les runs)** : sa table se retrouve à ≤ 1 % ;
(a) OUI, mean-median = -7,98e-6 plat à 3 chiffres jusqu'à pli-1e-12 (réduction numba) ;
son ajustement retrouvé (B = 1,4075e-5, q = 0,4703, lignes intermédiaires à 0,01-0,03 %) ;
**son « 0,63 sd » est faux** (sd(d3) saturé = 2,2e-5, la queue vaut 0,363 sd ; c'est B/sd
= 0,64 qui vaut 0,63 par hasard) et la même erreur est dans MA lettre 60 ;
(b) le rapport δ'_r4/δ'_d3 des médianes = 0,961 / 0,967, dispersion 3e-4 entre phases :
l'écart à 1 est réel et converge vers ~0,968 ; le MODE conjoint donne 0,79 / 0,82 : l'état
calme n'est le nœud d'aucun δ décalé ; « nœud de l'objectif moyenné » : non calculé.
**Correction de mon propre énoncé** : d4 (ligne 4) n'est pas gelé, il dérive en 1/t sous le
plancher d'Adam (t^-0,856, réduction) ; p → 1/2 est invariant au pli effectif.
**Coupure de courant le 07/10 sans dégât** (dépôt, carnet, lettres intacts, aucun processus
orphelin).

**Pistes concrètes pour la suite :**
1. FAIT : la lettre 61 est partie (sa critique du tour 62 y répond).
2. Sa question de retour : le rapport 0,968 se déduit-il de la covariance des salves ?
   (calcul de second ordre non fait).
3. Nœud de l'objectif moyenné sur les salves : non calculé ; seul test qui trancherait (b).
4. d4 sur le réseau complet (dérive 1/t vue dans la réduction seulement).

## Tour 60 (vraie critique de dipankarsarkar, 29/09/2026) — LETTRE PRÊTE

**Lettre** : `docs/REPONSE_ORDRE60.md` (gitignorée), PRÊTE À ENVOYER, écrite
à la première personne, avec le compte rendu de ce qui a été fait en interne
pendant son absence (demande de Théo). La lettre 59 reste figée.
**Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR, 29/09/2026 (tour 60,
PAS simulée) », juste avant « ## 9. ».

**Sa critique** : il reconnaît son erreur de comptage (grille linéaire
ratant les racines effondrées) ; il convertit le biais mean-median des
salves en unités de δ (pente du nœud 16 / 54 / 552) et prédit des seuils ; il
note que le décalage converti à eps 1e-10 n'est pas stationnaire ; il
demande si à pli-1e-9 il continue de baisser vers mon « 6e-10 à 1e-9 ».
**Acquis (tous vérifiés, prédictions poussées avant les runs)** : ses
racines, pentes et conversions se retrouvent ; son « -2,0e-8 » à pli-1e-8
est une erreur reprise de MON carnet (-1,46e-8). Décalage converti à eps 1e-10
(6 phases, aucune échappée) : -4,98e-8 / -4,24e-8 / -2,78e-8 / -1,46e-8 /
-6,4e-9 / -3,9e-9 / -2,4e-9 à pli-1e-5 … pli-1e-10 : il baisse sans
rejoindre 1e-9 à pli-1e-9, l'exposant local monte de 0,07 à 0,45 (vers
1/2 : le décalage tend vers 0 au pli). Sa conversion prédit les seuils
hors échantillon à 5 % (eps 2e-8) et 9 % (5e-8). La lettre corrige la
lettre 59 (le « 6e-10 à 1e-9 » était le transitoire d'Adam neuf).

**Pistes concrètes pour la suite :**
1. FAIT : la lettre 60 est partie (sa critique du tour 61 y répond).
2. FAIT le 29/09 (demande de Théo « fais le », intégré à la lettre 60) : p_local
   0,451 / 0,472 / 0,484 / 0,491 / 0,494 à pli-1e-10 … -1e-12 (réduction numba,
   validée à 0,1-0,6 % contre le réseau complet, `verifier_tour60_exposant_reduction.py`) :
   p → 1/2. Le biais de d3 sature à ≈ 1,4e-5 (≈ sd des salves 1,27e-5), l'écart
   nœud-jumeau se contracte en √ ; la conversion en unités de δ vaut tant que
   |biais|/écart ≪ 1 (franchit 1 vers pli-3e-9).
3. Réseau complet sous pli+6,0e-9 : ÉCARTÉ par Théo (~20 h), ne pas lancer.
4. Forme de λ(δ) et exposant : non déterminés ; pas de mécanisme.
5. Conversion sur r4 (fait sur d3 seul) ; quatrième code ; lr > 0,05.

## Tour 59 (vraie critique de dipankarsarkar, lue le 27/09/2026) — CLOS (lettre 59 figée)

**Lettre** : `docs/REPONSE_ORDRE59.md` (gitignorée), prête à envoyer
(excuses pour le retard incluses, demande de Théo). **Carnet** : section
« VRAIE CRITIQUE DE DIPANKARSARKAR, 27/09/2026 (tour 59, PAS simulée) ».

**Acquis :** ses formes fermées (pli, table `δ_c(C)`) revérifiées à tous
les chiffres ; pli exact mpmath 0,0134372100660973. La collision 6/14
(12345 k=3) casse au même endroit que le mur 23, à 1e-9 près (objet, pas
code). Aucune salve ne « traverse d'abord » géométriquement : elles
dépassent le jumeau (3,9× en d3, 20× en r4) sans bascule. Sous 1e-8, le
seuil appartient à l'optimiseur : sans salves (eps 1e-6) pile au pli
(±3e-10), avec salves décalé de 1e-9 à 1e-8, signe = asymétrie des
salves (4/4), grandeur non expliquée. Son étiquette corrigée (2,47e-8 =
déficit, pas √v).

**LETTRE 59 FIGÉE (consigne de Théo, 28/09) : tout ce qui suit est
dans `CARNET.md` (sous-section « Après la lettre ») et ira dans la
lettre 60 quand dipankar reposte.**

**Acquis après la lettre (28/09, dipankar toujours silencieux) :**
- **Correction de la lettre 59 à porter :** le « (6e-10 ; 1e-9) » du
  seuil à eps 1e-10 était le transitoire (Adam neuf). Sur état chauffé,
  le seuil est **(6e-9 ; 7e-9]** pour le mur 23 ET pour la 6/14 dans
  les deux sens (objet, pas code), reproductible sur deux états chauffés.
- Loi du fantôme exacte `t = κ(eps)/√(δ-δ_c')`, t0 ≈ 0, R² = 1,0000 (6
  et 8 points) ; `δ_c' - pli` = +9,1e-9 (1e-8), +3,2e-10 (3e-8),
  -5,5e-9 (1e-7, reproduit à 1 pas près sur un autre état chauffé).
- Au-dessus du pli, jusqu'à +6e-9, l'état tient 60 000 pas SANS point
  fixe déterministe (médiane à 1-2,5e-6 du pli, pics ±1e-4) : point
  fixe moyenné par les salves.
- Zéro de Δ(eps) vers 3,1e-8 (Δ = +9,1 / +3,85 / +0,32 / -3,66 / -5,5e-9
  à eps 1e-8 / 2e-8 / 3e-8 / 5e-8 / 1e-7, prédictions d'interpolation
  tombées).

**Retour de l'agent style dipankar (29/09, critique SIMULÉE), vérifié
par mon propre code :**
- Ma « quasi-annulation Kapitza/Adam à 97 % » (H59-18) était FAUSSE :
  j'avais pris `u` (récepteur) comme variable lente. Le mode critique est
  à 99,1 % sur la ligne 3 de l'émetteur (Hessien du réseau complet, 2 × 729
  paramètres, `verifier_tour59_hessien_reseau_complet.py`). Le mode rigide
  (99 % récepteur) est esclave.
- Sa lecture, confirmée sur le RÉSEAU COMPLET par trois expériences
  précommises : **E1** (lr 0,035 : seuils +3,25e-9 / -2,10e-9 contre
  +3,3e-9 / -2,1e-9 prédits) ; **E2** (eps 3e-6 : bascule en 13 846 pas
  à pli+1e-7, prédit 13 846) ; **E3** (eps 1e-7 sur la seule ligne 3
  de l'émetteur : seuil -1,36e-8 contre -1,34e-8 ; sur le récepteur seul :
  falaise positive +2e-9 / +4e-9 contre +3,2e-9).
- **Sa formule de κ recalculée par moi sur les 2 × 729 paramètres** :
  `κ/eps = π/(lr √(αβ)) = 1,6422e6` (α = 1,538e-3, D³J = 1,903e-6) contre
  1,617e6 mesuré (E2). Ma droite `0,0711 + 1,389e6·eps` avait la
  mauvaise asymptote (12 % trop bas à 3e-6).
- Mécanisme du signe : Adam fixe l'axe des salves via ses gains sur la
  ligne 3 (`p_z = lr[1/(√v+eps) + 1/(√v+26 eps)]`) et le récepteur
  (`p_u = 2 lr/(√v_l+eps)`), le zéro de Δ est où l'axe croise la pente
  critique `s* = -0,337`.
- Le seuil DÉPEND de lr (`Δ ∝ lr^3,4`) : « objet, pas code » est vrai
  entre mur 23 et 6/14 à lr fixe, pas en lr.

**La falaise à eps 1e-10 (réglage par défaut du projet) n'est PAS un
seuil déterministe (E4-E8, réseau complet) :** hold/break non monotone
à 1e-10 de pas (casse à +6,6e-9, tient à +6,7 et +6,8, casse à +6,9 et
+7,0) ; à δ = +6,5e-9 une phase sur dix casse (pas 27 444) ; le même δ
donne 465 ou 1 110 pas selon l'arrondi ; eps 1e-12 ne change pas la
position ((6e-9 ; 7e-9]) ; tient 300 000 pas à +6e-9 ; β2 = 0,998 :
l'état s'effondre déjà à pli-1e-6 (bursts bien plus grands).
**Taux d'échappement à eps 1e-10 (CORRIGÉ le 29/09 après le second agent,
vérifié par V1-V4 dans le carnet) :** ce n'est PAS un seuil, et ma loi de
puissance (δ* = 5,7e-9, γ = 3,5) est RETIRÉE (artefact de fenêtre
d'ajustement). La réduction à deux lignes (port numba, 3,9e10 pas) donne
λ = 1,7e-9 (5,0e-9), 9e-9 (5,25), 4e-8 (5,5), 2e-7 (5,75), 6e-7 (6,0),
1,2e-6 (6,1), 2,8e-6 (6,3), 6,6e-6 (6,5), 2e-5 (6,8), 3,7e-5 (7,0), 1,1e-4
(7,5), 1,7e-4 (8), 4,5e-4 (10) par pas ; j'ai retrouvé 7,0 / 6,5 / 6,0 /
5,5 avec une autre graine. Le réseau complet est d'accord à δ ≥ 6,0e-9
(54 événements contre 48,8) et n'a jamais été mesuré en dessous. Ma ligne à
1e-8 (phases k ≤ 55, 409-490 pas) était biaisée ×5 par le plan
d'expérience (V1 : k ≥ 89 donne 423-3309 pas). Le « seuil » dépend de
l'horizon : λ = 1/T pour T = 6e4 / 1e6 / 1e9 pas donne pli + 6,75 / 6,07 /
5,00e-9. Forme de λ(δ) non déterminée (puissance γ = 12 sur [5;7], queue
gaussienne, Kramers 3/2 avec x_c = 7,4e-9 : ajustées sans rejet). Aucun
mécanisme (Lyapunov de l'agent : D_KY = 11,3, γ ≈ 9-10 par comptage naïf,
non recalculé). β2 : 0,998 → l'état s'effondre déjà à pli-1e-6 ; 0,9995 →
seuil +3 à 6e-9 (mon pari « au-dessus de 8e-9 » faux). Scripts de l'agent
gardés et rejoués : `reduction_deux_lignes_tour59.py`,
`reduction_numba_tour59.py`, `hasard_reduction_tour59.py`,
`verifier_tour59_reduction_controle.py`.

**Ajouts du 29/09 (soir) :** E3 refait sur la collision 6/14 (autre code,
graine 12345) : mêmes seuils que le mur 23 à ~5 % pour les trois masques
(tout : -5,70e-9 ; ligne 6 : -1,31e-8 ; récepteur : falaise +2 à +4e-9) et
mêmes temps de bascule à < 1 % : le décalage dépendant d'eps est une
propriété de l'objet à (N, β, lr) fixés. Temps d'attente exponentiels
sous ~7e-9 seulement (sd/moyenne 0,75-0,97), pas au-dessus (0,45-0,74) :
mes MLE à taux constant ne valaient que sous 7e-9. Frontière lisse/stochastique en eps : sd/moyenne des temps de bascule sur
10 phases à pli+1,5e-8 = 0,77 / 0,71 / 0,45 / 0,087 / 0,011 / 0,004 pour eps
1e-10 / 1e-9 / 3e-9 / 1e-8 / 3e-8 / 1e-7 : régime stochastique pour eps
≲ 3e-9 (dont le réglage par défaut 1e-10), déterministe (loi du fantôme)
pour eps ≥ 1e-8. La frontière dépend TRÈS fortement de lr (≈ 5e-9 à lr
0,05 ; ≈ 2e-9 à 0,035 ; < 1e-10 à 0,02, où eps 1e-10 donne cv 0,11) : la
lecture eps ≈ 0,1 √v avec √v ∝ lr est RÉFUTÉE (facteur ≥ 5 observé pour
lr ÷ 2,5). Ablation de l'agent confirmée sur le réseau complet : `e3msg`
seul (seuil ∈ (+9e-9 ; +1,2e-8]) et `o3` seul (∈ (+6e-9 ; +9e-9]) restent
POSITIFS, seule la ligne entière donne -1,36e-8. lr 0,05 (le taux de tout
le projet) est dans le régime stochastique à eps par défaut. Un TROISIÈME code
(77777 k=1, référents 5/20, message 14, cas `c7771`) donne aussi les
mêmes seuils (tout -5,67e-9, ligne 5 -1,45e-8) : « objet, pas code »
tient sur 3 codes pour le mécanisme dépendant d'eps. Théo a écarté le
test de 20 h (réseau complet sous 6,0e-9) : ne pas le lancer.

**RÈGLE DE CALCUL (Théo, 29/09) : jamais plus de 6 processus python à la
fois, un lanceur à la fois** (`NPROC` dans les lanceurs, défaut 6). À 6
processus les runs vont ~4× plus vite qu'à 22 sur 12 cœurs.

**Pistes concrètes pour la suite :**
1. Retour de l'agent ciblé (λ(δ) dans la réduction, forme de la loi,
   mécanisme de γ, artefacts possibles) : à vérifier sur le réseau complet.
2. Lettre 60 : reprendre, quand dipankar reposte, (a) la correction du
   « 6e-10 ; 1e-9 » (transitoire), (b) H59-18 fausse et la variable
   lente correcte, (c) κ, E1-E3, (d) la falaise = taux d'échappement.
3. E1-E3 sur la 6/14 (fait sur le mur 23 seul) : le décalage dépend-il du
   code par le gain d'Adam ?
4. Localiser la frontière entre le régime lisse (eps ≥ 1e-8, loi du
   fantôme) et le régime stochastique (eps 1e-10) : eps 1e-9, 3e-9.
5. Seuil à β2 = 0,998 (chauffer plus bas que pli-1e-6 puis balayer).

## Tour 58 (vraie critique de dipankarsarkar, 23/09/2026) — CLOS

**Lettre** : `docs/REPONSE_ORDRE58.md` (gitignorée), PRÊTE À ENVOYER
(corrections des deux agents style dipankar intégrées après vérification
indépendante de chacune) ; `REPONSE_ORDRE56.md` reste interdite de
modification (consigne de Théo). **Carnet** : section « VRAIE CRITIQUE DE DIPANKARSARKAR,
23/09/2026 (tour 58, PAS simulée) », juste avant « ## 9. ».

**Acquis du tour (tous avec prédictions poussées avant les runs) :**
- Bug du tour 54 : `sigmoid(logit brut)` au lieu de `s3`/`R4`. Vrai état
  à `delta=0` : `1-s3=3,5e-10`, récepteur à 50/50, vrai `R4=0,5`.
- Test aligné sur les événements (sa demande) : pente X/gap 0,937
  (réel) contre 4,75e-5 (`delta=0`), 15/15 événements — non nulle.
- Mécanisme : la saturation atténue à travers le plancher `eps` d'Adam,
  `pente=½[u/(u+eps)+(u/26)/(u/26+eps)]` ; ablation eps ligne 3 (5
  décades), balayage en delta (transition ~0,009), séparabilité exacte
  (`poids[4]` : effet nul). Le « canal par l'asymétrie » du tour 54 est
  réfuté. Le ×7,35 du tour 56 était l'extrémité 61000 de
  l'interpolation.
- La salve est un bord de stabilité adaptatif (seuil 38/lr, Cohen et al.
  2022) : mode couplé à delta réel (4,9 % ligne 3), pur à `delta=0`.
- Les « murs supplémentaires » du 21/09 n'en sont pas : le référent 5
  est ORPHELIN (optimum uniforme exact), 8/12 sont des synonymes
  (optimum ½ exact), tous tenus au bord de stabilité ;
  `ln K-⟨H⟩=½(lr/38-eps·NK/β)²`, `eps_c=3,61e-8` confirmé.

**Acquis après audit (deux agents, chaque chiffre clé revérifié) :** la
ligne 3 AVANCE la salve de 106 pas à delta réel (valeur propre du mode
couplé, `ΔS=pente·S_gap·κ`, linéaire en `s3(1-s3)`, sans eps) ; la
formule de pente n'est que la moitié vecteur propre, `u` est un proxy
(loi sous plancher `G·poids·s(1-s)·r3r4`) ; l'orphelin est un optimum
de BLOC dans un piège (`ΔJ=0,034082`) ; constantes `c_K` (c_2=1,080,
c_27=1,0116) ; collisions spontanées à 0,5/0,5 dans 12345 k=3 et
77777 k=1.

**Pistes concrètes pour la suite :**
1. Question de clôture posée à dipankar : `delta_c` est-il une propriété
   de l'objet collision ou du code ? Test précommis : collision 14/6 du
   message 8 de 12345 k=3 — casse à 0,01344 ± 1 % si propriété de l'objet.
2. Décalage additif ~10,5 pas dans le retard de timing (β2 0,999 et
   0,998) : canal de « germe », pas de forme fermée.
3. Forme fermée de `c_K` (le jouet donne les valeurs, pas la dérivation).
4. Non-monotonie en β2 d'une question antérieure (462/110/163) :
   possible artefact de seuil de détecteur selon l'agent, à revérifier.
5. Relire les tours 20-38 : lister les comparaisons qui enjambent le
   plancher eps (`√v/eps` des deux côtés) — les rapports entre deux
   lignes sous le plancher sont, eux, indépendants d'eps.
6. Refaire tout recensement de « lignes non effondrées » en couvrant
   AUSSI les lignes du récepteur (`verifier_autres_murs_systeme.py` ne
   voit aucune collision).

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
13. **[RÉPONDUE le 21/09/2026, CORRIGÉE le 21/09/2026 par un agent
    d'audit + vérification indépendante, voir `CARNET.md` section
    « Question 13 des 20 ».] L'arrêt anticipé est-il vulnérable au
    hasard de timing d'un cycle plancher-de-`v` ?** Réponse : OUI,
    confirmé empiriquement, et plus riche que ma première mesure ne le
    montrait. **Correction majeure** : ma grille de phase d'origine
    (pas=40) était trop grossière — même famille de biais que la
    question 3. À grille fine (pas=1, `verifier_arret_anticipe_grille_fine.py`),
    mon affirmation « la vulnérabilité CROÎT avec la patience »
    (`eps=1e-9`) ne survit PAS (ratio non monotone à grille fine) —
    c'était un artefact du RATIO, pas de la vraie dynamique. Ce qui
    survit et se révèle plus fort : à `eps=3e-9`, deux seuils NETS de
    `patience` (exactement à `465` et `529`) où le comportement d'arrêt
    bascule discrètement — pas une pente continue comme d'abord publié.
    Coïncidence notable non expliquée : `465` est quasi identique à la
    période du cycle de kick (~456-500 pas). La vraie vulnérabilité est
    un effet de queue (8-36% des phases selon `eps`), pas une fonction
    lisse de la patience.
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
