"""Cherche une trajectoire d'APPROCHE NATURELLE (sans placer R
artificiellement sur la variete predite par dipankarsarkar) qui passerait
pres du point selle fantome (s3=0,994300, R[10,4]=0,829390, delta=0,013,
mur referents 3/4 message 10 -- cf. CARNET.md, "point selle" H6) apres un
nombre significatif de pas, pour "rechauffer" Adam avant l'approche du col
(analogue au retard de ~600 pas de verifier_masse_fond_systeme_reel.py /
verifier_trajectoire_renversement.py).

RESULTAT (18/09/2026) : le col predit (0,9943 ; 0,8294) n'est approche par
AUCUNE trajectoire naturelle testee, mais un VRAI seuil d'effondrement
existe -- ailleurs, entre s3_init=0,97 et 0,98 (perturbation de s3 SEUL,
R naturel). Detail complet ci-dessous, y compris les deux pistes qui n'ont
rien donne (R seul perturbe ; delta proche de delta_c).

1) Etat de depart EXACT de construire_mur23() (avant tout fixer_*) :
   s[3,10] = 0.99999999966..., s[4,10] = 0.99999999999..., R[10,4] = 0.500009,
   R[10,3] = 0.499991 -- donc s3 est deja SATURE a 1.0, pas a 0.5. Ce n'est
   PAS le point de depart "s3=0,5" suppose par le round 1 de
   verifier_sonde_bassin.py.

2) Trace de la trajectoire NATURELLE (aucun fixer_s3/fixer_r4, delta=0,013,
   4000 pas, tous les 20 pas -- voir tracer_trajectoire_naturelle() ci-
   dessous) :
   - t=0 a ~700 : s3 reste sature a 1.000000 pendant que R[10,4] entre
     immediatement (des t=20) dans un cycle-2 d'Adam autour de 0,781-0,788
     (oscillation de pas, PAS une approche lente).
   - t=~740 a ~840 : s3 chute de 1.000000 a 0.998902 en ~100 pas -- chute
     RAPIDE, pas un ralentissement.
   - t>840 : re-stabilise dans un cycle-2 permanent, s3 oscille entre
     0.998990 et 0.999014, R[10,4] entre 0,7896 et 0,7956. Ce point
     (s3~0,999, R4~0,79) est a distance 0,0044 en s3 et 0,04 en R4 du point
     selle predit -- proche en ordre de grandeur mais jamais un
     ralentissement observable (aucun pas ou dR4/dt ou ds3/dt s'approche de
     zero pres de (0,9943 ; 0,8294) specifiquement).
   Le systeme file donc DIRECTEMENT vers la branche graduee sans jamais
   ralentir pres du col -- pas de "delay-then-veer" ici, contrairement au
   cas masse-de-fond sur le recepteur.

3) Balayage delta proche de delta_c=0,0134372 (0,0134 / 0,01343 / 0,013437,
   depart naturel = etat de construire_mur23() sans aucun fixer), 4000 pas
   chacun : meme comportement qualitatif, avec le point d'atterrissage qui
   se rapproche legerement du col a mesure que delta -> delta_c (s3_final
   0,997918 -> 0,997650 -> 0,997549 ; R4 oscillant 0,804-0,808 -> 0,808-
   0,814) mais reste a >0,003 en s3 et >0,015 en R4 du col, et la aussi
   sans ralentissement : le systeme atteint son cycle-2 stationnaire en
   <1200 pas dans les trois cas.

4) EXTENSION DE LA PISTE 1 (s3_init tres bas, R NATUREL, delta=0,013,
   40000 pas -- terminee apres le reste de ce fichier) :
     s3_init=0,98000 -> GRADUEE (s3_final=0,999000, R4=0,794022)
     s3_init=0,90000 -> EFFONDRE (s3_final=0,037161, R4=1,000000)
     s3_init=0,70/0,50/0,30/0,10/0,02/0,005/0,001 -> tous EFFONDRENT au
       meme point (s3_final~0,037, R4=1,000000)
   Il existe donc un vrai seuil, mais LOIN du col predit. Bissection
   ulterieure (4000 pas, suffisant d'apres le point 2 ci-dessus) :
     s3_init=0,97/0,96/0,95/0,94/0,93/0,92/0,91 -> TOUS EFFONDRENT
       (s3_final~0,037, R4~0,999998)
   Donc le seuil est resserre entre 0,97 (effondre) et 0,98 (gradue) --
   un intervalle etroit d'environ 0,01 de large, situe pres de s3=1 mais
   PAS a l'endroit predit par la forme fermee du col (0,9943 est DANS la
   zone qui effondre : 0,9943 < 0,97 est faux -- 0,9943 est entre 0,97 et
   0,98, donc DANS l'intervalle du seuil, mais le point "col" precis
   0,994300 lui-meme n'a pas ete teste directement ici avec R naturel
   -- seulement avec R fixe a 0,5 (construire_mur23 brut) ou a 0,829390
   (round 2 de verifier_sonde_bassin.py). A tester : cible=0,9943 pile
   avec R naturel, pour voir si ce point exact est sur la frontiere.

5) PISTE 2 (perturber R SEUL, s3 laisse NATUREL c-a-d sature a 1,0,
   delta=0,013, 40000 pas) -- ECHEC NET, pas de bifurcation :
     R4_init de 0,050 a 0,950 (dont 0,829390, 0,829400, exactement la
     valeur du col predit) -> TOUJOURS BRANCHE GRADUEE (s3_final~0,999,
     R4_final~0,794). Perturber R sans faire bouger s3 hors de sa
     saturation ne rapproche jamais du col -- confirme que le col n'est
     visitable qu'en bougeant s3 (coherent avec le point 4 : c'est
     precisement le mouvement de s3 loin de 1,0 qui declenche la bascule,
     pas celui de R).

CONCLUSION (mise a jour) : le seuil de bascule "naturel" (R non fixe a la
main) pour delta=0,013 se situe entre s3_init=0,97 et 0,98 -- proche mais
distinct du col ferme predit a 0,9943 (qui tombe dans cet intervalle).
Aucune des trajectoires testees jusqu'ici (item 2, item 3, item 5) ne
ralentit visiblement pres de (0,9943 ; 0,8294) ; il reste a verifier
directement si LE SEUIL 0,97-0,98 lui-meme montre un ralentissement --
prochain travail concret : narrower ce seuil par bissection fine (deja
commencee : 0,975/0,978/0,979/0,9795, non terminee dans cette session) PUIS
tracer s3(t)/R4(t) juste au-dessus et en-dessous, comme fait au point 2
pour le col, afin de voir si CE seuil-la (contrairement au col) montre un
authentique ralentissement -- c'est la piste la plus prometteuse restante
pour mesurer `a` via un Adam naturellement rechauffe.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import continuer_sous_prior, objectif_pondere, etat
from verifier_sonde_bassin import poids_delta, fixer_s3, fixer_r4

ADAM_EPS = 1e-10
DELTA = 0.013
DELTA_C = 0.0134372


def etat_initial_brut():
    """Imprime l'etat de construire_mur23() AVANT tout fixer_* -- la
    valeur de depart reelle que "l'approche naturelle" doit utiliser."""
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    with torch.no_grad():
        s_, r_ = e.loi(), r.loi()
        print(f"s[3,10]={s_[3,10].item():.12f}  s[4,10]={s_[4,10].item():.12f}")
        print(f"R[10,3]={(s_[3,10]*r_[10,3]).item():.6f}  "
              f"R[10,4]={(s_[4,10]*r_[10,4]).item():.6f}")
    return e, r


def tracer_trajectoire_naturelle(delta=DELTA, pas_total=4000, pas_bloc=20):
    """Trace s3(t), s4(t), R[10,4](t), R[10,3](t) depuis l'etat NATUREL de
    construire_mur23() (aucune perturbation), sous le prior asymetrique
    delta. Retourne la liste des points (t, s3, s4, R4, R3).

    CORRIGE le 18/09/2026 (bug trouve par un agent-dipankar) : la version
    d'origine appelait continuer_sous_prior() EN BOUCLE par blocs de
    pas_bloc pas -- or cette fonction reconstruit un torch.optim.Adam NEUF
    a CHAQUE appel (verifier_prior_asymetrique.py), donc l'optimiseur etait
    remis a zero tous les pas_bloc pas. Meme mecanisme deja diagnostique et
    refute ailleurs dans ce projet (Essai 3, §7.65 du carnet) pour un autre
    script, jamais applique a celui-ci avant cette correction. Optimiseur
    UNIQUE et continu ici (une seule instance Adam, construite une fois)."""
    poids = poids_delta(delta)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=ADAM_EPS)
    R, H, m, Hb, S = etat(e, r)
    trace = [(0, S[3], S[4], R[4], R[3])]
    for i in range(pas_total):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if (i + 1) % pas_bloc == 0:
            R, H, m, Hb, S = etat(e, r)
            trace.append((i + 1, S[3], S[4], R[4], R[3]))
    return trace


def balayage_delta_naturel(deltas, pas_total=4000, pas_bloc=200):
    """Pour chaque delta < delta_c, depart naturel (aucun fixer_*), imprime
    la trace grossiere (tous les pas_bloc) pour reperer un eventuel
    ralentissement pres du col a mesure que delta -> delta_c.

    CORRIGE le 18/09/2026 -- meme bug et meme correction que
    tracer_trajectoire_naturelle() ci-dessus (optimiseur unique continu,
    pas reconstruit par bloc)."""
    for delta in deltas:
        poids = poids_delta(delta)
        e, r = construire_mur23(adam_eps=ADAM_EPS)
        activer(e, r)
        opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=ADAM_EPS)
        print(f"--- delta={delta} ---")
        for i in range(pas_total):
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
            if (i + 1) % pas_bloc == 0:
                R, H, m, Hb, S = etat(e, r)
                print(f"  t={i + 1}  s3={S[3]:.6f}  R4={R[4]:.6f}")


if __name__ == "__main__":
    print("=== Etat initial brut (avant tout fixer_*) ===")
    etat_initial_brut()

    print("\n=== Trajectoire naturelle, delta=0.013, 4000 pas, tous les 20 ===")
    for t, s3, s4, R4, R3 in tracer_trajectoire_naturelle():
        if t <= 200 or t % 200 == 0 or (700 <= t <= 900):
            print(f"  t={t}  s3={s3:.6f}  s4={s4:.6f}  R4={R4:.6f}  R3={R3:.6f}")

    print("\n=== Balayage delta proche de delta_c, depart naturel ===")
    balayage_delta_naturel((0.0134, 0.01343, 0.013437))
