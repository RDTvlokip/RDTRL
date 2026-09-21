"""Tour 56 (21/09/2026, VRAIE critique de dipankarsarkar) : resout le
point laisse "non resolu" dans verifier_precommis_dipankar_sigma_row10.py
-- les 25 autres entrees de la ligne 10 du recepteur (hors r[10,3] et
r[10,4]) bougent 20x plus en somme absolue que r3/r4 individuellement
pendant le kick (0,222 contre ~0,011 chacun). Est-ce du bruit de
derive Adam de fond, ou un signal lie au kick ?

Trouve par un agent de verification (worktree isole, lance pour
challenger le resultat avant envoi a dipankarsarkar), revérifié
indépendamment ici : compare les 6 plus gros contributeurs (hors 3,4)
entre la config reelle (delta=0,013026615, kick present) et delta=0
(aucun kick). Memes referents dominants {9,5,18,22,8,10}, dans le
MEME ordre, aux DEUX configs, avec des valeurs LEGEREMENT PLUS
GRANDES a delta=0 -- signature d'un bruit de derive Adam de fond
partage par toute la ligne (meme compteur de pas, meme correction de
biais, meme eps), PAS d'un signal cause par le kick (si c'etait le
kick, les valeurs devraient differer nettement avec/sans lui).

Resultat integre dans docs/REPONSE_ORDRE56.md plutot que laisse en
suspens pour un tour futur.
"""

import sys
sys.path.insert(0, '.')

from verifier_precommis_dipankar_sigma_row10 import tracer, DELTA

PAS_BASE = 59000
PAS_TEST = 59989


def top_contributeurs(delta):
    logits = tracer(delta)
    _, _, _, _, ligne10_0, _, _ = logits[PAS_BASE]
    _, _, _, _, ligne10_t, _, _ = logits[PAS_TEST]
    d = ligne10_t - ligne10_0
    autres_idx = [j for j in range(27) if j not in (3, 4)]
    vals = [(j, d[j].item()) for j in autres_idx]
    vals.sort(key=lambda x: abs(x[1]), reverse=True)
    return vals


def main():
    for label, delta in (("reel", DELTA), ("delta=0", 0.0)):
        vals = top_contributeurs(delta)
        somme = sum(abs(v) for _, v in vals)
        print(f"{label}  top6={vals[:6]}")
        print(f"{label}  somme |d_logit_r[10,j]| (j!=3,4) = {somme:.6f}")


if __name__ == "__main__":
    main()
