"""Piste emetteur (le "26" de H7), jamais testee -- generalise
verifier_jouet_2_referents.py (qui fige K=1, l'ablation totale des
25 autres messages) a K VARIABLE, pour tester si k(R_init) -- le
rapport de vitesse relatif emetteur/recepteur, deja trouve non
constant (1.42 a 2.45) et deja refute contre H_momentum/beta2 (tous
deux de simples reechelonnages, cf. CARNET.md 7.65/8ter) -- depend du
nombre K de messages "ailleurs" symetriques que l'emetteur normalise.

H7 (confirmee le 14/09, CARNET.md ~L6608) donne la forme fermee
d3 = K*exp(-N*poids3*r3/beta) au voisinage du col (ici K=26 dans le
vrai systeme). Le prefacteur K n'affecte QUE le seuil/l'echelle du
cote emetteur -- H13 a deja confirme que le reduire de 26 a 1 deplace
delta_c et le point d'atterrissage de l'effondrement (1/2 au lieu de
1/27). Ce qui n'a JAMAIS ete teste : est-ce que K affecte aussi
k(R_init), la PENTE de la derive de k avec l'etat initial -- la
question posee depuis le tour 51/52 et jamais refermee cote emetteur.

Construction : chaque emetteur (referent 3, referent 4) a un softmax
a K+1 issues (message10 vs K categories "ailleurs" symetriques,
repliees par symetrie en UN SEUL parametre agrege -- pas besoin
d'instancier K parametres separes, le facteur K entre directement dans
la fonction de partition et le terme d'entropie).
"""

import sys
import math
import torch

sys.path.insert(0, '.')

BETA = 0.02
N = 27


def construire_toy(K, s3_init=0.999, s4_init=0.999, r_init=0.5):
    p3 = torch.tensor([math.log(s3_init / (1 - s3_init))], dtype=torch.float64, requires_grad=True)
    p4 = torch.tensor([math.log(s4_init / (1 - s4_init))], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([math.log(r_init / (1 - r_init))], dtype=torch.float64, requires_grad=True)
    return p3, p4, q


def entropie_k(p, K):
    """Entropie du softmax a K+1 issues (1 pic + K categories symetriques
    de masse (1-s)/K chacune), exprimee via s=sigmoid(p) seul."""
    s = torch.sigmoid(p)
    un_moins_s = 1 - s
    terme_pic = -s * torch.log(s.clamp_min(1e-300))
    terme_autres = -un_moins_s * torch.log((un_moins_s / K).clamp_min(1e-300))
    return terme_pic + terme_autres


def objectif_toy(p3, p4, q, poids3, poids4, K):
    s3 = torch.sigmoid(p3)
    s4 = torch.sigmoid(p4)
    r4 = torch.sigmoid(q)
    r3 = 1 - r4
    recompense = poids3 * s3 * r3 + poids4 * s4 * r4
    entropie = entropie_k(p3, K) + entropie_k(p4, K) + (
        -(r4 * torch.log(r4.clamp_min(1e-300)) + r3 * torch.log(r3.clamp_min(1e-300)))
    )
    return (recompense + (BETA / N) * entropie).sum()


def entrainer(delta, K, r_init=0.5, pas=40000, lr=0.05, adam_eps=1e-10):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy(K, r_init=r_init)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=adam_eps)
    for _ in range(pas):
        j = objectif_toy(p3, p4, q, poids3, poids4, K)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    with torch.no_grad():
        s3 = torch.sigmoid(p3).item()
        r4 = torch.sigmoid(q).item()
    return s3, r4


def bissecter_delta_c(K, r_init, lo, hi, tol=1e-5, pas=40000):
    """Seuil de bascule a mi-chemin entre la valeur effondree exacte
    (1/(K+1), verifie empiriquement pour K=1,8,26) et la valeur graduee
    (~1) -- le seuil fixe 0.5 est ambigu/faux des que K != 1 (K=8
    effondre a 1/9=0.111, K=26 a 1/27=0.037, tous deux < 0.5 mais pas
    a 0.5 lui-meme)."""
    seuil = 0.5 * (1.0 + 1.0 / (K + 1))

    def grade(delta):
        s3, r4 = entrainer(delta, K, r_init=r_init, pas=pas)
        return s3 > seuil
    grade_lo, grade_hi = grade(lo), grade(hi)
    assert grade_lo != grade_hi, f"bornes ne separent pas a K={K}: lo={lo} hi={hi}"
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if grade(mid) == grade_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    print("=== non-regression : K=1 doit reproduire verifier_jouet_2_referents.py ===")
    for delta in (0.013, 0.02):
        s3, r4 = entrainer(delta, K=1)
        print(f"  K=1  delta={delta}  s3={s3:.6f}  r4={r4:.6f}")
