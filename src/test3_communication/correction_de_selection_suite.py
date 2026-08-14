"""Suite de correction_de_selection.py : les trois points restes ouverts.

  A. pourquoi ma route parametrique donne un maximum plus grand que la sienne
  B. le bout-en-bout conditionne sur la paire reellement gagnante
  C. la ligne beta, qui offrait un contraste PLUS GRAND que celui que j'ai lu,
     et que je n'ai jamais mentionne — plus sa replication sur les 60 runs neufs
"""

import json
import numpy as np
from scipy import stats as st

CHEMIN_DECOUVERTE = "results_test3/effet_par_beta_30graines_g0.json"
CHEMIN_REPLICATION = "results_test3/effet_par_beta_12graines_g7.json"


def charger(chemin):
    with open(chemin, encoding="utf-8") as flux:
        return json.load(flux)


def contrastes(valeurs, etiquettes, niveaux):
    moyennes, effectifs = {}, {}
    somme_carres, ddl = 0.0, 0
    for niveau in niveaux:
        bloc = valeurs[etiquettes == niveau]
        moyennes[niveau] = bloc.mean()
        effectifs[niveau] = len(bloc)
        somme_carres += ((bloc - bloc.mean()) ** 2).sum()
        ddl += len(bloc) - 1
    sigma = np.sqrt(somme_carres / ddl)
    sortie = []
    for i, a in enumerate(niveaux):
        for b in niveaux[i + 1:]:
            erreur = sigma * np.sqrt(1 / effectifs[a] + 1 / effectifs[b])
            d = moyennes[a] - moyennes[b]
            sortie.append(((a, b), d, erreur, d / erreur))
    return sortie, sigma, ddl


def max_t_bloc(effectifs, tirages, generateur, sigma_connu):
    """max |t| sur dix contrastes, avec sigma connu ou reestime."""
    effectifs = np.asarray(effectifs)
    total, k = effectifs.sum(), len(effectifs)
    frontieres = np.cumsum(effectifs)[:-1]
    paires = [(i, j) for i in range(k) for j in range(i + 1, k)]
    facteur = np.array([np.sqrt(1 / effectifs[i] + 1 / effectifs[j]) for i, j in paires])
    indice_cible = paires.index((2, 3))

    lots_t, lots_d, lots_c = [], [], []
    restant = tirages
    while restant > 0:
        n = min(20_000, restant)
        restant -= n
        echantillon = generateur.normal(0.0, 1.0, (n, total))
        morceaux = np.split(echantillon, frontieres, axis=1)
        moyennes = np.stack([m.mean(axis=1) for m in morceaux], axis=1)
        if sigma_connu:
            sigma = np.ones(n)
        else:
            sc = sum(((m - m.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) for m in morceaux)
            sigma = np.sqrt(sc / (total - k))
        d = np.stack([moyennes[:, i] - moyennes[:, j] for i, j in paires], axis=1)
        t = d / (sigma[:, None] * facteur[None, :])
        gagnant = np.argmax(np.abs(t), axis=1)
        lignes = np.arange(n)
        lots_t.append(np.abs(t[lignes, gagnant]))
        lots_d.append(np.abs(d[lignes, gagnant]))
        lots_c.append(gagnant == indice_cible)
    return np.concatenate(lots_t), np.concatenate(lots_d), np.concatenate(lots_c)


def main():
    decouverte = charger(CHEMIN_DECOUVERTE)
    replication = charger(CHEMIN_REPLICATION)
    ecarts_d = np.array([r["ecart"] for r in decouverte["runs"]])
    R_d = np.array([r["R"] for r in decouverte["runs"]])
    beta_d = np.array([r["beta"] for r in decouverte["runs"]])
    ecarts_r = np.array([r["ecart"] for r in replication["runs"]])
    R_r = np.array([r["R"] for r in replication["runs"]])
    beta_r = np.array([r["beta"] for r in replication["runs"]])

    niveaux_R = [27, 26, 25, 24, 23]
    effectifs = [int((R_d == n).sum()) for n in niveaux_R]
    liste_R, sd_R, ddl_R = contrastes(ecarts_d, R_d, niveaux_R)

    print("=" * 72)
    print("A. SIGMA CONNU CONTRE SIGMA REESTIME")
    print("=" * 72)
    print("   Sa loi du max est plus serree que la mienne. La seule difference")
    print("   de procedure possible est la : t a 145 ddl a des queues plus")
    print("   lourdes qu'une gaussienne.\n")
    for etiquette, connu in [("sigma connu (z)", True), ("sigma reestime (t, 145 ddl)", False)]:
        gen = np.random.default_rng(4242)
        mt, _, _ = max_t_bloc(effectifs, 400_000, gen, connu)
        print(f"   {etiquette:30s}  E[max] {mt.mean():.3f}   q90 {np.quantile(mt,0.90):.3f}   "
              f"P(>=2.43) {(mt>=2.43).mean():.4f}")
    print("\n   le sien : E[max] 1.620   q90 2.427   P(>=2.40) 0.1066")
    print("   ma permutation : E[max] 1.625   q90 2.429   P(>=2.40) 0.1075")

    print()
    print("=" * 72)
    print("B. BOUT EN BOUT, CONDITIONNE SUR LA PAIRE GAGNANTE")
    print("=" * 72)
    d_obs = [x for x in liste_R if x[0] == (25, 24)][0]
    err_d = d_obs[2]
    bloc25, bloc24 = ecarts_r[R_r == 25], ecarts_r[R_r == 24]
    sc, ddl = 0.0, 0
    for niveau in sorted(set(R_r.tolist())):
        b = ecarts_r[R_r == niveau]
        if len(b) >= 2:
            sc += ((b - b.mean()) ** 2).sum()
            ddl += len(b) - 1
    sd_r = np.sqrt(sc / ddl)
    err_r = sd_r * np.sqrt(1 / len(bloc25) + 1 / len(bloc24))
    d_rep = bloc25.mean() - bloc24.mean()

    gen = np.random.default_rng(31415)
    mt, dg, cible = max_t_bloc(effectifs, 400_000, gen, False)
    dg = dg * sd_R  # remise a l'echelle
    rep_nul = gen.normal(0.0, err_r, len(dg))
    pd_, pr_ = 1 / err_d ** 2, 1 / err_r ** 2
    commun = (dg * pd_ + rep_nul * pr_) / (pd_ + pr_)
    commun_c = commun[cible]
    commun_obs = (d_obs[1] * pd_ + d_rep * pr_) / (pd_ + pr_)

    print(f"   selection = max des dix, sans condition")
    print(f"     E[d decouverte]  {dg.mean():+.5f}    E[d commun]  {commun.mean():+.5f}")
    print(f"     P(commun >= {commun_obs:+.5f})  {(commun >= commun_obs).mean():.3f}")
    print(f"   selection = max des dix, ET le gagnant est la paire 25/24")
    print(f"     E[d decouverte]  {dg[cible].mean():+.5f}    E[d commun]  {commun_c.mean():+.5f}")
    print(f"     P(commun >= {commun_obs:+.5f})  {(commun_c >= commun_obs).mean():.3f}")
    print(f"\n   le sien : decouverte +0.00414, commun +0.00256, P = 0.434")
    print(f"   -> son bout-en-bout a conditionne sur la paire (0.00414),")
    print(f"      ce qui est le bon conditionnement, mais alors le P se lit")
    print(f"      contre {commun_obs:+.5f} et non contre +0.0028.")

    print()
    print("=" * 72)
    print("C. LA LIGNE BETA — LE CONTRASTE QUE JE N'AI JAMAIS RAPPORTE")
    print("=" * 72)
    betas = sorted(set(beta_d.tolist()))
    liste_b, sd_b, ddl_b = contrastes(ecarts_d, beta_d, betas)
    liste_b_triee = sorted(liste_b, key=lambda x: -abs(x[3]))
    print("   les trois plus grands contrastes de la ligne beta, a la decouverte :")
    for (a, b), d, e, t in liste_b_triee[:3]:
        p = 2 * (1 - st.t.cdf(abs(t), ddl_b))
        print(f"     beta={a} vs beta={b}   d = {d:+.5f}   t = {t:+.3f}   p nominal {p:.4f}")
    print(f"\n   pour memoire, le contraste R que j'ai publie : t = {d_obs[3]:+.3f}")
    print("   -> il n'etait meme pas le plus grand de la table ce jour-la.")

    (ba, bb), d_b, e_b, t_b = liste_b_triee[0]
    ra, rb = ecarts_r[beta_r == ba], ecarts_r[beta_r == bb]
    sc_b, ddl_bb = 0.0, 0
    for niveau in betas:
        blk = ecarts_r[beta_r == niveau]
        if len(blk) >= 2:
            sc_b += ((blk - blk.mean()) ** 2).sum()
            ddl_bb += len(blk) - 1
    sd_rb = np.sqrt(sc_b / ddl_bb)
    e_rb = sd_rb * np.sqrt(1 / len(ra) + 1 / len(rb))
    d_rb = ra.mean() - rb.mean()
    print(f"\n   sa replication sur les 60 runs independants :")
    print(f"     beta={ba} vs beta={bb}   d = {d_rb:+.5f}   SE = {e_rb:.5f}   "
          f"t = {d_rb/e_rb:+.3f}")
    print(f"     decouverte  {d_b:+.5f}  ->  replication  {d_rb:+.5f}")

    ecarts_tous = np.concatenate([ecarts_d, ecarts_r])
    beta_tous = np.concatenate([beta_d, beta_r])
    liste_t, _, ddl_t = contrastes(ecarts_tous, beta_tous, betas)
    pool_b = [x for x in liste_t if x[0] == (ba, bb)][0]
    print(f"     210 runs repooles : d = {pool_b[1]:+.5f}  t = {pool_b[3]:+.3f}")

    # F de la ligne beta, avant et apres
    def anova(valeurs, etiquettes, niveaux):
        grande = valeurs.mean()
        entre = sum(len(valeurs[etiquettes == n]) *
                    (valeurs[etiquettes == n].mean() - grande) ** 2 for n in niveaux)
        intra = sum(((valeurs[etiquettes == n] - valeurs[etiquettes == n].mean()) ** 2).sum()
                    for n in niveaux)
        d1, d2 = len(niveaux) - 1, len(valeurs) - len(niveaux)
        f = (entre / d1) / (intra / d2)
        eta = d1 * f / (d1 * f + d2)
        omega = d1 * (f - 1) / (d1 * (f - 1) + len(valeurs))
        return f, d1, d2, eta, omega, 1 - st.f.cdf(f, d1, d2)

    print(f"\n   analyse de variance de la ligne beta :")
    for etiquette, v, e in [("150 runs", ecarts_d, beta_d), ("210 runs", ecarts_tous, beta_tous)]:
        f, d1, d2, eta, omega, p = anova(v, e, betas)
        print(f"     {etiquette}  F({d1},{d2}) = {f:.3f}  p = {p:.3f}  "
              f"eta^2 = {eta*100:.2f} %  omega^2 = {omega*100:+.2f} %")

    print()
    print("=" * 72)
    print("D. LE p CORRIGE, JEU DE SELECTION COMPLET")
    print("=" * 72)
    gen = np.random.default_rng(777)
    t_R = abs(d_obs[3])
    t_B = abs(t_b)
    n_perm = 100_000
    d10 = d20 = d10b = 0
    for _ in range(n_perm):
        melange = gen.permutation(ecarts_d)
        lR, _, _ = contrastes(melange, R_d, niveaux_R)
        lB, _, _ = contrastes(melange, beta_d, betas)
        mR = max(abs(x[3]) for x in lR)
        mB = max(abs(x[3]) for x in lB)
        if mR >= t_R:
            d10 += 1
        if max(mR, mB) >= t_R:
            d20 += 1
        if max(mR, mB) >= t_B:
            d10b += 1
    print(f"   p nominal du contraste R seul                        "
          f"{2*(1-st.t.cdf(t_R, ddl_R)):.4f}")
    print(f"   corrige sur les 10 contrastes de la ligne R          {d10/n_perm:.4f}")
    print(f"   corrige sur les 20 contrastes lus le meme jour       {d20/n_perm:.4f}")
    print(f"   et pour le plus grand de tous (beta, t = {t_B:.2f}) :   {d10b/n_perm:.4f}")


if __name__ == "__main__":
    main()
