"""Le bump de la cellule (beta = 0,03, R = 25) : verification, puis quatre crans plus loin.

Le relecteur montre qu'une cellule de treize runs porte les deux contrastes de tete.
Ce script verifie ce point, puis ouvre ce qu'il a laisse ferme :

  1. sa verification, refaite ici
  2. la cellule vue run par run — treize, ou un ?
  3. sa propre diagnostique est un maximum sur vingt-cinq retraits
  4. le test d'interaction qu'aucun de nous deux n'a fait
  5. la colonne du fichier que personne n'a ouverte : reward
  6. ce que devient le bump au second tirage
"""

import json
import itertools
import numpy as np
from scipy import stats as st

DECOUVERTE = "results_test3/effet_par_beta_30graines_g0.json"
REPLICATION = "results_test3/effet_par_beta_12graines_g7.json"


def charger(chemin):
    with open(chemin, encoding="utf-8") as flux:
        return json.load(flux)["runs"]


def champs(runs):
    return (
        np.array([r["ecart"] for r in runs]),
        np.array([r["beta"] for r in runs]),
        np.array([r["R"] for r in runs]),
        np.array([r["reward"] for r in runs]),
        np.array([r["max"] for r in runs]),
        np.array([r["appariee"] for r in runs]),
    )


def t_contraste(valeurs, etiquettes, a, b, niveaux):
    """t d'un contraste, ecart-type mis en commun sur tous les niveaux presents."""
    somme, ddl = 0.0, 0
    for niveau in niveaux:
        bloc = valeurs[etiquettes == niveau]
        if len(bloc) >= 2:
            somme += ((bloc - bloc.mean()) ** 2).sum()
            ddl += len(bloc) - 1
    if ddl == 0:
        return np.nan
    sigma = np.sqrt(somme / ddl)
    va, vb = valeurs[etiquettes == a], valeurs[etiquettes == b]
    if len(va) < 1 or len(vb) < 1:
        return np.nan
    erreur = sigma * np.sqrt(1 / len(va) + 1 / len(vb))
    return (va.mean() - vb.mean()) / erreur


def main():
    runs_d = charger(DECOUVERTE)
    runs_r = charger(REPLICATION)
    ecart, beta, R, reward, mx, app = champs(runs_d)
    ecart_r, beta_r, R_r, reward_r, _, _ = champs(runs_r)
    niv_R = [27, 26, 25, 24, 23]
    niv_B = sorted(set(beta.tolist()))

    print("=" * 74)
    print("1. SA VERIFICATION, REFAITE ICI")
    print("=" * 74)
    t_R0 = t_contraste(ecart, R, 25, 24, niv_R)
    t_B0 = t_contraste(ecart, beta, 0.005, 0.03, niv_B)
    print(f"   sans rien retirer   t(R 25v24) = {t_R0:+.3f}   t(beta .005v.03) = {t_B0:+.3f}")

    lignes = []
    for b, r in itertools.product(niv_B, niv_R):
        garde = ~((beta == b) & (R == r))
        n_cell = int((~garde).sum())
        if n_cell == 0:
            continue
        tR = t_contraste(ecart[garde], R[garde], 25, 24, niv_R)
        tB = t_contraste(ecart[garde], beta[garde], 0.005, 0.03, niv_B)
        lignes.append((b, r, n_cell, tR, tB))

    lignes.sort(key=lambda x: abs(x[3]))
    print("\n   retrait d'une cellule, les cinq qui font le plus bouger t(R) :")
    print(f"   {'cellule':<22}{'n':>4}{'t(R)':>10}{'t(beta)':>11}{'delta t(R)':>13}")
    for b, r, n, tR, tB in lignes[:5]:
        print(f"   beta={b:<8} R={r:<8}{n:>4}{tR:>10.3f}{tB:>11.3f}{tR-t_R0:>13.3f}")
    print("\n   amplitude sur les vingt autres cellules :")
    reste = [abs(x[3] - t_R0) for x in lignes[5:]]
    print(f"     |delta t(R)| max = {max(reste):.3f}")

    print()
    print("=" * 74)
    print("2. LA CELLULE VUE RUN PAR RUN — TREIZE, OU UN ?")
    print("=" * 74)
    masque = (beta == 0.03) & (R == 25)
    idx = np.where(masque)[0]
    print(f"   {len(idx)} runs a beta = 0,03, R = 25")
    print(f"   {'#':>3}{'ecart':>11}{'max':>11}{'appariee':>11}{'reward':>11}")
    for rang, i in enumerate(sorted(idx, key=lambda j: -ecart[j])):
        print(f"   {rang+1:>3}{ecart[i]:>11.5f}{mx[i]:>11.5f}{app[i]:>11.5f}{reward[i]:>11.5f}")
    print(f"\n   moyenne {ecart[idx].mean():.5f}   mediane {np.median(ecart[idx]):.5f}")
    autres = ecart[~masque]
    print(f"   le reste du plan : moyenne {autres.mean():.5f}  mediane {np.median(autres):.5f}")

    print("\n   retrait d'un run a l'interieur de la cellule :")
    print(f"   {'run retire':>12}{'t(R)':>10}{'t(beta)':>11}")
    pires = []
    for i in idx:
        garde = np.ones(len(ecart), dtype=bool)
        garde[i] = False
        tR = t_contraste(ecart[garde], R[garde], 25, 24, niv_R)
        tB = t_contraste(ecart[garde], beta[garde], 0.005, 0.03, niv_B)
        pires.append((abs(tR), i, tR, tB))
    pires.sort()
    for _, i, tR, tB in pires[:4]:
        print(f"   ecart={ecart[i]:.5f}{tR:>10.3f}{tB:>11.3f}")
    print(f"   ... et un seul run de 150 fait passer t(R) de {t_R0:.3f} a {pires[0][2]:.3f}")

    # combien de runs faut-il retirer pour passer sous 1,98 ?
    ordre = sorted(idx, key=lambda j: -ecart[j])
    for k in range(1, len(ordre) + 1):
        garde = np.ones(len(ecart), dtype=bool)
        garde[ordre[:k]] = False
        tR = t_contraste(ecart[garde], R[garde], 25, 24, niv_R)
        if abs(tR) < 1.98:
            print(f"   il suffit de retirer les {k} plus grands ecarts de la cellule "
                  f"({k}/150 runs) pour passer sous 1,98 : t = {tR:+.3f}")
            break

    print()
    print("=" * 74)
    print("3. SA DIAGNOSTIQUE EST ELLE-MEME UN MAXIMUM SUR VINGT-CINQ RETRAITS")
    print("=" * 74)
    print("   Il retire les 25 cellules et lit la plus grande chute. C'est exactement")
    print("   la faute qu'il m'a apprise, un etage plus haut. Loi nulle de la chute")
    print("   maximale, par permutation des ecarts sur le plan reel :\n")
    chute_obs = t_R0 - min(x[3] for x in lignes)
    generateur = np.random.default_rng(4242)
    chutes = []
    for _ in range(4000):
        melange = generateur.permutation(ecart)
        t0 = t_contraste(melange, R, 25, 24, niv_R)
        pire = t0
        for b, r in itertools.product(niv_B, niv_R):
            garde = ~((beta == b) & (R == r))
            if garde.all():
                continue
            tt = t_contraste(melange[garde], R[garde], 25, 24, niv_R)
            if not np.isnan(tt) and abs(tt) < abs(pire):
                pire = tt
        chutes.append(abs(t0) - abs(pire))
    chutes = np.array(chutes)
    print(f"   chute observee                          {chute_obs:.3f}")
    print(f"   E[chute max] sous le nul                {chutes.mean():.3f}")
    print(f"   q90 / q95 / q99                         "
          f"{np.quantile(chutes,.90):.3f} / {np.quantile(chutes,.95):.3f} / "
          f"{np.quantile(chutes,.99):.3f}")
    print(f"   P(chute nulle >= chute observee)        {(chutes >= chute_obs).mean():.4f}")

    print()
    print("=" * 74)
    print("4. LE TEST D'INTERACTION, QUE NI LUI NI MOI N'AVONS FAIT")
    print("=" * 74)
    print("   Il montre que l'ajustement additif ne l'attrape pas, et s'arrete la.")
    print("   L'enonce qu'il decrit — un effet dans une cellule — a un test.\n")

    def interaction_f(v, b, r, niveaux_b, niveaux_r):
        """Somme des carres d'interaction par ajustement de moyennes de cellules."""
        cellules = [(x, y) for x in niveaux_b for y in niveaux_r
                    if ((b == x) & (r == y)).sum() >= 2]
        if len(cellules) < len(niveaux_b) + len(niveaux_r) - 1:
            return np.nan, 0, 0
        # modele complet : moyenne par cellule
        sc_complet, n_tot = 0.0, 0
        for x, y in cellules:
            bloc = v[(b == x) & (r == y)]
            sc_complet += ((bloc - bloc.mean()) ** 2).sum()
            n_tot += len(bloc)
        p_complet = len(cellules)
        # modele additif : moindres carres sur indicatrices
        masque = np.zeros(len(v), dtype=bool)
        for x, y in cellules:
            masque |= (b == x) & (r == y)
        vv, bb, rr = v[masque], b[masque], r[masque]
        colonnes = [np.ones(len(vv))]
        for x in niveaux_b[1:]:
            colonnes.append((bb == x).astype(float))
        for y in niveaux_r[1:]:
            colonnes.append((rr == y).astype(float))
        X = np.stack(colonnes, axis=1)
        coef, *_ = np.linalg.lstsq(X, vv, rcond=None)
        residus = vv - X @ coef
        sc_additif = (residus ** 2).sum()
        p_additif = np.linalg.matrix_rank(X)
        ddl1 = p_complet - p_additif
        ddl2 = n_tot - p_complet
        if ddl1 <= 0 or ddl2 <= 0:
            return np.nan, ddl1, ddl2
        f = ((sc_additif - sc_complet) / ddl1) / (sc_complet / ddl2)
        return f, ddl1, ddl2

    f_obs, d1, d2 = interaction_f(ecart, beta, R, niv_B, niv_R)
    print(f"   F d'interaction observe    F({d1},{d2}) = {f_obs:.3f}   "
          f"p nominal = {1 - st.f.cdf(f_obs, d1, d2):.4f}")

    generateur = np.random.default_rng(777)
    fs = []
    for _ in range(4000):
        melange = generateur.permutation(ecart)
        f, _, _ = interaction_f(melange, beta, R, niv_B, niv_R)
        if not np.isnan(f):
            fs.append(f)
    fs = np.array(fs)
    print(f"   p par permutation          {(fs >= f_obs).mean():.4f}   "
          f"(la permutation ne suppose pas la normalite)")

    print()
    print("=" * 74)
    print("5. LA COLONNE QUE PERSONNE N'A OUVERTE : reward")
    print("=" * 74)
    print("   R et reward sont deux sorties du meme run. Si l'ecart suit reward et")
    print("   que R suit reward, alors R n'est qu'un proxy et le contraste n'est")
    print("   meme pas 'a propos d'une cellule' : il est a propos d'autre chose.\n")
    print(f"   corr(ecart, reward)  {st.pearsonr(reward, ecart)[0]:+.4f}   "
          f"p = {st.pearsonr(reward, ecart)[1]:.4f}")
    print(f"   corr(R, reward)      {st.pearsonr(reward, R.astype(float))[0]:+.4f}")
    print(f"   corr(ecart, R)       {st.pearsonr(R.astype(float), ecart)[0]:+.4f}")
    print()
    print("   reward par cellule autour du bump :")
    for b in niv_B:
        bout = []
        for r in [26, 25, 24]:
            m = (beta == b) & (R == r)
            bout.append(f"R={r}: {reward[m].mean():.4f} ({m.sum():2d})" if m.sum() else f"R={r}:    ---   ")
        print(f"     beta={b:<7} " + "   ".join(bout))
    print()
    m25 = (beta == 0.03) & (R == 25)
    m24 = (beta == 0.03) & (R == 24)
    print(f"   la cellule du bump : reward {reward[m25].mean():.5f} contre "
          f"{reward[m24].mean():.5f} une colonne plus loin")

    # contraste R apres ajustement lineaire sur reward
    X = np.stack([np.ones(len(ecart)), reward], axis=1)
    coef, *_ = np.linalg.lstsq(X, ecart, rcond=None)
    residus = ecart - X @ coef
    print(f"\n   contraste R 25v24 sur l'ecart brut          t = {t_R0:+.3f}")
    print(f"   sur le residu apres regression sur reward   "
          f"t = {t_contraste(residus, R, 25, 24, niv_R):+.3f}")

    print()
    print("=" * 74)
    print("6. LE BUMP AU SECOND TIRAGE")
    print("=" * 74)
    print("   Il note que la cellule s'inverse. De combien, et est-ce lisible ?\n")
    print(f"   {'beta':<8}{'R=25 decouv.':>15}{'R=24 decouv.':>15}"
          f"{'R=25 replic.':>15}{'R=24 replic.':>15}")
    for b in niv_B:
        def cel(v, bb, rr, x, y):
            m = (bb == x) & (rr == y)
            return f"{v[m].mean():.4f} ({m.sum():2d})" if m.sum() else "   ---     "
        print(f"   {b:<8}{cel(ecart,beta,R,b,25):>15}{cel(ecart,beta,R,b,24):>15}"
              f"{cel(ecart_r,beta_r,R_r,b,25):>15}{cel(ecart_r,beta_r,R_r,b,24):>15}")

    m25d = (beta == 0.03) & (R == 25)
    m24d = (beta == 0.03) & (R == 24)
    m25r = (beta_r == 0.03) & (R_r == 25)
    m24r = (beta_r == 0.03) & (R_r == 24)
    gap_d = ecart[m25d].mean() - ecart[m24d].mean()
    gap_r = ecart_r[m25r].mean() - ecart_r[m24r].mean()
    print(f"\n   ecart interne a beta = 0,03 : decouverte {gap_d:+.5f} "
          f"(n = {m25d.sum()} et {m24d.sum()})")
    print(f"                                 replication {gap_r:+.5f} "
          f"(n = {m25r.sum()} et {m24r.sum()})")

    f_r, d1r, d2r = interaction_f(ecart_r, beta_r, R_r, niv_B, [26, 25, 24])
    print(f"\n   F d'interaction sur les 60 runs neufs : F({d1r},{d2r}) = {f_r:.3f}  "
          f"p = {1 - st.f.cdf(f_r, d1r, d2r):.4f}" if not np.isnan(f_r)
          else "\n   les 60 runs ne remplissent pas assez de cellules pour l'interaction")

    print()
    print("=" * 74)
    print("7. SCHEFFE, ET MA PHRASE FAUSSE")
    print("=" * 74)
    seuil = np.sqrt(4 * st.f.ppf(0.90, 4, 145))
    print(f"   Scheffe, 5 niveaux, 145 ddl, famille 0,10 : |t| >= {seuil:.3f}")
    p_scheffe = 1 - st.f.cdf(t_R0 ** 2 / 4, 4, 145)
    print(f"   p de Scheffe pour t = {t_R0:.3f} : {p_scheffe:.3f}   (le sien : 0,212)")
    p_scheffe_b = 1 - st.f.cdf(t_B0 ** 2 / 4, 4, 145)
    print(f"   p de Scheffe pour t = {t_B0:.3f} : {p_scheffe_b:.3f}")
    print(f"\n   ma phrase : « tout mon tableau etait sous le q90 corrige » (2,73)")
    print(f"   le maximum du tableau vaut {abs(t_B0):.3f} > 2,73. Faux.")


if __name__ == "__main__":
    main()
