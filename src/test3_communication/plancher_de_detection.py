"""Le nombre de rupture contre sa loi nulle, la puissance, et ce qu'aucune des deux ne pose.

Le relecteur montre que la regle 6 publiait un entier nu : « meurt a 2 runs sur
150 » ne veut rien dire tant qu'on ne sait pas a combien de runs meurt un effet
vrai de cette taille. Il plante l'effet et mesure. Verifie ici, puis :

  A. sa verification du plafond recompense <= R/27, qui est un denombrement
  B. le nombre de rupture contre son nul plante
  C. sa puissance de 0,665 est son p en costume — identite de Hoenig et Heisey
  D. le plancher de detection pour chaque contraste du tableau
  E. ce que ni la puissance ni le plancher ne posent : le seuil de pertinence
"""

import json
import numpy as np
from scipy import stats as st

DECOUVERTE = "results_test3/effet_par_beta_30graines_g0.json"
REPLICATION = "results_test3/effet_par_beta_12graines_g7.json"


def charger(chemin):
    with open(chemin, encoding="utf-8") as flux:
        return json.load(flux)["runs"]


def sigma_commun(valeurs, etiquettes, niveaux):
    somme, ddl = 0.0, 0
    for niveau in niveaux:
        bloc = valeurs[etiquettes == niveau]
        if len(bloc) >= 2:
            somme += ((bloc - bloc.mean()) ** 2).sum()
            ddl += len(bloc) - 1
    return np.sqrt(somme / ddl), ddl


def t_contraste(valeurs, etiquettes, a, b, niveaux):
    sigma, _ = sigma_commun(valeurs, etiquettes, niveaux)
    va, vb = valeurs[etiquettes == a], valeurs[etiquettes == b]
    if len(va) < 2 or len(vb) < 2:
        return np.nan
    return (va.mean() - vb.mean()) / (sigma * np.sqrt(1 / len(va) + 1 / len(vb)))


def rupture(valeurs, etiquettes, a, b, niveaux, seuil=1.98, kmax=40):
    """Plus petit nombre de runs a retirer, choisis gloutonnement, pour passer sous le seuil."""
    garde = np.ones(len(valeurs), dtype=bool)
    for k in range(1, kmax + 1):
        meilleur = None
        for i in np.where(garde)[0]:
            garde[i] = False
            t = t_contraste(valeurs[garde], etiquettes[garde], a, b, niveaux)
            garde[i] = True
            if not np.isnan(t) and (meilleur is None or abs(t) < meilleur[0]):
                meilleur = (abs(t), i)
        if meilleur is None:
            return kmax + 1
        garde[meilleur[1]] = False
        if meilleur[0] < seuil:
            return k
    return kmax + 1


def planter(residus, effectifs, delta, generateur):
    """Un echantillon ou l'effet est vrai par construction et identique partout."""
    niveaux = list(effectifs.keys())
    valeurs, etiquettes = [], []
    for niveau in niveaux:
        tirage = generateur.choice(residus, size=effectifs[niveau], replace=True)
        if niveau == 25:
            tirage = tirage + delta
        valeurs.append(tirage)
        etiquettes.append(np.full(effectifs[niveau], niveau))
    return np.concatenate(valeurs), np.concatenate(etiquettes)


def main():
    runs_d = charger(DECOUVERTE)
    runs_r = charger(REPLICATION)
    ecart = np.array([r["ecart"] for r in runs_d])
    beta = np.array([r["beta"] for r in runs_d])
    R = np.array([r["R"] for r in runs_d])
    reward = np.array([r["reward"] for r in runs_d])
    ecart_r = np.array([r["ecart"] for r in runs_r])
    beta_r = np.array([r["beta"] for r in runs_r])
    R_r = np.array([r["R"] for r in runs_r])
    reward_r = np.array([r["reward"] for r in runs_r])
    niv_R = [27, 26, 25, 24, 23]
    niv_B = sorted(set(beta.tolist()))

    print("=" * 74)
    print("A. LE PLAFOND recompense <= R/27 EST-IL UN DENOMBREMENT ?")
    print("=" * 74)
    for etiquette, w, r in [("decouverte", reward, R), ("replication", reward_r, R_r)]:
        sous = int((w <= r / 27 + 1e-9).sum())
        deficit = np.round((r / 27 - w) * 27).astype(int)
        compte = {int(v): int((deficit == v).sum()) for v in sorted(set(deficit.tolist()))}
        print(f"   {etiquette:12s} recompense <= R/27 : {sous}/{len(w)}   "
              f"deficit {compte}   min {deficit.min()}  max {deficit.max()}")
    deficit = np.round((R / 27 - reward) * 27).astype(int)
    print(f"\n   les {int((deficit == 1).sum())} runs a deficit 1, par R :")
    for niveau in niv_R:
        m = (deficit == 1) & (R == niveau)
        if m.sum():
            print(f"     R = {niveau} : {int(m.sum())} runs, beta = "
                  f"{sorted(beta[m].tolist())}")
    print(f"\n   ecart des runs a deficit 1 : {ecart[deficit == 1].mean():.5f}   "
          f"des autres : {ecart[deficit == 0].mean():.5f}")
    m_bump = (beta == 0.03) & (R == 25)
    print(f"   dans la cellule du bump (beta=0,03 R=25) : "
          f"{int((deficit[m_bump] == 1).sum())} des {int(m_bump.sum())} runs a deficit 1")

    # la colonne de remplacement qu'il propose
    k = np.round(27 * reward).astype(int)
    print(f"\n   contraste sur R                                t = "
          f"{t_contraste(ecart, R, 25, 24, niv_R):+.3f}")
    print(f"   contraste sur k = round(27 * recompense)       t = "
          f"{t_contraste(ecart, k, 25, 24, sorted(set(k.tolist()), reverse=True)):+.3f}"
          f"   (il annonce +2,226)")

    print()
    print("=" * 74)
    print("B. LE NOMBRE DE RUPTURE CONTRE SON NUL PLANTE")
    print("=" * 74)
    sigma, ddl = sigma_commun(ecart, R, niv_R)
    residus = np.concatenate([ecart[R == n] - ecart[R == n].mean() for n in niv_R])
    effectifs = {n: int((R == n).sum()) for n in niv_R}
    erreur = sigma * np.sqrt(1 / effectifs[25] + 1 / effectifs[24])
    t_obs = t_contraste(ecart, R, 25, 24, niv_R)
    print(f"   sigma {sigma:.6f}   SE du contraste {erreur:.6f}   t observe {t_obs:+.3f}")

    generateur = np.random.default_rng(20260815)
    resultats = {}
    for facteur in [1.0, 1.5, 2.0, 3.0]:
        delta = facteur * t_obs * erreur
        ruptures, ts = [], []
        for _ in range(600):
            v, e = planter(residus, effectifs, delta, generateur)
            ts.append(t_contraste(v, e, 25, 24, niv_R))
            ruptures.append(rupture(v, e, 25, 24, niv_R))
        ruptures = np.array(ruptures)
        ts = np.array(ts)
        resultats[facteur] = (ruptures, ts)
        if facteur == 1.0:
            print(f"\n   effet vrai calibre a E[t] = {t_obs:.3f} :")
            print(f"     mediane {np.median(ruptures):.0f}   moyenne {ruptures.mean():.1f}")
            print(f"     q10 {np.quantile(ruptures,.10):.0f}   q25 {np.quantile(ruptures,.25):.0f}   "
                  f"q50 {np.quantile(ruptures,.50):.0f}   q75 {np.quantile(ruptures,.75):.0f}   "
                  f"q90 {np.quantile(ruptures,.90):.0f}")
            print(f"     P(rupture <= 2) {(ruptures <= 2).mean():.3f}   "
                  f"P(rupture <= 3) {(ruptures <= 3).mean():.3f}")
            print(f"     part des replicats a |t| >= 1,98 : {(np.abs(ts) >= 1.98).mean():.3f}")

    print(f"\n   {'facteur':>8}{'E[t]':>8}{'mediane rupture':>18}{'P(>= 10)':>11}")
    for facteur, (ruptures, ts) in resultats.items():
        print(f"   {facteur:>8.1f}{np.abs(ts).mean():>8.2f}"
              f"{np.median(ruptures):>18.1f}{(ruptures >= 10).mean():>11.2f}")

    print()
    print("=" * 74)
    print("C. SA PUISSANCE DE 0,665 EST SON p EN COSTUME")
    print("=" * 74)
    print("   La puissance calculee a la taille d'effet observee est une fonction")
    print("   biunivoque decroissante du p (Hoenig et Heisey 2001). Elle ne peut donc")
    print("   rien contenir que t ne contienne deja. Verification directe :\n")
    print(f"   {'t':>6}{'p bilateral':>14}{'puissance a la taille observee':>32}")
    for t in [1.0, 1.5, 1.98, 2.43, 2.97, 3.5, 4.0]:
        p = 2 * (1 - st.t.cdf(t, ddl))
        seuil = st.t.ppf(0.975, ddl)
        puissance = (1 - st.nct.cdf(seuil, ddl, t)) + st.nct.cdf(-seuil, ddl, t)
        print(f"   {t:>6.2f}{p:>14.4f}{puissance:>32.3f}")
    seuil = st.t.ppf(0.975, ddl)
    puissance_obs = (1 - st.nct.cdf(seuil, ddl, t_obs)) + st.nct.cdf(-seuil, ddl, t_obs)
    print(f"\n   sa valeur 0,665 pour t = {t_obs:.3f} : analytique {puissance_obs:.3f}, "
          f"simulee {(np.abs(resultats[1.0][1]) >= 1.98).mean():.3f}")
    print("   Elle est vraie. Elle est aussi entierement determinee par t, donc elle")
    print("   ne mesure pas le plan : elle rehabille p = 0,016.")

    print()
    print("=" * 74)
    print("D. LE PLANCHER DE DETECTION DE CHAQUE CONTRASTE")
    print("=" * 74)
    print("   Quantite du plan et non des donnees : plus petit effet vrai detectable")
    print("   a p < 0,05 bilateral avec 80 % de puissance, delta = 2,80 * SE.\n")
    print(f"   {'contraste':<26}{'n':>10}{'SE':>10}{'plancher':>11}{'observe':>11}{'obs/plancher':>14}")
    lignes = []
    for a, b in [(27, 26), (26, 25), (25, 24), (24, 23)]:
        na, nb = effectifs[a], effectifs[b]
        se = sigma * np.sqrt(1 / na + 1 / nb)
        d = ecart[R == a].mean() - ecart[R == b].mean()
        lignes.append((f"R {a} contre {b}", f"{na}+{nb}", se, 2.80 * se, d))
    sigma_b, _ = sigma_commun(ecart, beta, niv_B)
    for a, b in [(0.005, 0.03), (0.005, 0.037)]:
        na = int((beta == a).sum())
        nb = int((beta == b).sum())
        se = sigma_b * np.sqrt(1 / na + 1 / nb)
        d = ecart[beta == a].mean() - ecart[beta == b].mean()
        lignes.append((f"beta {a} contre {b}", f"{na}+{nb}", se, 2.80 * se, d))
    for nom, n, se, plancher, d in lignes:
        print(f"   {nom:<26}{n:>10}{se:>10.5f}{plancher:>11.5f}{d:>+11.5f}"
              f"{abs(d)/plancher:>14.2f}")

    print("\n   tirage necessaire pour porter l'effet observe a 90 % de puissance :")
    for nom, cible in [("contraste R 25/24", abs(t_obs))]:
        # n par cellule tel que 3.24 * sigma * sqrt(2/n) <= effet observe
        effet = abs(ecart[R == 25].mean() - ecart[R == 24].mean())
        n_par_cellule = (3.24 * sigma * np.sqrt(2) / effet) ** 2
        print(f"     {nom} : {n_par_cellule:.0f} runs par cellule, soit environ "
              f"{n_par_cellule * 2 / (100/150):.0f} runs au total")
        print(f"     (les cellules R ne se reglent pas : ce tirage n'est pas commandable)")
    effet_b = abs(ecart[beta == 0.005].mean() - ecart[beta == 0.03].mean())
    n_b = (3.24 * sigma_b * np.sqrt(2) / effet_b) ** 2
    print(f"     contraste beta 0,005/0,03 : {n_b:.0f} graines par niveau, "
          f"{n_b*5:.0f} runs — commandable, lui")

    print()
    print("=" * 74)
    print("E. CE QUE NI LA PUISSANCE NI LE PLANCHER NE POSENT")
    print("=" * 74)
    print("   Les deux demandent « quel effet puis-je voir ». Aucun ne demande")
    print("   « quel effet changerait une conclusion ». Le tableau de §7.25 existait")
    print("   pour une seule raison : savoir si l'ecart produit par la dynamique")
    print("   approche le pire cas atteignable par recherche.\n")
    borne_pire = 0.1443
    observe = ecart.mean()
    print(f"   pire cas atteignable sous plancher R (publie)   {borne_pire:.4f}")
    print(f"   ecart moyen produit par la dynamique            {observe:.4f}")
    print(f"   rapport                                         {borne_pire/observe:.1f}")
    print()
    for rapport_cible in [2.0, 1.0]:
        besoin = borne_pire / rapport_cible - observe
        print(f"   pour ramener ce rapport a {rapport_cible:.0f}, l'ecart devrait monter de "
              f"{besoin:+.4f}")
    plancher_min = min(l[3] for l in lignes)
    plancher_max = max(l[3] for l in lignes)
    print(f"\n   planchers de detection du tableau : {plancher_min:.5f} a {plancher_max:.5f}")
    print(f"   plus grand effet observe du tableau : "
          f"{max(abs(l[4]) for l in lignes):.5f}")
    besoin = borne_pire / 2 - observe
    print(f"   plus petit effet qui changerait une conclusion : {besoin:.4f}")
    print(f"\n   le plan est {besoin/plancher_max:.0f} a {besoin/plancher_min:.0f} fois plus "
          f"sensible qu'il n'a besoin de l'etre,")
    print(f"   et chaque contraste du tableau est {besoin/max(abs(l[4]) for l in lignes):.0f} "
          f"fois sous le seuil de pertinence.")


if __name__ == "__main__":
    main()
