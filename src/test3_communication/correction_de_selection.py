"""Loi du maximum de contraste sous le nul, pour le tableau de §7.25.

Vérifie la correction de sélection proposée en neuvième tour de relecture : le
contraste R=25 contre R=24 était le plus grand de dix, et un plus grand de dix
n'a pas le p nominal d'un contraste unique.

Trois routes vers la même quantité, parce qu'une seule ne vaut rien ici :

  1. paramétrique, cellules fixes, écart-type commun    (la route du relecteur)
  2. permutation des écarts sur le plan réel            (aucune hypothèse)
  3. paramétrique, tailles de cellules retirées au sort (R est une sortie)

Et la correction d'omega carré, qui ne demande aucun tirage : elle se déduit des
deux F déjà publiés.
"""

import json
import numpy as np

CHEMIN_DECOUVERTE = "results_test3/effet_par_beta_30graines_g0.json"
CHEMIN_REPLICATION = "results_test3/effet_par_beta_12graines_g7.json"


def charger(chemin):
    with open(chemin, encoding="utf-8") as flux:
        return json.load(flux)


def contraste_t(valeurs, etiquettes, niveaux):
    """Tous les contrastes par paires, avec l'ecart-type mis en commun.

    Renvoie (differences, t, paires). L'ecart-type est estime sur les cinq
    cellules a la fois : c'est ce que fait une analyse de variance, et c'est
    l'estimateur du relecteur, pas les ecarts-types intra que j'avais publies.
    """
    moyennes = {}
    effectifs = {}
    somme_carres = 0.0
    ddl_intra = 0
    for niveau in niveaux:
        bloc = valeurs[etiquettes == niveau]
        moyennes[niveau] = bloc.mean()
        effectifs[niveau] = len(bloc)
        somme_carres += ((bloc - bloc.mean()) ** 2).sum()
        ddl_intra += len(bloc) - 1
    ecart_type = np.sqrt(somme_carres / ddl_intra)

    differences = []
    stats = []
    paires = []
    for i, a in enumerate(niveaux):
        for b in niveaux[i + 1:]:
            erreur = ecart_type * np.sqrt(1 / effectifs[a] + 1 / effectifs[b])
            difference = moyennes[a] - moyennes[b]
            differences.append(difference)
            stats.append(difference / erreur)
            paires.append((a, b))
    return np.array(differences), np.array(stats), paires


def route_1_parametrique(effectifs, ecart_type, tirages, generateur):
    """Cellules fixes, moyennes identiques, ecart-type commun connu.

    L'ecart-type est reestime a chaque tirage comme dans une vraie analyse :
    c'est ce qui rend t exactement libre d'echelle.
    """
    total = sum(effectifs)
    frontieres = np.cumsum(effectifs)[:-1]
    max_t = np.empty(tirages)
    d_gagnant = np.empty(tirages)
    gagnant_est_25_24 = np.empty(tirages, dtype=bool)

    # indice du contraste (25, 24) dans l'ordre lexicographique des paires
    # sur [27, 26, 25, 24, 23] : (27,26) (27,25) (27,24) (27,23) (26,25)
    # (26,24) (26,23) (25,24) (25,23) (24,23) -> position 7
    indice_cible = 7

    for tirage in range(tirages):
        echantillon = generateur.normal(0.0, ecart_type, total)
        blocs = np.split(echantillon, frontieres)
        moyennes = np.array([bloc.mean() for bloc in blocs])
        somme_carres = sum(((bloc - bloc.mean()) ** 2).sum() for bloc in blocs)
        sigma = np.sqrt(somme_carres / (total - len(effectifs)))

        differences = []
        stats = []
        for i in range(len(effectifs)):
            for j in range(i + 1, len(effectifs)):
                erreur = sigma * np.sqrt(1 / effectifs[i] + 1 / effectifs[j])
                difference = moyennes[i] - moyennes[j]
                differences.append(difference)
                stats.append(difference / erreur)
        differences = np.array(differences)
        stats = np.array(stats)

        gagnant = np.argmax(np.abs(stats))
        max_t[tirage] = np.abs(stats[gagnant])
        d_gagnant[tirage] = np.abs(differences[gagnant])
        gagnant_est_25_24[tirage] = gagnant == indice_cible

    return max_t, d_gagnant, gagnant_est_25_24


def route_1_vectorisee(effectifs, ecart_type, tirages, generateur, bloc=20000):
    """Meme chose, en blocs, pour tenir 400 000 tirages en un temps utile."""
    effectifs = np.asarray(effectifs)
    total = effectifs.sum()
    k = len(effectifs)
    frontieres = np.cumsum(effectifs)[:-1]
    paires = [(i, j) for i in range(k) for j in range(i + 1, k)]
    indice_cible = paires.index((2, 3))

    tous_max_t = []
    tous_d = []
    tous_cible = []
    restant = tirages
    while restant > 0:
        n = min(bloc, restant)
        restant -= n
        echantillon = generateur.normal(0.0, ecart_type, (n, total))
        morceaux = np.split(echantillon, frontieres, axis=1)
        moyennes = np.stack([m.mean(axis=1) for m in morceaux], axis=1)
        somme_carres = sum(
            ((m - m.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) for m in morceaux
        )
        sigma = np.sqrt(somme_carres / (total - k))

        differences = np.stack(
            [moyennes[:, i] - moyennes[:, j] for i, j in paires], axis=1
        )
        facteur = np.array(
            [np.sqrt(1 / effectifs[i] + 1 / effectifs[j]) for i, j in paires]
        )
        stats = differences / (sigma[:, None] * facteur[None, :])

        gagnant = np.argmax(np.abs(stats), axis=1)
        lignes = np.arange(n)
        tous_max_t.append(np.abs(stats[lignes, gagnant]))
        tous_d.append(np.abs(differences[lignes, gagnant]))
        tous_cible.append(gagnant == indice_cible)

    return (
        np.concatenate(tous_max_t),
        np.concatenate(tous_d),
        np.concatenate(tous_cible),
    )


def route_2_permutation(ecarts, niveaux_R, tirages, generateur):
    """Permute les ecarts sur les etiquettes R reellement observees.

    Ne suppose rien : ni normalite, ni ecart-type commun, ni independance des
    tailles de cellules vis-a-vis du plan. C'est le nul exact de la procedure
    qui a produit le tableau.
    """
    niveaux = sorted(set(niveaux_R), reverse=True)
    etiquettes = np.asarray(niveaux_R)
    paires = [(a, b) for i, a in enumerate(niveaux) for b in niveaux[i + 1:]]
    indice_cible = paires.index((25, 24))

    max_t = np.empty(tirages)
    d_gagnant = np.empty(tirages)
    cible = np.empty(tirages, dtype=bool)
    for tirage in range(tirages):
        melange = generateur.permutation(ecarts)
        differences, stats, _ = contraste_t(melange, etiquettes, niveaux)
        gagnant = np.argmax(np.abs(stats))
        max_t[tirage] = np.abs(stats[gagnant])
        d_gagnant[tirage] = np.abs(differences[gagnant])
        cible[tirage] = gagnant == indice_cible
    return max_t, d_gagnant, cible


def route_3_effectifs_aleatoires(effectifs, ecart_type, tirages, generateur):
    """Retire les tailles de cellules a chaque tirage.

    R est une sortie du run, pas un reglage. Les effectifs 8/30/53/47/12 sont
    eux-memes une realisation. Fixer les cellules sous-estime peut-etre la
    dispersion du maximum : cette route le verifie.
    """
    effectifs = np.asarray(effectifs)
    total = effectifs.sum()
    k = len(effectifs)
    probabilites = effectifs / total
    paires = [(i, j) for i in range(k) for j in range(i + 1, k)]

    max_t = np.empty(tirages)
    for tirage in range(tirages):
        tirés = generateur.multinomial(total, probabilites)
        if (tirés < 2).any():
            # une cellule vide ou singleton ne fournit pas de contraste lisible ;
            # on la retire du jeu de contrastes plutot que de rejeter le tirage
            actifs = [i for i in range(k) if tirés[i] >= 2]
        else:
            actifs = list(range(k))
        moyennes = np.empty(k)
        somme_carres = 0.0
        ddl = 0
        for i in range(k):
            if tirés[i] == 0:
                moyennes[i] = np.nan
                continue
            bloc = generateur.normal(0.0, ecart_type, tirés[i])
            moyennes[i] = bloc.mean()
            somme_carres += ((bloc - bloc.mean()) ** 2).sum()
            ddl += tirés[i] - 1
        sigma = np.sqrt(somme_carres / ddl)
        meilleur = 0.0
        for i, j in paires:
            if i not in actifs or j not in actifs:
                continue
            erreur = sigma * np.sqrt(1 / tirés[i] + 1 / tirés[j])
            stat = abs(moyennes[i] - moyennes[j]) / erreur
            meilleur = max(meilleur, stat)
        max_t[tirage] = meilleur
    return max_t


def bout_en_bout(effectifs, ecart_type, erreur_replication, tirages, generateur):
    """Decouverte, selection du maximum, replication independante, mise en commun.

    La decouverte est orientee positive, comme la mienne l'a ete : c'est ce qui
    deplace la loi du groupe sous le nul et rend le nombre mis en commun
    positif en moyenne alors que la verite vaut zero.
    """
    max_t, d_gagnant, _ = route_1_vectorisee(
        effectifs, ecart_type, tirages, generateur
    )
    # erreur type de la decouverte : celle du contraste 25/24, la plus serree,
    # car c'est la cellule ou le maximum tombe le plus souvent
    erreur_decouverte = ecart_type * np.sqrt(1 / effectifs[2] + 1 / effectifs[3])
    replication = generateur.normal(0.0, erreur_replication, tirages)

    poids_d = 1 / erreur_decouverte ** 2
    poids_r = 1 / erreur_replication ** 2
    commun = (d_gagnant * poids_d + replication * poids_r) / (poids_d + poids_r)
    erreur_commune = 1 / np.sqrt(poids_d + poids_r)
    return d_gagnant, replication, commun, erreur_commune, poids_d / (poids_d + poids_r)


def omega_carre(f, ddl1, ddl2):
    """omega carre a partir de F, et l'esperance d'eta carre sous le nul.

    Sous l'hypothese nulle, eta carre suit exactement une loi beta de
    parametres ddl1/2 et ddl2/2, donc son esperance vaut ddl1/(ddl1+ddl2) :
    ce n'est pas une approximation.
    """
    eta = ddl1 * f / (ddl1 * f + ddl2)
    eta_nul = ddl1 / (ddl1 + ddl2)
    total = ddl1 + ddl2 + 1
    omega = ddl1 * (f - 1) / (ddl1 * (f - 1) + total)
    return eta, eta_nul, omega


def main():
    generateur = np.random.default_rng(20260814)

    decouverte = charger(CHEMIN_DECOUVERTE)
    replication = charger(CHEMIN_REPLICATION)
    ecarts_d = np.array([r["ecart"] for r in decouverte["runs"]])
    R_d = np.array([r["R"] for r in decouverte["runs"]])
    ecarts_r = np.array([r["ecart"] for r in replication["runs"]])
    R_r = np.array([r["R"] for r in replication["runs"]])

    print("=" * 72)
    print("0. LE TABLEAU DE DECOUVERTE, RELU SUR LES RUNS")
    print("=" * 72)
    niveaux = [27, 26, 25, 24, 23]
    differences, stats, paires = contraste_t(ecarts_d, R_d, niveaux)
    for (a, b), d, t in zip(paires, differences, stats):
        marque = "  <-- publie" if (a, b) == (25, 24) else ""
        print(f"  R={a} vs R={b}   d = {d:+.5f}   t = {t:+.3f}{marque}")
    ordre = np.argsort(-np.abs(stats))
    print()
    print("  les trois plus grands en valeur absolue :")
    for rang in ordre[:3]:
        a, b = paires[rang]
        print(f"    {rang+1:2d}e paire  R={a}/{b}   |t| = {abs(stats[rang]):.3f}")

    # ecart-type mis en commun, les deux versions
    somme_carres = 0.0
    ddl = 0
    for niveau in niveaux:
        bloc = ecarts_d[R_d == niveau]
        somme_carres += ((bloc - bloc.mean()) ** 2).sum()
        ddl += len(bloc) - 1
    sd_runs = np.sqrt(somme_carres / ddl)
    print()
    print(f"  ecart-type mis en commun, sur les runs : {sd_runs:.6f}")
    print(f"  le sien, lu sur le tableau publie      : 0.012942")

    effectifs = [int((R_d == niveau).sum()) for niveau in niveaux]
    print(f"  effectifs                              : {effectifs}")

    print()
    print("=" * 72)
    print("1. ROUTE PARAMETRIQUE — SA SIMULATION, 400 000 TIRAGES")
    print("=" * 72)
    for etiquette, sd in [("le sien  0.012942", 0.012942), (f"le mien  {sd_runs:.6f}", sd_runs)]:
        gen = np.random.default_rng(1234)
        max_t, d_gagnant, cible = route_1_vectorisee(effectifs, sd, 400_000, gen)
        print(f"\n  ecart-type = {etiquette}")
        print(f"    E[max |t|]                    {max_t.mean():.3f}")
        print(f"    q90                           {np.quantile(max_t, 0.90):.3f}")
        print(f"    P(max |t| >= 2.40)            {(max_t >= 2.40).mean():.4f}")
        print(f"    P(max |t| >= 2.53)            {(max_t >= 2.53).mean():.4f}")
        print(f"    E[|d| du gagnant]             {d_gagnant.mean():.5f}")
        print(f"    P(gagnant = paire 25/24)      {cible.mean():.4f}")
        print(f"    E[|d| | gagnant = 25/24]      {d_gagnant[cible].mean():.5f}")

    print()
    print("=" * 72)
    print("2. ROUTE PAR PERMUTATION — LE PLAN REEL, AUCUNE HYPOTHESE")
    print("=" * 72)
    gen = np.random.default_rng(555)
    max_t_p, d_p, cible_p = route_2_permutation(ecarts_d, R_d, 40_000, gen)
    t_observe = abs(stats[paires.index((25, 24))])
    print(f"    E[max |t|]                    {max_t_p.mean():.3f}")
    print(f"    q90                           {np.quantile(max_t_p, 0.90):.3f}")
    print(f"    P(max |t| >= 2.40)            {(max_t_p >= 2.40).mean():.4f}")
    print(f"    P(max |t| >= {t_observe:.2f})            {(max_t_p >= t_observe).mean():.4f}")
    print(f"    E[|d| du gagnant]             {d_p.mean():.5f}")
    print(f"    P(gagnant = paire 25/24)      {cible_p.mean():.4f}")
    print(f"    E[|d| | gagnant = 25/24]      {d_p[cible_p].mean():.5f}")
    print()
    print("    p nominal du contraste seul, sans correction :")
    from scipy import stats as st
    p_nominal = 2 * (1 - st.t.cdf(t_observe, ddl))
    print(f"      |t| = {t_observe:.3f} a {ddl} ddl  ->  p = {p_nominal:.4f}")

    print()
    print("=" * 72)
    print("3. ROUTE A EFFECTIFS ALEATOIRES — R EST UNE SORTIE")
    print("=" * 72)
    gen = np.random.default_rng(999)
    max_t_a = route_3_effectifs_aleatoires(effectifs, sd_runs, 40_000, gen)
    print(f"    E[max |t|]                    {max_t_a.mean():.3f}")
    print(f"    q90                           {np.quantile(max_t_a, 0.90):.3f}")
    print(f"    P(max |t| >= 2.40)            {(max_t_a >= 2.40).mean():.4f}")
    print(f"    P(max |t| >= 2.53)            {(max_t_a >= 2.53).mean():.4f}")

    print()
    print("=" * 72)
    print("4. LE JEU DE SELECTION EST-IL VRAIMENT DE DIX ?")
    print("=" * 72)
    betas = sorted(set(r["beta"] for r in decouverte["runs"]))
    beta_d = np.array([r["beta"] for r in decouverte["runs"]])
    _, stats_beta, paires_beta = contraste_t(ecarts_d, beta_d, betas)
    print(f"    le tableau beta offre {len(paires_beta)} contrastes de plus,")
    print(f"    lus le meme jour, sur les memes 150 runs.")
    print(f"    max |t| observe sur la ligne beta : {np.abs(stats_beta).max():.3f}")
    gen = np.random.default_rng(777)
    depassements = 0
    for _ in range(40_000):
        melange = gen.permutation(ecarts_d)
        _, sR, _ = contraste_t(melange, R_d, niveaux)
        _, sB, _ = contraste_t(melange, beta_d, betas)
        if max(np.abs(sR).max(), np.abs(sB).max()) >= t_observe:
            depassements += 1
    print(f"    P(max des 20 |t| >= {t_observe:.2f})    {depassements/40_000:.4f}")

    print()
    print("=" * 72)
    print("5. BOUT EN BOUT — DECOUVRIR, SELECTIONNER, REPLIQUER, METTRE EN COMMUN")
    print("=" * 72)
    # ce que valent reellement mes deux moities independantes
    d_obs = differences[paires.index((25, 24))]
    err_d = sd_runs * np.sqrt(1 / effectifs[2] + 1 / effectifs[3])
    bloc_25 = ecarts_r[R_r == 25]
    bloc_24 = ecarts_r[R_r == 24]
    d_rep = bloc_25.mean() - bloc_24.mean()
    somme_carres_r = 0.0
    ddl_r = 0
    for niveau in sorted(set(R_r)):
        bloc = ecarts_r[R_r == niveau]
        if len(bloc) < 2:
            continue
        somme_carres_r += ((bloc - bloc.mean()) ** 2).sum()
        ddl_r += len(bloc) - 1
    sd_r = np.sqrt(somme_carres_r / ddl_r)
    err_r = sd_r * np.sqrt(1 / len(bloc_25) + 1 / len(bloc_24))
    print(f"    decouverte   d = {d_obs:+.5f}   SE = {err_d:.5f}   t = {d_obs/err_d:+.2f}")
    print(f"    replication  d = {d_rep:+.5f}   SE = {err_r:.5f}   t = {d_rep/err_r:+.2f}")
    poids_d = 1 / err_d ** 2
    poids_r = 1 / err_r ** 2
    commun_obs = (d_obs * poids_d + d_rep * poids_r) / (poids_d + poids_r)
    err_commune = 1 / np.sqrt(poids_d + poids_r)
    print(f"    mis en commun par variance inverse :")
    print(f"      d = {commun_obs:+.5f}   SE = {err_commune:.5f}   "
          f"poids decouverte = {poids_d/(poids_d+poids_r):.1%}")

    # ce que j'ai publie, obtenu en repoolant les 210 runs bruts
    ecarts_tous = np.concatenate([ecarts_d, ecarts_r])
    R_tous = np.concatenate([R_d, R_r])
    niveaux_tous = sorted(set(R_tous.tolist()), reverse=True)
    d_tous, s_tous, p_tous = contraste_t(ecarts_tous, R_tous, niveaux_tous)
    idx = p_tous.index((25, 24))
    print(f"    les 210 runs repooles bruts :")
    print(f"      d = {d_tous[idx]:+.5f}   t = {s_tous[idx]:+.2f}   (c'est mon +0.0028)")

    gen = np.random.default_rng(31415)
    d_g, rep, commun, err_c, part = bout_en_bout(
        effectifs, sd_runs, err_r, 200_000, gen
    )
    print()
    print("                              nul pur      moi")
    print(f"    d de decouverte          {d_g.mean():+.5f}    {d_obs:+.5f}")
    print(f"    d de replication         {rep.mean():+.5f}    {d_rep:+.5f}")
    print(f"    d mis en commun          {commun.mean():+.5f}    {commun_obs:+.5f}")
    print(f"    SE mise en commun         {err_c:.5f}     {err_commune:.5f}")
    print(f"    part de la decouverte    {part:.1%}")
    print(f"    P(commun >= {commun_obs:.5f})     {(commun >= commun_obs).mean():.3f}")
    print(f"    P(commun >= 0.0028)      {(commun >= 0.0028).mean():.3f}")

    print()
    print("=" * 72)
    print("6. OMEGA CARRE — AUCUN TIRAGE, MES PROPRES F")
    print("=" * 72)
    print("    F           eta^2    E[eta^2] si R n'explique rien    omega^2")
    for f, d1, d2 in [(1.552, 4, 145), (0.419, 6, 203)]:
        eta, eta_nul, omega = omega_carre(f, d1, d2)
        print(f"    F({d1},{d2}) = {f:.3f}   {eta*100:5.2f} %          "
              f"{eta_nul*100:5.2f} %              {omega*100:+6.2f} %")
    print()
    print("    verification que la loi est exacte et non approchee :")
    gen = np.random.default_rng(2718)
    for d1, d2 in [(4, 145), (6, 203)]:
        eta_sim = gen.beta(d1 / 2, d2 / 2, 400_000)
        print(f"      ddl ({d1},{d2}) : E[eta^2] simule {eta_sim.mean()*100:.3f} %  "
              f"contre ddl1/(ddl1+ddl2) = {d1/(d1+d2)*100:.3f} %")


if __name__ == "__main__":
    main()
