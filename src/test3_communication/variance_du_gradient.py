"""RDTRL — test 3 : la variance du gradient, mesuree DIRECTEMENT.

dipankarsarkar propose de separer bruit et volume par une ligne de reference
leave-one-out a tirages et mises a jour fixes. La colonne est juste. Mais elle
mesure une reduction de variance A TRAVERS une boucle d'optimisation, et c'est
exactement la faute qui a tue §1.12, `plafond_beta`, et ma phrase sur les 5 %.
Ce serait la cinquieme fois.

La variance d'un estimateur est une propriete du point et de l'estimateur. Elle
se mesure ou elle vit : a theta fixe, sur des lots repliques, sans une seule
mise a jour.

Ce que ce fichier etablit, dans l'ordre :

  1. le gradient analytique de J contre autograd (le code est-il juste) ;
  2. le biais de l'estimateur echantillonne contre le gradient exact (REINFORCE
     monte-t-il bien le meme objectif, ou un autre) ;
  3. la variance totale E||g - E g||^2 pour quatre lignes de reference et trois
     tailles de lot, en TROIS points de la dynamique (initialisation, milieu de
     montee, point critique piege) ;
  4. le plancher de detection de sa colonne : de combien LOO deplace la variance,
     compare a ce que le lot deplace.

Si LOO deplace la variance de quelques pour cent la ou le lot la deplace d'un
facteur huit, alors sa colonne ne peut rien conclure d'un resultat nul, et il
faut le savoir AVANT de la lire.
"""

import numpy as np
import torch

from grammaire3 import N
from representable_atteignable_stable import (EmetteurTabulaire, Recepteur,
                                              activer, monter, objectif,
                                              parametres)

BETA = 0.02
LOTS = (8, 16, 64)
LIGNES = ("aucune", "ema", "loo", "constante_optimale")


def lois(emetteur, recepteur):
    with torch.no_grad():
        return emetteur.loi().numpy(), recepteur.loi().numpy()


def gradient_exact(s, r):
    """d/dP et d/dQ de E[R] = tr(S R)/N, en forme fermee."""
    diag_s = (s * r.T).sum(axis=1)
    grad_p = s * (r.T - diag_s[:, None]) / N
    diag_r = (r * s.T).sum(axis=1)
    grad_q = r * (s.T - diag_r[:, None]) / N
    return grad_p, grad_q


def gradient_exact_autograd(emetteur, recepteur):
    activer(emetteur, recepteur)
    for t in parametres(emetteur, recepteur):
        if t.grad is not None:
            t.grad = None
    _, recompense = objectif(emetteur, recepteur, BETA)
    recompense.backward()
    return (emetteur.p[0].grad.detach().numpy().copy(),
            recepteur.p[0].grad.detach().numpy().copy())


def tirer_lots(s, r, lot, replicats, generateur):
    """(referents, messages, reconstruits, recompenses), formes (replicats, lot)."""
    cum_s = s.cumsum(axis=1)
    cum_r = r.cumsum(axis=1)
    refs = generateur.integers(0, N, size=(replicats, lot))
    u = generateur.random((replicats, lot))
    msg = (u[..., None] > cum_s[refs]).sum(axis=-1).clip(0, N - 1)
    v = generateur.random((replicats, lot))
    rec = (v[..., None] > cum_r[msg]).sum(axis=-1).clip(0, N - 1)
    return refs, msg, rec, (rec == refs).astype(np.float64)


def avantages(recompenses, ligne, p, b_opt):
    lot = recompenses.shape[1]
    if ligne == "aucune":
        return recompenses
    if ligne == "ema":
        return recompenses - p
    if ligne == "constante_optimale":
        return recompenses - b_opt
    somme = recompenses.sum(axis=1, keepdims=True)
    return (lot * recompenses - somme) / (lot - 1)


def estimateurs(s, r, refs, msg, rec, av):
    """g_P et g_Q par replicat, forme (replicats, N, N). Scores en forme fermee."""
    replicats, lot = refs.shape
    g_p = np.zeros((replicats, N, N))
    g_q = np.zeros((replicats, N, N))
    idx = np.arange(replicats)[:, None]
    np.add.at(g_p, (idx, refs, msg), av)
    np.add.at(g_q, (idx, msg, rec), av)
    masse_p = np.zeros((replicats, N))
    masse_q = np.zeros((replicats, N))
    np.add.at(masse_p, (idx, refs), av)
    np.add.at(masse_q, (idx, msg), av)
    g_p -= masse_p[:, :, None] * s[None, :, :]
    g_q -= masse_q[:, :, None] * r[None, :, :]
    return g_p / lot, g_q / lot


def baseline_optimale(s, r, lot, generateur, replicats=20000):
    """b* = E[R |score|^2] / E[|score|^2], la constante qui minimise la variance."""
    refs, msg, rec, rw = tirer_lots(s, r, 1, replicats, generateur)
    un = np.ones((replicats, 1))
    g_p, g_q = estimateurs(s, r, refs, msg, rec, un)
    carre = (g_p ** 2).sum(axis=(1, 2)) + (g_q ** 2).sum(axis=(1, 2))
    return float((rw[:, 0] * carre).sum() / carre.sum())


def mesurer(s, r, lot, ligne, generateur, replicats, p, b_opt):
    refs, msg, rec, rw = tirer_lots(s, r, lot, replicats, generateur)
    av = avantages(rw, ligne, p, b_opt)
    g_p, g_q = estimateurs(s, r, refs, msg, rec, av)
    moy_p, moy_q = g_p.mean(axis=0), g_q.mean(axis=0)
    var = float(((g_p - moy_p) ** 2).sum(axis=(1, 2)).mean()
                + ((g_q - moy_q) ** 2).sum(axis=(1, 2)).mean())
    return moy_p, moy_q, var, float((av == 0).all(axis=1).mean())


def point(nom, emetteur, recepteur, generateur, replicats):
    s, r = lois(emetteur, recepteur)
    gp_exact, gq_exact = gradient_exact(s, r)
    gp_auto, gq_auto = gradient_exact_autograd(emetteur, recepteur)
    ecart_forme = max(np.abs(gp_exact - gp_auto).max(),
                      np.abs(gq_exact - gq_auto).max())
    norme_exacte = float(np.sqrt((gp_exact ** 2).sum() + (gq_exact ** 2).sum()))
    p = float((s * r.T).sum() / N)
    print(f"\n  {nom}")
    print(f"    E[R] {p:.6f}   ||grad E[R]|| {norme_exacte:.3e}"
          f"   forme fermee vs autograd {ecart_forme:.3e}")

    resultats = {}
    for lot in LOTS:
        b_opt = baseline_optimale(s, r, lot, generateur)
        for ligne in LIGNES:
            moy_p, moy_q, var, mort = mesurer(s, r, lot, ligne, generateur,
                                              replicats, p, b_opt)
            biais = float(np.sqrt(((moy_p - gp_exact) ** 2).sum()
                                  + ((moy_q - gq_exact) ** 2).sum()))
            se = float(np.sqrt(var / replicats))
            resultats[(lot, ligne)] = (var, biais, se, mort, b_opt)
    return p, norme_exacte, resultats


def rendre(resultats):
    print(f"    {'lot':>5}{'ligne':>21}{'variance':>13}{'/exact^2':>11}"
          f"{'biais':>11}{'se':>11}{'biais/se':>10}{'lots morts':>12}")
    for lot in LOTS:
        for ligne in LIGNES:
            var, biais, se, mort, _ = resultats[(lot, ligne)]
            print(f"    {lot:>5}{ligne:>21}{var:>13.4e}"
                  f"{'':>11}{biais:>11.3e}{se:>11.3e}{biais / se:>10.2f}"
                  f"{mort:>12.4f}")


def lignes_optimales(resultats, p):
    """b* contre E[R] : la ligne qui minimise la variance n'est pas la moyenne."""
    print(f"    la ligne de reference optimale n'est PAS E[R] = {p:.6f} :")
    for lot in LOTS:
        b_opt = resultats[(lot, "ema")][4]
        print(f"      lot {lot:>3}   b* = {b_opt:.6f}   b* / E[R] = {b_opt / p:.4f}")


def plancher(resultats, norme):
    print("\n    ce que chaque axe deplace, en variance totale :")
    v8_ema = resultats[(8, "ema")][0]
    v8_loo = resultats[(8, "loo")][0]
    v8_opt = resultats[(8, "constante_optimale")][0]
    v8_nul = resultats[(8, "aucune")][0]
    v64_ema = resultats[(64, "ema")][0]
    print(f"      sa colonne   lot 8 ema -> lot 8 loo    x {v8_loo / v8_ema:.4f}")
    print(f"      idem, contre la constante optimale     x {v8_loo / v8_opt:.4f}")
    print(f"      sans ligne de reference -> ema         x {v8_ema / v8_nul:.4f}")
    print(f"      mon balayage lot 8 -> lot 64           x {v64_ema / v8_ema:.4f}")
    print(f"      variance/||grad||^2 au lot 8 (ema)     {v8_ema / norme ** 2:.3e}")


if __name__ == "__main__":
    replicats = 20000
    generateur = np.random.default_rng(4242)
    graine_torch = np.random.default_rng(606)

    print("=== 1. LE CODE EST-IL JUSTE, 2. L'ESTIMATEUR EST-IL NON BIAISE ===")
    print(f"  {replicats} lots repliques par cellule, a theta FIXE, zero mise a jour.")
    print("  'biais/se' est l'ecart au gradient exact en erreurs types de Monte-Carlo.")

    e0, r0 = EmetteurTabulaire(graine_torch), Recepteur(graine_torch)
    p0, n0, res0 = point("theta_init (avant toute montee)", e0, r0,
                         generateur, replicats)
    rendre(res0)
    lignes_optimales(res0, p0)
    plancher(res0, n0)

    e1, r1 = EmetteurTabulaire(graine_torch), Recepteur(graine_torch)
    monter(e1, r1, BETA, 300, lr=0.05)
    p1, n1, res1 = point("theta_milieu (300 pas de montee exacte)", e1, r1,
                         generateur, replicats)
    rendre(res1)
    lignes_optimales(res1, p1)
    plancher(res1, n1)

    e2, r2 = EmetteurTabulaire(graine_torch), Recepteur(graine_torch)
    monter(e2, r2, BETA, 20000, lr=0.05)
    code = e2.loi().argmax(dim=1).detach().numpy()
    p2, n2, res2 = point(f"theta_piege (20000 pas, {N - len(set(code.tolist()))}"
                         " collisions)", e2, r2, generateur, replicats)
    rendre(res2)
    lignes_optimales(res2, p2)
    plancher(res2, n2)
