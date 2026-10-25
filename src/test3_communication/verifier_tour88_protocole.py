"""Tour 88 : le plancher de protocole est-il le meme a chaque lr ? Seule compte la part qui differe d'un lr a l'autre.

Il dit : un biais de protocole commun a tous les lr s'annule dans tout exposant ; la difference bas - haut s'appuie surtout sur lr 0,2 (poids +1,35 dans le segment bas, -2,54 dans le haut, soit +3,89) ; abaisser le niveau de lr 0,2 de 0,10 % fait disparaitre la marche (0,0198 -> 0,0163 et 0,0118 -> 0,0163) et fait tomber bas - haut de 0,006 a 0,002 ; il demande de varier la fenetre et le pas de balayage aux cinq lr et de rapporter le decalage de chaque exposant local, et de lire lr 0,225.

Modes :
  arithmetique : ses chiffres a partir des niveaux du tour 87, et l'influence de +0,1 % a chaque lr sur chaque exposant local et sur bas - haut (aucune donnee lue)
  variantes    : lectures directes (60 salves par depart, memes salves, dernier franchissement) a pas 1 et 6 et a fenetre 210 contre la reference (pas 3, fenetre 140) ; decalage par lr et ses consequences
  grille       : niveaux d'enregistrement entier (C' x moyenne de 1/rho) sur la grille 0,1 a 0,3 de pas 0,025, ecart a la corde 0,15-0,25
  decalage_rho : niveaux d'enregistrement avec rho pris 1 pas ou 30 pas avant le groupe
"""

import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from verifier_tour84_coupures import droite  # noqa: E402

D = "D:/tmp"
LRS5 = [0.1, 0.15, 0.2, 0.25, 0.3]
MOTIF = re.compile(r"salve (\d+) : calme avant (\d+) pas ; duree (\d+) pas ; lr p_z ([0-9.e+]+) ; sqrt\(v_e3\) ([0-9.e+\-]+) ; avance ([0-9.\-]+) pas")
# niveaux d'enregistrement entier du tour 87 (moyenne de dix departs) et leur incertitude (entre departs + bloc, quadrature, en %)
NIVEAUX87 = np.array([3.24924e6, 3.27187e6, 3.29052e6, 3.29919e6, 3.30658e6])
SE87 = np.array([math.hypot(0.003, 0.005), math.hypot(0.002, 0.010), math.hypot(0.006, 0.018), math.hypot(0.010, 0.028), math.hypot(0.011, 0.033)]) / 100


def exposants_locaux(lrs, niv):
    return [math.log(niv[i + 1] / niv[i]) / math.log(lrs[i + 1] / lrs[i]) for i in range(len(lrs) - 1)]


def segments(lrs, niv, se):
    """(exposant bas 0,1-0,2, exposant haut 0,2-0,3, bas - haut) ; se : incertitude relative par point (poids 1/se^2) ou None pour non pondere."""
    x = np.log(lrs); y = np.log(niv)
    s = np.ones(len(lrs)) if se is None else se
    ib = [0, 1, 2]; ih = [2, 3, 4]
    eb = droite(x[ib], y[ib], s[ib])[0]; eh = droite(x[ih], y[ih], s[ih])[0]
    return eb, eh, eb - eh


def arithmetique():
    lrs = np.array(LRS5)
    x = np.log(lrs)
    for nom, ix in (("bas (0,1 ; 0,15 ; 0,2)", [0, 1, 2]), ("haut (0,2 ; 0,25 ; 0,3)", [2, 3, 4])):
        xs = x[ix]; w = (xs - xs.mean()) / ((xs - xs.mean()) ** 2).sum()
        print(f"poids non pondere des trois points dans la pente {nom} : {np.round(w, 3)}")
    print(f"  lr 0,2 dans bas - haut : {1.348:+.2f} - ({-2.54:+.2f}) = {1.348 + 2.54:+.2f} (lui : +3,89)")
    e = exposants_locaux(lrs, NIVEAUX87)
    print("exposants locaux a partir des niveaux du tour 87 : " + " ".join(f"{v:+.4f}" for v in e))
    niv = NIVEAUX87.copy(); niv[2] *= math.exp(-0.001)
    e2 = exposants_locaux(lrs, niv)
    print("apres avoir abaisse le niveau de lr 0,2 de 0,10 % : " + " ".join(f"{v:+.4f}" for v in e2))
    for nom, se in (("non pondere", None), ("pondere par l'incertitude du tour 87 (bruit + entre departs)", SE87), ("pondere avec plancher 0,1 %", np.hypot(SE87, 0.001))):
        eb, eh, d0 = segments(lrs, NIVEAUX87, se)
        eb2, eh2, d1 = segments(lrs, niv, se)
        print(f"  bas - haut, {nom} : {d0:+.4f} (bas {eb:+.4f}, haut {eh:+.4f}) ; apres -0,10 % a lr 0,2 : {d1:+.4f}")
    print("influence de +0,10 % du niveau de chaque lr sur chaque exposant local et sur bas - haut :")
    for i, lr in enumerate(LRS5):
        n3 = NIVEAUX87.copy(); n3[i] *= math.exp(0.001)
        el = exposants_locaux(lrs, n3)
        ligne = f"   lr {lr}: locaux " + " ".join(f"{a - b:+.4f}" for a, b in zip(el, e))
        for nom, se in (("non pondere", None), ("pondere", SE87)):
            ligne += f" ; bas - haut ({nom}) {segments(lrs, n3, se)[2] - segments(lrs, NIVEAUX87, se)[2]:+.4f}"
        print(ligne)
    # l'ecart de lr 0,2 a la corde 0,15-0,25
    ec = math.log(NIVEAUX87[1]) + math.log(NIVEAUX87[3] / NIVEAUX87[1]) * math.log(0.2 / 0.15) / math.log(0.25 / 0.15)
    print(f"niveau de lr 0,2 au-dessus de la corde 0,15-0,25 (log-log) : {100 * (math.log(NIVEAUX87[2]) - ec):+.3f} %")


def lire_log(chemin):
    """dict g -> (lecture, avance) pour un journal de lectures."""
    try:
        t = open(chemin, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return {}
    return {int(g): (float(r), float(av)) for g, c, d, r, sv, av in MOTIF.findall(t)}


def decalages_exposants(delta_pct):
    """delta_pct : decalage du niveau (en %) a chaque lr de LRS5 ; retourne les decalages des exposants locaux et de bas - haut (non pondere, pondere)."""
    lrs = np.array(LRS5)
    niv = NIVEAUX87 * np.exp(np.array(delta_pct) / 100)
    e0 = exposants_locaux(lrs, NIVEAUX87); e1 = exposants_locaux(lrs, niv)
    out = [b - a for a, b in zip(e0, e1)]
    dh = []
    for se in (None, SE87):
        dh.append(segments(lrs, niv, se)[2] - segments(lrs, NIVEAUX87, se)[2])
    return out, dh


def variantes():
    base = {lr: {k: lire_log(f"{D}/t87d_alea_lr{lr}_k{k}.log") for k in range(3)} for lr in LRS5}
    resume = {}
    for nom in ("pas1", "pas6", "fen210"):
        print(f"===== variante {nom} contre la reference (pas 3, fenetre 140), memes salves, dernier franchissement")
        deltas = []
        for lr in LRS5:
            tous = []; par_depart = []; ident = 0; n = 0; plus_de_manquantes = 0
            for k in range(3):
                v = lire_log(f"{D}/t88_{nom}_alea_lr{lr}_k{k}.log")
                b = base[lr][k]
                communs = sorted(set(v) & set(b))
                plus_de_manquantes += len(b) - len(communs)
                d = np.array([100 * (v[g][0] / b[g][0] - 1) for g in communs])
                if len(d):
                    tous.extend(d); par_depart.append(d.mean())
                    ident += int((np.abs(d) < 0.01).sum()); n += len(d)
            if not tous:
                print(f"   lr {lr}: pas de donnees")
                deltas.append(float("nan"))
                continue
            tous = np.array(tous)
            deltas.append(float(tous.mean()))
            print(f"   lr {lr}: {n} salves appariees (manquantes {plus_de_manquantes}) ; decalage moyen {tous.mean():+.5f} % (sd salve a salve {tous.std(ddof=1):.5f}, ET {tous.std(ddof=1) / math.sqrt(len(tous)):.5f}) ; par depart " + " ".join(f"{x:+.5f}" for x in par_depart) + f" ; identiques (< 0,01 %) {ident}/{n} ; max |decalage| {np.abs(tous).max():.4f} %")
        resume[nom] = deltas
        if not any(math.isnan(x) for x in deltas):
            ee, dh = decalages_exposants(deltas)
            print(f"   decalage des exposants locaux (0,1-0,15 ; 0,15-0,2 ; 0,2-0,25 ; 0,25-0,3) : " + " ".join(f"{x:+.5f}" for x in ee) + f" ; bas - haut : non pondere {dh[0]:+.5f}, pondere {dh[1]:+.5f}")
            chord = 0.437 * deltas[1] + 0.563 * deltas[3]
            print(f"   lr 0,2 par rapport a ses voisins (decalage moins corde 0,15-0,25) : {deltas[2] - chord:+.5f} %")
    if "pas1" in resume and "pas6" in resume:
        print("loi du pas : decalage(pas1) / (1 - 9) contre decalage(pas6) / (36 - 9), par lr (si loi en pas^2, les deux coincident)")
        for lr, d1, d6 in zip(LRS5, resume["pas1"], resume["pas6"]):
            print(f"   lr {lr}: {d1 / -8:+.6f} contre {d6 / 27:+.6f} % par pas^2 ; rapport d6/d1 {d6 / d1 if d1 else float('nan'):+.2f} (loi en pas^2 : -3,375)")


def niveaux_grille(lrs_prefixes, ks_de):
    """niveaux d'enregistrement entier (C' x moyenne de 1/rho au decalage 30) par lr : (niveau moyen, ET entre departs, bruit de bloc, nombre de departs)."""
    res = {}
    for lr, prefixe in lrs_prefixes:
        niv, sb = [], []
        for k in ks_de(lr):
            ech = lire_log(f"{D}/{prefixe}_alea_lr{lr}_k{k}.log")
            try:
                z = np.load(f"{D}/t86_lr{lr}_k{k}.npz")
            except FileNotFoundError:
                continue
            if len(ech) < 20:
                continue
            rho = z["RHO"][:, 1]
            cprim = np.median([r * rho[g] for g, (r, av) in ech.items()])
            inv = 1 / rho
            x = inv[9:][np.isfinite(inv[9:])]
            niv.append(cprim * x.mean())
            nb = len(x) // 100
            bl = x[:nb * 100].reshape(nb, 100).mean(axis=1)
            sb.append(bl.std(ddof=1) / bl.mean() / math.sqrt(nb))
        if niv:
            niv = np.array(niv)
            res[lr] = (niv.mean(), niv.std(ddof=1) / math.sqrt(len(niv)) / niv.mean() if len(niv) > 1 else 0.0, float(np.mean(sb)), len(niv))
    return res


def grille():
    anciens = [(lr, "t87d") for lr in LRS5]
    nouveaux = [(lr, "t88_ref") for lr in (0.125, 0.175, 0.225, 0.275)]
    res = niveaux_grille(anciens + nouveaux, lambda lr: range(10) if lr in LRS5 else range(5))
    lrs = sorted(res)
    print("niveaux d'enregistrement entier (C' x moyenne de 1/rho, salves >= 9) : lr, niveau, ET entre departs (%), bruit de bloc (%), departs")
    for lr in lrs:
        m, sb, bl, n = res[lr]
        print(f"   lr {lr}: {m:.5e}  {100 * sb:.4f}  {100 * bl:.4f}  {n}")
    se = {lr: math.hypot(res[lr][1], res[lr][2]) for lr in lrs}
    print("exposants locaux sur la grille (ET : entre departs + bloc, en quadrature) :")
    for a, b in zip(lrs[:-1], lrs[1:]):
        e = math.log(res[b][0] / res[a][0]) / math.log(b / a)
        s = math.hypot(se[a], se[b]) / math.log(b / a)
        print(f"   {a} -> {b} : {e:+.4f} +- {s:.4f}")
    print("ecart de chaque lr a la corde log-log 0,15-0,25 (en %, +- ET) ; prediction : courbure lisse ~ +0,085 (0,175), +0,100 (0,2), +0,067 (0,225) ; decalage isole de lr 0,2 : ~0 sauf lr 0,2")
    l15, l25 = res[0.15][0], res[0.25][0]
    for lr in lrs:
        u = math.log(lr / 0.15) / math.log(0.25 / 0.15)
        ch = math.log(l15) * (1 - u) + math.log(l25) * u
        ec = 100 * (math.log(res[lr][0]) - ch)
        s = 100 * math.hypot(se[lr], math.hypot((1 - u) * se[0.15], u * se[0.25])) if lr not in (0.15, 0.25) else float("nan")
        print(f"   lr {lr}: {ec:+.4f} % +- {s:.4f}")
    # ajustement lisse (polynome de degre 2 et 3 en ln lr) et residus
    x = np.log(lrs); y = np.log([res[l][0] for l in lrs]); w = 1 / np.array([se[l] for l in lrs])
    for deg in (2, 3):
        c = np.polyfit(x, y, deg, w=w)
        r = 100 * (y - np.polyval(c, x))
        chi2 = float(((r / 100 * w) ** 2).sum())
        print(f"ajustement de degre {deg} en ln lr, pondere : chi2 {chi2:.1f} pour {len(x) - deg - 1} ddl ; residus (%) " + " ".join(f"{l}:{v:+.3f}" for l, v in zip(lrs, r)))
    # ajustement lisse sans lr 0,2 : le niveau de lr 0,2 est-il au-dessus de la courbe de ses voisins ?
    for deg in (2, 3):
        keep = [i for i, l in enumerate(lrs) if l != 0.2]
        c = np.polyfit(x[keep], y[keep], deg, w=w[keep])
        i2 = lrs.index(0.2)
        print(f"degre {deg} sans lr 0,2 : lr 0,2 au-dessus de la courbe de ses voisins de {100 * (y[i2] - np.polyval(c, x[i2])):+.4f} % (ET du niveau {100 * se[0.2]:.4f} %)")


def decomposition():
    """niveau = C'(lr) x M(lr) ; exposants locaux de chacun sur la grille."""
    lrp = [(lr, "t87d") for lr in LRS5] + [(lr, "t88_ref") for lr in (0.125, 0.175, 0.225, 0.275)]
    Cs, Ms = {}, {}
    for lr, prefixe in lrp:
        cs, ms = [], []
        for k in (range(10) if lr in LRS5 else range(5)):
            ech = lire_log(f"{D}/{prefixe}_alea_lr{lr}_k{k}.log")
            try:
                z = np.load(f"{D}/t86_lr{lr}_k{k}.npz")
            except FileNotFoundError:
                continue
            if len(ech) < 20:
                continue
            rho = z["RHO"][:, 1]
            cs.append(np.median([r * rho[g] for g, (r, av) in ech.items()]))
            x = (1 / rho)[9:]
            ms.append(x[np.isfinite(x)].mean())
        Cs[lr] = (np.mean(cs), np.std(cs, ddof=1) / math.sqrt(len(cs)) / np.mean(cs)); Ms[lr] = (np.mean(ms), np.std(ms, ddof=1) / math.sqrt(len(ms)) / np.mean(ms))
    lrs = sorted(Cs)
    print("exposants locaux (grille 0,025) de C'(lr) (seul facteur qui utilise le balayage), de M(lr) = moyenne de 1/rho (aucun balayage) et du niveau :")
    ec, em = [], []
    for a, b in zip(lrs[:-1], lrs[1:]):
        l = math.log(b / a)
        c = math.log(Cs[b][0] / Cs[a][0]) / l; m = math.log(Ms[b][0] / Ms[a][0]) / l
        ec.append(c); em.append(m)
        print(f"   {a} -> {b} : C' {c:+.5f} ; M {m:+.5f} ; niveau {c + m:+.5f}")
    # marche : exposant moyen 0,15->0,2 contre 0,2->0,3 (par fusion des intervalles)
    def expo(f, a, b):
        return math.log(f[b][0] / f[a][0]) / math.log(b / a)
    for nom, f in (("C'", Cs), ("M", Ms)):
        print(f"   {nom} : exposant 0,15 -> 0,2 : {expo(f, 0.15, 0.2):+.5f} ; 0,2 -> 0,3 : {expo(f, 0.2, 0.3):+.5f} ; difference {expo(f, 0.15, 0.2) - expo(f, 0.2, 0.3):+.5f}")
    n = {lr: (Cs[lr][0] * Ms[lr][0], 0) for lr in lrs}
    print(f"   niveau : exposant 0,15 -> 0,2 : {expo(n, 0.15, 0.2):+.5f} ; 0,2 -> 0,3 : {expo(n, 0.2, 0.3):+.5f} ; difference {expo(n, 0.15, 0.2) - expo(n, 0.2, 0.3):+.5f}")
    print("   structure de l'exposant de C' autour de lr 0,2 (variations d'un intervalle a l'autre) : " + " ".join(f"{b - a:+.5f}" for a, b in zip(ec[:-1], ec[1:])))
    print("   C'(lr) en valeur (tous departs) : " + " ".join(f"{lr}:{Cs[lr][0]:.6e}" for lr in lrs))


def decalage_rho():
    print("niveau d'enregistrement avec rho pris 1 pas ou 30 pas avant le groupe (C' recalculee au meme decalage), dix departs : decalage de niveau (1 pas contre 30 pas), %")
    deltas = []
    for lr in LRS5:
        d = []
        for k in range(10):
            ech = lire_log(f"{D}/t87d_alea_lr{lr}_k{k}.log")
            try:
                z = np.load(f"{D}/t86_lr{lr}_k{k}.npz")
            except FileNotFoundError:
                continue
            niv = []
            for j in (0, 1):
                rho = z["RHO"][:, j]
                cprim = np.median([r * rho[g] for g, (r, av) in ech.items()])
                x = (1 / rho)[9:]
                niv.append(cprim * x[np.isfinite(x)].mean())
            d.append(100 * (niv[0] / niv[1] - 1))
        d = np.array(d)
        deltas.append(float(d.mean()))
        print(f"   lr {lr}: {d.mean():+.5f} % [{d.min():+.5f} .. {d.max():+.5f}] ({len(d)} departs)")
    ee, dh = decalages_exposants(deltas)
    print(f"decalage des exposants locaux : " + " ".join(f"{x:+.5f}" for x in ee) + f" ; bas - haut : non pondere {dh[0]:+.5f}, pondere {dh[1]:+.5f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "variantes":
        variantes()
    elif mode == "grille":
        grille()
    elif mode == "decalage_rho":
        decalage_rho()
    elif mode == "decomposition":
        decomposition()
    elif mode == "arithmetique":
        arithmetique()
