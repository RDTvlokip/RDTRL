"""Tour 92 : le budget d'energie d'une salve contre sa duree L, et ou il se perd (pic ou fin).

Il dit : jusqu'a lr 0,15 la pente de ln I sur L est nulle, donc la salve porte un budget fixe et l'energie par pas baisse deja en 1/L ; ce qui s'allume a lr 0,175 est la chute du budget avec L (-3,7e-3, -8,9e-3, -1,31e-2, -2,04e-2 par pas) ; gamma est le demi-ecart des deux pentes et le rapport l3 / e3 reste a 1,05-1,08 ; cela suppose que le calme ajoute peu a I. Avant de separer pic et longueur : jusqu'a lr 0,15, ln(I / L) sur L est-il a -1/L ? A partir de 0,175, la perte est-elle dans le pic ou dans la fin de la salve ?

Le gradient au carre de chaque pas se relit exactement sur les v enregistres : v_t = B2 v_(t-1) + (1 - B2) q_t, donc q_t = (v_t - B2 v_(t-1)) / (1 - B2), pour e3 (rec[:, 3]^2) et l3 (rec[:, 4]^2). L'injection de cycle est I = v(i2) - B2^T v(i0) = (1 - B2) sum_t B2^(i2 - t) q_t, t de i0 + 1 a i2 ; l'energie de la salve seule est E_b = sum q_t sur le groupe elargi de `pad` pas de chaque cote (pas de v(i0)).

Modes :
  enregistrer <lr> <k> : regenere l'enregistrement (meme protocole que le tour 91, base, x = 1e-4, 300 000 pas) et ecrit D:/tmp/t92_lr<lr>_k<k>.npz (sqrt v_e3, sqrt v_l3, u = l3 - l4)
  analyse <seuil> <pad> : pour chaque lr, par depart, pentes de ln I, ln E_b, ln(E_b / L), ln pic, ln largeur, fractions par tiers, avec controles, part du calme, bootstrap par blocs de 25 salves
  profil : profil moyen de q (e3, l3) selon le pas dans la salve, par tercile de L
"""

import sys

import numpy as np

sys.path.insert(0, ".")

D = "D:/tmp"
B2 = 0.999
GRILLE = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
DEPARTS = range(5)


def groupes(u, seuil):
    """Comme verifier_tour86_longue.groupes_de, avec le seuil en parametre."""
    D2 = u[2:] - 2 * u[1:-1] + u[:-2]
    A = np.abs(D2) / 4.0
    E = np.maximum(A[:-1], A[1:])
    idx = np.nonzero(E > seuil)[0]
    coupe = np.nonzero(np.diff(idx) > 30)[0]
    debuts = np.concatenate([[0], coupe + 1]); fins = np.concatenate([coupe, [len(idx) - 1]])
    return [(int(idx[a]), int(idx[b])) for a, b in zip(debuts, fins)]


def enregistrer(lr, k):
    from verifier_tour90_memoire import enregistrement
    rec, groupes_ref = enregistrement("reduction_numba_tour71_lrvar", lr, k, "1e-4")
    if rec is None:
        print(f"lr {lr} k {k}: effondre")
        return
    np.savez(f"{D}/t92_lr{lr}_k{k}.npz", se=rec[:, 3], sl=rec[:, 4], u=rec[:, 2])
    print(f"lr {lr} k {k}: {len(groupes_ref)} groupes ecrits")


def charger(lr, k):
    try:
        z = np.load(f"{D}/t92_lr{lr}_k{k}.npz")
    except FileNotFoundError:
        return None
    return z["se"] ** 2, z["sl"] ** 2, z["u"]


def pente(x, y):
    return float(np.polyfit(x, y, 1)[0])


def pente_controlee(x, controles, y):
    A = np.column_stack([np.ones(len(y)), x - x.mean()] + [c - c.mean() for c in controles])
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def bootstrap_blocs(x, y, rng, nb=200, bloc=25):
    n = len(x)
    nbl = int(np.ceil(n / bloc))
    pentes = []
    for _ in range(nb):
        starts = rng.integers(0, max(n - bloc, 1), nbl)
        idx = np.concatenate([np.arange(s, min(s + bloc, n)) for s in starts])
        pentes.append(pente(x[idx], y[idx]))
    return float(np.std(pentes))


def un_depart(lr, k, seuil, pad, rng):
    c = charger(lr, k)
    if c is None:
        return None
    ve, vl, u = c
    gr = groupes(u, seuil)
    a = np.array([g[0] for g in gr]); en = np.array([g[1] for g in gr])
    q = {}
    for nom, v in (("e", ve), ("l", vl)):
        qq = np.full(len(v), np.nan)
        qq[1:] = (v[1:] - B2 * v[:-1]) / (1 - B2)
        q[nom] = qq
    q3 = {nom: np.convolve(np.nan_to_num(q[nom]), np.ones(3) / 3, mode="same") for nom in q}
    v0 = {"e": ve, "l": vl}
    g = np.arange(10, len(a) - 1)
    out = {"ident": 0.0, "n": 0}
    res = {nom: {k2: [] for k2 in ("lnI", "lnE", "lnpk", "lnW", "f1", "f2", "f3", "calm", "calm20")} for nom in ("e", "l")}
    L = []; calmA = []; lnv0 = {"e": [], "l": []}; Tl = []
    for j in g:
        s, e, s2 = a[j], en[j], a[j + 1]
        i0, i2 = s - 1, s2 - 1
        T = i2 - i0
        L.append(e - s + 1); calmA.append(s2 - 1 - e); Tl.append(T)
        t = np.arange(i0 + 1, i2 + 1)
        w = B2 ** (i2 - t)
        ws, we = s - pad, e + pad
        lw = we - ws + 1
        b1_, b2_ = ws + int(round(lw / 3)), ws + int(round(2 * lw / 3))
        for nom in ("e", "l"):
            qq = q[nom]
            I_v = v0[nom][i2] - B2 ** T * v0[nom][i0]
            I_q = (1 - B2) * float((w * qq[t]).sum())
            out["ident"] = max(out["ident"], abs(I_q / I_v - 1))
            Eb = float(qq[ws:we + 1].sum())
            pk = float(q3[nom][ws:we + 1].max())
            tt = np.arange(max(ws, i0 + 1), min(we, i2) + 1)
            part = (1 - B2) * float((B2 ** (i2 - tt) * qq[tt]).sum())
            tt20 = np.arange(max(s - 20, i0 + 1), min(e + 20, i2) + 1)
            part20 = (1 - B2) * float((B2 ** (i2 - tt20) * qq[tt20]).sum())
            r = res[nom]
            r["lnI"].append(np.log(I_v) if I_v > 0 else np.nan)
            r["lnE"].append(np.log(Eb) if Eb > 0 else np.nan)
            r["lnpk"].append(np.log(pk) if pk > 0 else np.nan)
            r["lnW"].append(np.log(Eb / pk) if Eb > 0 and pk > 0 else np.nan)
            r["f1"].append(float(qq[ws:b1_].sum()) / Eb); r["f2"].append(float(qq[b1_:b2_].sum()) / Eb); r["f3"].append(float(qq[b2_:we + 1].sum()) / Eb)
            r["calm"].append(1 - part / I_v); r["calm20"].append(1 - part20 / I_v)
            lnv0[nom].append(np.log(v0[nom][i0]))
    L = np.array(L, float); calmA = np.array(calmA, float); Tl = np.array(Tl, float)
    for nom in res:
        for k2 in res[nom]:
            res[nom][k2] = np.array(res[nom][k2])
        lnv0[nom] = np.array(lnv0[nom])
    out["L"] = L; out["calmA"] = calmA; out["T"] = Tl; out["res"] = res; out["lnv0"] = lnv0; out["G"] = len(a)
    out["nombre"] = len(L)
    return out


def analyse(seuil, pad):
    rng = np.random.default_rng(92)
    print(f"seuil {seuil:g}, pad {pad} ; q_t = g_t^2 relu sur les v ; I_q relu a I_v ; E_b = somme de q sur le groupe elargi de {pad} pas ; pic = max de la moyenne de q sur 3 pas ; W = E_b / pic")
    tab = {}
    for lr in GRILLE:
        recs = []
        for k in DEPARTS:
            o = un_depart(lr, k, seuil, pad, rng)
            if o is not None:
                recs.append(o)
        if not recs:
            continue
        mesures = {}
        for nom in ("e", "l"):
            L_list = {n: [] for n in ("sI", "sE", "sEL", "sref", "sPk", "sW", "sf1", "sf2", "sf3", "elast", "sIc", "calm", "calm20", "bI", "bE", "r2semi", "r2log")}
            for o in recs:
                r = o["res"][nom]; L = o["L"]
                ok = np.isfinite(r["lnI"]) & np.isfinite(r["lnE"]) & np.isfinite(r["lnpk"])
                x = L[ok]
                L_list["sI"].append(pente(x, r["lnI"][ok]))
                L_list["sE"].append(pente(x, r["lnE"][ok]))
                L_list["sEL"].append(pente(x, r["lnE"][ok] - np.log(x)))
                L_list["sref"].append(pente(x, -np.log(x)))
                L_list["sPk"].append(pente(x, r["lnpk"][ok]))
                L_list["sW"].append(pente(x, r["lnW"][ok]))
                L_list["sf1"].append(pente(x, r["f1"][ok])); L_list["sf2"].append(pente(x, r["f2"][ok])); L_list["sf3"].append(pente(x, r["f3"][ok]))
                L_list["elast"].append(pente(np.log(x), r["lnE"][ok]))
                p1 = np.polyfit(x, r["lnE"][ok], 1); p2 = np.polyfit(np.log(x), r["lnE"][ok], 1)
                tot = ((r["lnE"][ok] - r["lnE"][ok].mean()) ** 2).sum()
                L_list["r2semi"].append(1 - ((r["lnE"][ok] - np.polyval(p1, x)) ** 2).sum() / tot)
                L_list["r2log"].append(1 - ((r["lnE"][ok] - np.polyval(p2, np.log(x))) ** 2).sum() / tot)
                L_list["sIc"].append(pente_controlee(x, [o["lnv0"]["e"][ok], o["lnv0"]["l"][ok], o["calmA"][ok]], r["lnI"][ok]))
                L_list["calm"].append(np.nanmean(r["calm"])); L_list["calm20"].append(np.nanmean(r["calm20"]))
                L_list["bI"].append(bootstrap_blocs(x, r["lnI"][ok], rng)); L_list["bE"].append(bootstrap_blocs(x, r["lnE"][ok], rng))
            mesures[nom] = {n: np.array(v) for n, v in L_list.items()}
        tab[lr] = (recs, mesures)
    def fmt(v, f="+.3e"):
        return f"{v.mean():{f}} [{v.min():{f}} .. {v.max():{f}}]"
    for lr in tab:
        recs, m = tab[lr]
        Lm = np.mean([o["L"].mean() for o in recs]); sdL = np.mean([o["L"].std() for o in recs])
        cor_ca = np.mean([np.corrcoef(o["L"], o["calmA"])[0, 1] for o in recs])
        cor_v0 = np.mean([np.corrcoef(o["L"], o["lnv0"]["l"])[0, 1] for o in recs])
        print(f"lr {lr}: {len(recs)} departs, {np.mean([o['nombre'] for o in recs]):.0f} salves ; L moyen {Lm:.1f} (sd {sdL:.1f}) ; corr(L, calme apres) {cor_ca:+.3f} ; corr(L, ln v_l(i0)) {cor_v0:+.3f} ; identite I_q / I_v max ecart {max(o['ident'] for o in recs):.1e}")
        for nom in ("e", "l"):
            s = m[nom]
            print(f"   {nom}3: pente ln I {fmt(s['sI'])} (bloc sd {s['bI'].mean():.1e}) ; pente ln E_b {fmt(s['sE'])} (bloc sd {s['bE'].mean():.1e}) ; ln(E_b / L) {s['sEL'].mean():+.3e} contre -ln L {s['sref'].mean():+.3e} (-1/L moyen {-1 / Lm:+.3e}) ; elasticite {s['elast'].mean():+.3f} ; R2 semi-log {s['r2semi'].mean():.3f} log-log {s['r2log'].mean():.3f}")
            print(f"        pic {fmt(s['sPk'])} ; largeur {fmt(s['sW'])} ; |pic| / |E_b| {abs(s['sPk'].mean()) / max(abs(s['sE'].mean()), 1e-12):.2f} ; fractions par tiers (pente sur L) : debut {s['sf1'].mean():+.2e}, milieu {s['sf2'].mean():+.2e}, fin {s['sf3'].mean():+.2e} ; ln I controle (ln v(i0) e et l, calme apres) {fmt(s['sIc'])} ; part du calme dans I : pad {pad} {100 * s['calm'].mean():.2f} %, pad 20 {100 * s['calm20'].mean():.2f} %")
        rap = m["l"]["sI"].mean() / m["e"]["sI"].mean() if abs(m["e"]["sI"].mean()) > 1e-9 else float("nan")
        rap_dep = np.array([m["l"]["sI"][i] / m["e"]["sI"][i] for i in range(len(recs))])
        print(f"   rapport des pentes de ln I, l3 / e3 : {rap:.3f} (par depart {np.array2string(rap_dep, precision=3)}) ; gamma = demi-ecart des pentes de ln I : {0.5 * (m['e']['sI'].mean() - m['l']['sI'].mean()):+.3e}")
    # point d'allumage : premier lr ou la pente moyenne de ln I passe sous -2e-3
    ral = [lr for lr in tab if tab[lr][1]["e"]["sI"].mean() < -2e-3 and tab[lr][1]["l"]["sI"].mean() < -2e-3]
    print(f"premier lr ou les pentes moyennes de ln I (e3 et l3) sont sous -2e-3 : {ral[0] if ral else 'aucun'} ; pente de ln I a 0,25 : e3 {tab[0.25][1]['e']['sI'].mean():+.3e}, l3 {tab[0.25][1]['l']['sI'].mean():+.3e}" if 0.25 in tab else "")


def etat(seuil, pad):
    """Post hoc (tour 92) : la pente de ln I et de ln E_b sur L survit-elle aux controles d'etat ? Controles : ln v_e(i0) et ln v_l(i0) (etat d'entree), calme apres la salve (T - L)."""
    rng = np.random.default_rng(92)
    print(f"seuil {seuil:g}, pad {pad} ; pentes sur L par pas, moyenne des departs [min .. max]")
    for lr in GRILLE:
        recs = [o for o in (un_depart(lr, k, seuil, pad, rng) for k in DEPARTS) if o is not None]
        if not recs:
            continue
        print(f"lr {lr}:")
        gardes = {}
        for nom in ("e", "l"):
            cols ={"ln I": "lnI", "ln E_b": "lnE", "ln pic": "lnpk", "ln largeur": "lnW"}
            for etiq, cle in cols.items():
                vals = {"aucun": [], "v(i0)": [], "calme apres": [], "v(i0)+calme": []}
                for o in recs:
                    r = o["res"][nom]; L = o["L"]
                    ok = np.isfinite(r[cle])
                    x = L[ok]; y = r[cle][ok]
                    v0 = [o["lnv0"]["e"][ok], o["lnv0"]["l"][ok]]; ca = o["calmA"][ok]
                    vals["aucun"].append(pente(x, y)); vals["v(i0)"].append(pente_controlee(x, v0, y))
                    vals["calme apres"].append(pente_controlee(x, [ca], y)); vals["v(i0)+calme"].append(pente_controlee(x, v0 + [ca], y))
                print(f"   {nom}3 {etiq:11s}: " + " ; ".join(f"{k2} {np.mean(v):+.2e} [{min(v):+.1e} .. {max(v):+.1e}]" for k2, v in vals.items()))
                gardes[(nom, etiq)] = vals
        for etiq in ("ln I", "ln E_b"):
            ge = gardes[("e", etiq)]; gl = gardes[("l", etiq)]
            print(f"   gamma = demi-ecart (e3 - l3) des pentes de {etiq} sur L : " + " ; ".join(f"{k2} {0.5 * np.mean(np.array(ge[k2]) - np.array(gl[k2])):+.2e} [{0.5 * min(np.array(ge[k2]) - np.array(gl[k2])):+.1e} .. {0.5 * max(np.array(ge[k2]) - np.array(gl[k2])):+.1e}]" for k2 in ge))
        # ce qui relie L a l'etat d'entree
        sl = [pente(o["lnv0"]["l"], o["L"]) for o in recs]; se = [pente(o["lnv0"]["e"], o["L"]) for o in recs]
        sc = [pente(o["calmA"], o["L"]) for o in recs]
        cal = [np.nanmax(o["res"][n]["calm"]) for o in recs for n in ("e", "l")]
        print(f"   pente de L sur ln v_l(i0) {np.mean(sl):+.2f} pas, sur ln v_e(i0) {np.mean(se):+.2f} pas ; pente de L sur le calme apres {np.mean(sc):+.3f} ; part du calme dans I : maximum sur toutes les salves {max(cal):.2e}")


def mediation(seuil, pad):
    """Post hoc (tour 92) : pentes de L, ln pic, ln E_b, ln largeur, ln I sur l'etat d'entree ln v_i(i0) de la meme coordonnee (sans passer par L) ; R2 et correlation de L avec le pic."""
    rng = np.random.default_rng(92)
    print(f"seuil {seuil:g}, pad {pad} ; pente sur ln v_i(i0) (i = e3 pour e3, l3 pour l3), moyenne des departs [min .. max], et R2")
    for lr in GRILLE:
        recs = [o for o in (un_depart(lr, k, seuil, pad, rng) for k in DEPARTS) if o is not None]
        if not recs:
            continue
        print(f"lr {lr}: L moyen {np.mean([o['L'].mean() for o in recs]):.1f}")
        for nom in ("e", "l"):
            lignes = []
            for etiq, cle in (("L", None), ("ln pic", "lnpk"), ("ln E_b", "lnE"), ("ln largeur", "lnW"), ("ln I", "lnI")):
                p = []; r2 = []
                for o in recs:
                    x = o["lnv0"][nom]
                    y = o["L"] if cle is None else o["res"][nom][cle]
                    ok = np.isfinite(y)
                    p.append(pente(x[ok], y[ok])); r2.append(np.corrcoef(x[ok], y[ok])[0, 1] ** 2)
                lignes.append(f"{etiq} {np.mean(p):+.3g} [{min(p):+.2g} .. {max(p):+.2g}] (R2 {np.mean(r2):.2f})")
            cor = np.mean([np.corrcoef(o["L"], o["res"][nom]["lnpk"])[0, 1] for o in recs])
            sdv = np.mean([o["lnv0"][nom].std() for o in recs])
            sdE = np.mean([np.nanstd(o["res"][nom]["lnE"]) for o in recs]); sdL = np.mean([o["L"].std() for o in recs])
            print(f"   {nom}3: " + " ; ".join(lignes) + f" ; corr(L, ln pic) {cor:+.3f} ; sd de ln v(i0) {sdv:.3f} ; sd de ln E_b {100 * sdE:.2f} % ; sd de L {sdL:.2f} pas")


def profil_pic(seuil=3e-4):
    """Post hoc (tour 92) : profil de q aligne sur le pic de chaque salve (argmax de la moyenne de q sur 3 pas dans le groupe elargi de 5 pas), terciles de ln v_l(i0) ; rapport haut / bas et forme normalisee."""
    taus = list(range(-30, 111, 10))
    for lr in (0.1, 0.15, 0.2, 0.25):
        acc = {nom: {t: [] for t in range(3)} for nom in ("e", "l")}
        forme = {nom: {t: [] for t in range(3)} for nom in ("e", "l")}
        for k in DEPARTS:
            c = charger(lr, k)
            if c is None:
                continue
            ve, vl, u = c
            gr = groupes(u, seuil)
            a = np.array([g[0] for g in gr]); en = np.array([g[1] for g in gr])
            sel = np.arange(10, len(a) - 1)
            lv = np.log(vl[a[sel] - 1])
            q1, q2 = np.quantile(lv, [1 / 3, 2 / 3]); terc = np.digitize(lv, [q1, q2])
            for nom, v in (("e", ve), ("l", vl)):
                qq = np.full(len(v), np.nan); qq[1:] = (v[1:] - B2 * v[:-1]) / (1 - B2)
                q3 = np.convolve(np.nan_to_num(qq), np.ones(3) / 3, mode="same")
                profs = {t: [] for t in range(3)}; fs = {t: [] for t in range(3)}
                for j, tj in zip(sel, terc):
                    ws, we = a[j] - 5, en[j] + 5
                    tp = ws + int(np.argmax(q3[ws:we + 1]))
                    if tp - 30 < 1 or tp + 125 >= len(qq):
                        continue
                    seg = qq[tp - 30:tp + 121]
                    bins = np.array([seg[t + 30:t + 40].sum() for t in taus])
                    profs[tj].append(bins); fs[tj].append(bins / bins.sum())
                for t in range(3):
                    acc[nom][t].append(np.mean(profs[t], axis=0)); forme[nom][t].append(np.mean(fs[t], axis=0))
        print(f"lr {lr}: profil de q aligne sur le pic (pas {taus[0]} .. {taus[-1]} par 10), terciles de ln v_l(i0) : bas (entree a faible v) et haut ; rapport haut / bas ; forme normalisee par q au pic (haut puis bas)")
        for nom in ("e", "l"):
            P = [np.mean(acc[nom][t], axis=0) for t in range(3)]; F = [np.mean(forme[nom][t], axis=0) for t in range(3)]
            print(f"   {nom}3 energie par tranche de 10 pas, haut / bas : " + " ".join(f"{P[2][i] / P[0][i]:.2f}" for i in range(len(taus))))
            print(f"   {nom}3 part de l'energie par tranche, haut : " + " ".join(f"{F[2][i]:.3f}" for i in range(len(taus))))
            print(f"   {nom}3 part de l'energie par tranche, bas  : " + " ".join(f"{F[0][i]:.3f}" for i in range(len(taus))))


def profil():
    for lr in (0.1, 0.15, 0.2, 0.25):
        prof = {nom: {t: [] for t in range(3)} for nom in ("e", "l")}
        for k in DEPARTS:
            c = charger(lr, k)
            if c is None:
                continue
            ve, vl, u = c
            gr = groupes(u, 3e-4)
            a = np.array([g[0] for g in gr]); en = np.array([g[1] for g in gr])
            L = (en - a + 1)[10:-1]; aa = a[10:-1]
            q1, q2 = np.quantile(L, [1 / 3, 2 / 3])
            terc = np.digitize(L, [q1, q2])
            for nom, v in (("e", ve), ("l", vl)):
                qq = np.full(len(v), np.nan); qq[1:] = (v[1:] - B2 * v[:-1]) / (1 - B2)
                for t in range(3):
                    sel = aa[terc == t]
                    prof[nom][t].append(np.array([qq[s - 5:s + 140] for s in sel if s + 140 < len(qq)]).mean(axis=0))
        if not prof["e"][0]:
            continue
        print(f"lr {lr}: q moyen par pas dans la salve (pas -5 .. 135 par 10), terciles de L (court, moyen, long), e3 puis l3, moyenne des departs ; rapport long / court")
        for nom in ("e", "l"):
            P = [np.mean(prof[nom][t], axis=0) for t in range(3)]
            idx = list(range(0, 145, 10))
            print(f"   {nom}3 court: " + " ".join(f"{P[0][i]:.2e}" for i in idx))
            print(f"   {nom}3 long : " + " ".join(f"{P[2][i]:.2e}" for i in idx))
            print(f"   {nom}3 long / court: " + " ".join(f"{P[2][i] / P[0][i]:.2f}" for i in idx))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "enregistrer":
        enregistrer(float(sys.argv[2]), int(sys.argv[3]))
    elif mode == "analyse":
        analyse(float(sys.argv[2]), int(sys.argv[3]))
    elif mode == "etat":
        etat(float(sys.argv[2]), int(sys.argv[3]))
    elif mode == "mediation":
        mediation(float(sys.argv[2]), int(sys.argv[3]))
    elif mode == "profil_pic":
        profil_pic()
    elif mode == "profil":
        profil()
