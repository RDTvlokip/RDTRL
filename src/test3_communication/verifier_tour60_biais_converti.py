"""Tour 60 (29/09/2026) : E60-1 et E60-2, reponse a la question de
dipankarsarkar sur le biais des salves converti en unites de delta.

Lit les traces de lancer_tour60_traces.sh (colonnes : pas, 1-s3, R_b, r4, ...).
Pour chaque (eps, offset) : par phase, sur les 30 000 derniers pas,
  biais_nœud = moyenne(d3) - d3 du noeud ferme a ce delta,
  biais_med  = moyenne(d3) - mediane(d3),
converti en unites de delta par la pente locale dd3/ddelta du noeud
(differences centrees de la fermeture), s = biais / pente. Sort la moyenne
et l'ecart-type sur les phases qui ne se sont PAS echappees (R_b < 0,9 sur
toute la trace) et le nombre de phases echappees. Pour E60-2 (eps 2e-8, 5e-8)
sort aussi le seuil predit pli - s_med et le compare aux mesures par loi du
fantome (+3,85e-9 a 2e-8, -3,66e-9 a 5e-8).
"""

import glob
import re
import sys
from collections import defaultdict

import numpy as np

from verifier_tour59_branches_fermees import branche_et_jumeau, mesures, F_PLI

MESURES_DELTA = {"2e-08": +3.85e-9, "5e-08": -3.66e-9}


def pente(d):
    h = abs(d - F_PLI) * 1e-3
    v = []
    for dd in (d - h, d + h):
        rs = branche_et_jumeau(dd)
        v.append(mesures(rs[0], dd)[0])
    return (v[1] - v[0]) / (2 * h)


if __name__ == "__main__":
    filtre_eps = sys.argv[1] if len(sys.argv) > 1 else None
    groupes = defaultdict(list)
    for f in glob.glob("D:/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta*_pas40000_eps*_chauffe20000_200[0-9][0-9]*.txt") + \
             glob.glob("D:/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta*_pas40000_eps*_chauffe20000_2[0-9][0-9][0-9][0-9].txt"):
        m = re.search(r"delta([0-9.e-]+)_pas40000_eps([0-9.e+-]+)_chauffe20000_(\d+)\.txt$", f)
        if not m:
            continue
        d, eps, ce = float(m.group(1)), m.group(2), int(m.group(3))
        if ce == 20000 or (filtre_eps and eps != filtre_eps):
            continue  # ce == 20000 : traces d'avant le tour 60 (protocole a une seule phase), exclues
        groupes[(eps, round(d - F_PLI, 12))].append((ce - 20000, f))
    print("eps      offset    phases (echappees)   pente    s_noeud = biais/pente      s_med = (moy-med)/pente     seuil predit = pli - s_med")
    for (eps, off), L in sorted(groupes.items(), key=lambda kv: (float(kv[0][0]), kv[0][1])):
        d = F_PLI + off
        rs = branche_et_jumeau(d)
        dA = mesures(rs[0], d)[0]
        sl = pente(d)
        sn, sm, esc = [], [], 0
        for k, f in sorted(L):
            T = np.loadtxt(f)[-30000:]
            full = np.loadtxt(f)
            if full[:, 2].max() > 0.9:
                esc += 1
                continue
            x = T[:, 1]
            sn.append((x.mean() - dA) / sl)
            sm.append((x.mean() - np.median(x)) / sl)
        if not sn:
            print(f"{eps}  {off:+.0e}   {len(L)} ({esc})   tous echappes")
            continue
        sn, sm = np.array(sn), np.array(sm)
        pred = -sm.mean()
        ligne = (f"{eps}  {off:+.0e}   {len(L)} ({esc})   {sl:8.2f}   {sn.mean():+.3e} +- {sn.std(ddof=1) if len(sn) > 1 else 0:.1e}"
                 f"     {sm.mean():+.3e} +- {sm.std(ddof=1) if len(sm) > 1 else 0:.1e}      {pred:+.3e}")
        if eps in MESURES_DELTA:
            mes = MESURES_DELTA[eps]
            ligne += f"   (mesure loi du fantome {mes:+.2e} ; ecart {100 * (pred / mes - 1):+.0f} %)"
        print(ligne)
