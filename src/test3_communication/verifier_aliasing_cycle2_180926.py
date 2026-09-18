"""Suite du challenge agent-dipankar (18/09/2026) : le point 4 du script
precedent (verifier_decouverte_seuil_naturel_180926.py) a trouve un
vrai cycle-2 d'Adam (changements de signe de Δs3 : 74/74 pas dans la
fenetre t=225-300, amplitude quasi constante ~5,1-5,2e-4). Ici :
teste directement l'hypothese d'ALIASING -- que le "plateau quasi
immobile 55 pas, t=235-290" rapporte par l'agent (base sur un
echantillonnage check_tous=20, PAIR) est un artefact d'echantillonner
toujours la MEME phase d'un cycle-2, pas un vrai ralentissement.

Calcule, sur LA MEME trajectoire (s3_init=0,972646, meme graine/config),
a la fois :
  - Δs3 PAS A PAS (stride=1, la vraie dynamique)
  - Δs3 echantillonne tous les 20 pas (stride=20, comme
    scratch_trace_naturel.py / le protocole qui a produit "t=263")
et compare directement les deux vues sur la meme fenetre temporelle.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from verifier_prior_asymetrique import continuer_sous_prior, etat
from verifier_sonde_bassin import poids_delta, fixer_s3

ADAM_EPS = 1e-10
DELTA = 0.013

poids = poids_delta(DELTA)
e, r = construire_mur23(adam_eps=ADAM_EPS)
fixer_s3(e, 0.972646)

trace = []
R, H, m, Hb, S = etat(e, r)
trace.append((0, S[3], R[4]))
for t in range(1, 401):
    continuer_sous_prior(e, r, BETA, 1, 0.05, ADAM_EPS, poids)
    R, H, m, Hb, S = etat(e, r)
    trace.append((t, S[3], R[4]))

s3_of = {t: s for (t, s, rv) in trace}

print("=== Vue stride=20 (comme scratch_trace_naturel.py, PAS_BLOC=20) ===")
print("  t     s3          delta20(=s3[t]-s3[t-20])")
stride20 = []
for t in range(20, 401, 20):
    d20 = s3_of[t] - s3_of[t - 20]
    stride20.append((t, s3_of[t], d20))
    marque = "  <-- min |delta20|" if False else ""
    print(f"  {t:4d}  {s3_of[t]:.6f}  {d20:+.3e}")

t_min20 = min(stride20, key=lambda x: abs(x[2]))
print(f"\n  minimum |delta20| : t={t_min20[0]}  s3={t_min20[1]:.6f}  delta20={t_min20[2]:+.3e}")

print("\n=== Vue stride=1 (vraie dynamique pas-a-pas) dans la MEME fenetre t=200-320 ===")
print("  t     s3          delta1(=s3[t]-s3[t-1])")
for t in range(200, 321):
    d1 = s3_of[t] - s3_of[t - 1]
    print(f"  {t:4d}  {s3_of[t]:.6f}  {d1:+.3e}")

print("\n=== Comparaison directe des magnitudes ===")
amp1 = [abs(s3_of[t] - s3_of[t-1]) for t in range(201, 321)]
amp20 = [abs(d) for (_, _, d) in stride20]
print(f"  |delta1|  min={min(amp1):.3e}  max={max(amp1):.3e}  moyenne={sum(amp1)/len(amp1):.3e}")
print(f"  |delta20| min={min(amp20):.3e}  max={max(amp20):.3e}  moyenne={sum(amp20)/len(amp20):.3e}")
print(f"  ratio moyenne(|delta1|)*20 / moyenne(|delta20|) = "
      f"{(sum(amp1)/len(amp1))*20 / (sum(amp20)/len(amp20)):.1f}")
print("  (si ce ratio est >>1 : les 20 pas individuels s'annulent presque")
print("   completement dans la somme -- signature d'aliasing sur un cycle-2,")
print("   pas d'un vrai ralentissement ou chaque pas individuel serait petit)")
