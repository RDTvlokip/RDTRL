"""Agent (role dipankarsarkar, tour 59 simule) : tous les nombres en forme
fermee cites dans la critique, recalcules a partir des chiffres publies
(REPONSE_ORDRE58.md, logs du balayage du tour 58) et des logs agent_dk59_*.

  kappa = (1-delta) s3(1-s3) / beta = H(e3[10],r4)/H(r4,r4)
  dS    = pente * S_gap * kappa
  dt    = 2 * pente * kappa / (1 - beta2)
  loi sous le plancher : pente = G * poids3 * s3(1-s3) * r3 r4, G epingle a delta=0
"""
import math

BETA, BETA2, N, EPS = 0.02, 0.999, 27, 1e-10

print("-- dS : forme fermee contre eigh exact (agent_dk59_hessien_explicite.py) --")
for pas, pente, sgap, h3, h44, eigh in [(59440, 0.93672, 35.9782, 6.1766e-6, 1.2083e-4, 1.7228),
                                        (59520, 0.93429, 37.4483, 6.1769e-6, 1.2083e-4, 1.7885)]:
    print(f"  pas {pas}: pente*S_gap*kappa = {pente*sgap*h3/h44:.4f}   eigh {eigh}")
print(f"  d ln S_gap/dt (59440->59520) = {math.log(37.4483/35.9782)/80:.4e}   (1-beta2)/2 = {(1-BETA2)/2:.1e}")
print(f"  kappa reel = {0.98697346*1.03587e-3/BETA:.6f} ; H(e310,r4)/H44 = {6.1766e-6/1.2083e-4:.6f}")
print(f"  H44 = (beta/N) r3 r4 : reel {BETA/N*0.20524*0.79476:.5e} (eigh 1.2083e-4), delta=0 {BETA/N*0.25:.5e} (1.8519e-4)")
print(f"  u ~ sqrt(v r10[4]) * kappa : reel {3.2581e-7*0.051118:.4e} (mesure 1.7715e-8), "
      f"delta=0 {4.8544e-7*1.7505e-8:.4e} (mesure 8.625e-15)")
print(f"  delta=0, eps3=0 : dS = 0.68 * 38 * 1.7505e-8 = {0.68*38*1.7505e-8:.2e}")

print("-- dt par delta (pente et X final du balayage du tour 58) --")
for d, X, pente in [(0.013026615, 10.128537, 0.937), (0.012, 11.29519, 0.850), (0.010, 13.29145, 0.586)]:
    a = 26 * math.exp(-X)
    u = a / (1 + a)
    ss = u * (1 - u)
    kappa = (1 - d) * ss / BETA
    print(f"  delta={d}: s3(1-s3)={ss:.5e}  kappa={kappa:.5e}  dt={2*pente*kappa/(1-BETA2):.2f} pas   "
          f"(beta2=0.998 : {2*pente*kappa/0.002:.1f})")

print("-- loi de couplage sous le plancher --")
c0 = (1 / N) * 3.5002e-10 * 0.25
G = 4.7485e-5 / c0
print(f"  G = {G:.5e}")
for d, un, r3, r4, u, mes, ratio_u in [(0.002, 4.5634e-9, 0.45017, 0.54983, 1.1493e-13, 6.1665e-4, 1.0364),
                                       (0.004, 5.4382e-8, 0.40131, 0.59869, 1.3454e-12, 7.0536e-3, 1.0246)]:
    c = ((1 - d) / N) * un * (1 - un) * r3 * r4
    corr = (26 * EPS / (u + EPS) + EPS / (u / 26 + EPS)) / 27
    print(f"  delta={d}: G*c3={G*c:.4e}  x correction eps {corr:.6f} = {G*c*corr:.4e}  mesure {mes:.4e}  "
          f"rapport {mes/(G*c*corr):.4f}  (formule(u) : {ratio_u})")
print(f"  ligne4/ligne3 a delta=0 : predit {4.3612e-12/3.5002e-10:.4f}  mesure {5.5316e-7/4.7485e-5:.4f}")
print(f"  decroissance de u pendant le retard : 0.9995^98 = {0.9995**98:.4f} ; mesure 1.6608/1.7443 = {1.6608/1.7443:.4f}")
print(f"  periode : base {(61769-59989)/4:.1f}, ligne 3 gelee {(61591-60095)/3:.1f}")
print(f"  relache (test A) : ln(2.734e-4/6.578e-10) = {math.log(2.734e-4/6.578e-10):.2f} nats")
