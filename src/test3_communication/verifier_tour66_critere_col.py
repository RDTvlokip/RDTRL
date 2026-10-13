"""Tour 66 : critere d'effondrement POST HOC, lu sur les coupures en x du tour 65 (eps/lr = 8e-7) :
l'etat s'effondre quand l'ecart noeud-col en d3, 0,2213 sqrt(x) (verifier_tour66_ecart_col.py), tombe sous
k * sd_d3, avec D_d > 0 (eps/lr au-dessus du zero de D_d, 6,1-6,2e-7, verifier_tour66_paire_pli.py).
sd_d3 ne depend presque pas de x (tour 65) : on prend sd mesure a x = 1e-8 au meme eps (interpole).
Puis : coupures predites pour eps/lr = 1e-6 et 2e-6 (test precommis P66-9).
"""

import math

A = 0.2213  # ecart col-noeud = A sqrt(x)


def ecart(x):
    return A * math.sqrt(x)


if __name__ == "__main__":
    # sd_d3 a eps/lr = 8e-7 : lr 0,05 eps 4e-8 entre (3,5e-8 : 2,2148e-6) et (5e-8 : 2,1020e-6) ;
    # lr 0,02 eps 1,6e-8 entre (1,4e-8 : 8,7561e-7, x 1e-8) et (2e-8 : 7,1646e-7, x 1e-8)
    sd05 = 2.2148e-6 + (2.1020e-6 - 2.2148e-6) * (4e-8 - 3.5e-8) / (5e-8 - 3.5e-8)
    sd02 = 8.7561e-7 + (7.1646e-7 - 8.7561e-7) * (1.6e-8 - 1.4e-8) / (2e-8 - 1.4e-8)
    print(f"sd_d3 a eps/lr 8e-7 : lr 0,05 {sd05:.4e}, lr 0,02 {sd02:.4e}, rapport {sd05 / sd02:.3f}")
    for lr, sd, xs in ((0.05, sd05, ((2e-9, "16/16"), (3e-9, "0/16"))), (0.02, sd02, ((3e-10, "16/16 (a 8,0e-7)"), (4e-10, "0/16")))):
        for x, res in xs:
            print(f"   lr {lr} x {x:.0e} : ecart {ecart(x):.3e}  ecart/sd = {ecart(x) / sd:.2f}   observe {res}")
    KLO = max(ecart(2e-9) / sd05, ecart(3e-10) / sd02)  # plus grand rapport ou l'etat s'effondre
    KHI = min(ecart(3e-9) / sd05, ecart(4e-10) / sd02)  # plus petit rapport ou il tient
    print(f"   => seuil k entre {KLO:.2f} et {KHI:.2f} (intersection des encadrements des deux lr)")
    print(f"   rapport des coupures predit par x_c ~ sd^2 : {(sd02 / sd05) ** 2:.3f}  (mesure au tour 65 : 0,10 a 0,20)")
    print("\ncoupures predites x_c = (k sd / A)^2, k dans [KLO ; KHI], sd mesure a x = 1e-8 :")
    for lr, eps, sd in ((0.05, 5e-8, 2.1020e-6), (0.02, 2e-8, 7.1646e-7), (0.05, 1e-7, 1.5905e-6), (0.02, 4e-8, 4.1435e-7)):
        lo, hi = (KLO * sd / A) ** 2, (KHI * sd / A) ** 2
        print(f"   lr {lr} eps {eps:.1e} (eps/lr {eps / lr:.0e}) sd {sd:.3e} : x_c entre {lo:.2e} et {hi:.2e}")
