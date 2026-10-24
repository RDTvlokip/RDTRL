"""Tour 87 : les salves en retard ont-elles une definition avant la lecture, et la population a +0,4 % ?

Il dit : oui au niveau de l'enregistrement, non a "toutes les salves comptees" ; rapporter p_late(lr) et l'exposant sur les lectures non en retard, "en retard" defini par l'instant de franchissement avant le groupe (jamais par la lecture) ; la population a +0,4 % a-t-elle une definition avant le groupe ?

Modes :
  arithmetique : ses chiffres a partir de mes chiffres du tour 86 (aucune donnee lue)
"""

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from verifier_tour86_derive import poids_ponderes  # noqa: E402


def arithmetique():
    d = math.log(3.29510e6 / 3.30752e6)
    print(f"ln(3,29510e6 / 3,30752e6) = {100 * d:+.4f} % ; 31/300 = {100 * 31 / 300:.2f} % ; deficit moyen des salves en retard si tout l'ecart vient d'elles : {-100 * d / (31 / 300):.2f} %")
    print(f"exposant haut : 0,0127 - 0,0053 = {0.0127 - 0.0053:.4f} ; divise par {-d:.5f} = poids {(0.0127 - 0.0053) / -d:.2f} ; part de la difference bas - haut : {0.0074 / 0.0131 * 100:.1f} %")
    # poids du point 0,3 dans les deux ajustements (ET entre departs + plancher 0,1 %), pour verifier que 1,97 est un poids
    for nom, se03 in (("toutes les lectures", 0.00102), ("sans les salves en retard", 0.00024)):
        se = [math.hypot(0.00009, 0.001), math.hypot(0.00018, 0.001), math.hypot(se03, 0.001)]
        w = poids_ponderes(se)
        print(f"poids des points 0,2 / 0,25 / 0,3 ({nom}, ET entre departs 0,009 / 0,018 / {100 * se03:.3f} % + plancher 0,1 %) : {np.round(w, 3)}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "arithmetique":
        arithmetique()
