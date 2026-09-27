"""Tour 59 : dans les etats de depart des deux collisions (mur 23 et 6/14
de 12345 k=3), les lignes des collisionneurs ont-elles bien C = 26
concurrents SANS recompense (hypothese du pli de dipankar) ? Imprime
1-s, C_eff = (1-s) e^{r/beta} (26 a l'equilibre), la plus grande
recompense r[m,i] hors du message de collision, et la masse du recepteur
hors des deux collisionneurs."""

import sys
sys.path.insert(0, '.')
import math
import torch

from verifier_tour59_delta_c_dynamique import depart

if __name__ == "__main__":
    for cas, a, b in (("mur23", 3, 4), ("c814", 6, 14)):
        (e, r), msg = depart(cas)
        with torch.no_grad():
            S, R = e.loi(), r.loi()
            for i in (a, b):
                autres = [m for m in range(27) if m != msg]
                rew = max(R[m, i].item() for m in autres)
                print(f"{cas} ref {i} : 1-s={1 - S[i, msg].item():.4e}  "
                      f"C_eff={(1 - S[i, msg].item()) * math.exp(R[msg, i].item() / 0.02):.3f}  "
                      f"max r[m,{i}] hors msg={rew:.2e}  R[msg,{i}]={R[msg, i].item():.6f}  "
                      f"masse autres refs sur msg={(1 - R[msg, a] - R[msg, b]).item():.2e}")
