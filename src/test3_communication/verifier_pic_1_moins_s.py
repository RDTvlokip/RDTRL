"""Le pic non-monotone de 1-s[0,0] a pas=8000 (2.1,2.3,2.6,2.9,3.3,2.9,2.5,2.2,2.0e-9)
est-il un vrai signal ou du bruit numerique a deux chiffres significatifs ?

Regarde directement les logits bruts (pas le softmax) du referent 0 : si le
pic correspond a un second competiteur qui monte puis retombe pendant que
le message 0 bascule, c'est un vrai phenomene. Si c'est juste le plancher
de precision float64 qui remonte de bruit, les logits ne montreront rien
de coherent.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_idx5 import replay_idx5, BETA
from representable_atteignable_stable import monter

if __name__ == "__main__":
    torch.set_printoptions(precision=15)
    for pas_suite in (7000, 7500, 8000, 8500, 9000):
        e, r = replay_idx5(40000)
        with torch.no_grad():
            e.p[0][0, 0] += 24.0
        monter(e, r, BETA, pas_suite, lr=0.05)
        with torch.no_grad():
            row = e.p[0][0]
            vals, idx = row.topk(3)
            s_ = torch.softmax(row, dim=0)
        print(f"pas={pas_suite:6d}  top3 logits: "
              f"{[(int(i), round(float(v), 6)) for i, v in zip(idx, vals)]}  "
              f"1-s[0,0]={1.0 - s_[0].item():.6e}")
