"""Adam avec un eps PAR COORDONNEE (tour 59, 29/09/2026).

Meme formule que torch.optim.Adam (sans amsgrad, sans weight decay) :
    m <- b1 m + (1-b1) g ;  v <- b2 v + (1-b2) g^2
    p <- p - lr/(1-b1^t) * m / ( sqrt(v)/sqrt(1-b2^t) + eps )
avec eps un tenseur de la forme du parametre. Sert a tester ou Adam agit :
eps grand sur une seule ligne, 1e-10 partout ailleurs.
Verifie contre torch.optim.Adam par verifier_tour59_adam_masque_equivalence.py.
"""

import math
import torch


class AdamMasque:
    def __init__(self, params, lr, eps_tenseurs, betas=(0.9, 0.999)):
        self.params = list(params)
        self.lr = lr
        self.eps = [e.clone() for e in eps_tenseurs]
        self.b1, self.b2 = betas
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def charger_depuis_torch(self, opt):
        """Reprend moments et compteur d'un torch.optim.Adam (etat chauffe)."""
        for p, m, v in zip(self.params, self.m, self.v):
            st = opt.state[p]
            m.copy_(st["exp_avg"]); v.copy_(st["exp_avg_sq"])
        self.t = int(opt.state[self.params[0]]["step"])

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, m, v, eps in zip(self.params, self.m, self.v, self.eps):
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            denom = (v.sqrt() / math.sqrt(bc2)) + eps
            p.addcdiv_(m, denom, value=-self.lr / bc1)
