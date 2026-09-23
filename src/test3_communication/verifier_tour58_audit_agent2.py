"""Tour 58 (23/09/2026) : verification INDEPENDANTE (regle 5bis), avec mon
propre code, des trois affirmations cles du second agent style dipankar
(referent 5 orphelin, synonymes, bord de stabilite).

(i)  JOUET c_K. Il affirme que l'ecart residuel (+1,1 % orphelin, +8 %
     synonymes) n'est pas "v un peu au-dessus du seuil a cause des salves"
     mais une constante qui ne depend que de la taille K du bloc
     d'entropie : sur un quadratique pur f = 1/2 h z^T (I - 11^T/K) z
     optimise par Adam (lr 0,05, eps 1e-10, betas par defaut),
     z_rms / (lr/38 - eps/h) = c_K, c_2 = 1,078-1,082, c_27 = 1,0113.
     Ici : mon propre jouet, K = 2 et 27, 200 000 pas, moyenne sur les
     100 000 derniers.
(ii) LE PIEGE. L'orphelin n'est qu'un optimum de bloc : deplacer le
     referent 5 sur le message 19 (ligne 5 +30 sur 19, ligne 8 -30 sur 19,
     recepteur ligne 19 +30 sur le referent 5) puis relaxer gagnerait
     Delta J = 1/N - (beta/N)(ln 27 + ln 2) = 0,034082.
(iii) COLLISIONS SPONTANEES. Un recensement limite aux lignes emettrices
     ne voit pas les collisions ; 12345 k=3 en aurait trois (messages 3,
     7, 8 : referents 3/25, 16/19, 6/14, recepteur a 0,5/0,5) et
     77777 k=1 une (message 14 : referents 5/20).
"""

import sys
sys.path.insert(0, '.')
import math
import torch

N = 27


def partie_i():
    for K in (2, 27):
        torch.manual_seed(1)
        h = 2.7435e-5
        lr, eps = 0.05, 1e-10
        z = (torch.randn(K, dtype=torch.float64) * 1e-3).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr, eps=eps)
        P = torch.eye(K, dtype=torch.float64) - 1.0 / K
        rms2, sv = [], []
        for t in range(200000):
            opt.zero_grad()
            f = 0.5 * h * (z @ P @ z)
            f.backward()
            opt.step()
            if t >= 100000:
                with torch.no_grad():
                    zc = z - z.mean()
                    rms2.append((zc * zc).mean().item())
                    sv.append(opt.state[z]["exp_avg_sq"].sqrt().mean().item())
        z_rms = math.sqrt(sum(rms2) / len(rms2))
        ref = lr / 38 - eps / h
        print(f"(i) K={K:2d} : z_rms/(lr/38 - eps/h) = {z_rms / ref:.4f}   "
              f"<sqrt v>/(lr h/38 - eps) = {sum(sv) / len(sv) / (lr * h / 38 - eps):.4f}   (agent : c_2 1,078-1,082 ; c_27 1,0113)")


def partie_ii():
    from replay_mur23_referent3 import BETA
    from verifier_prior_asymetrique import objectif_pondere
    from sauver_checkpoints_mur23_tour58 import reprendre

    def relaxer(modifier):
        e, r, opt, poids = reprendre(0.013026615)
        if modifier:
            with torch.no_grad():
                e.p[0][5, 19] += 30.0
                e.p[0][8, 19] -= 30.0
                r.p[0][19, 5] += 30.0
            opt = torch.optim.Adam(opt.param_groups[0]["params"], lr=0.05, eps=1e-10)
        for _ in range(5000):
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
        with torch.no_grad():
            j, _ = objectif_pondere(e, r, BETA, poids)
            S = e.loi()
        return j.item(), S

    j0, _ = relaxer(False)
    j1, S1 = relaxer(True)
    pred = 1 / N - (BETA / N) * (math.log(27) + math.log(2))
    print(f"(ii) J code orphelin = {j0:.6f}   J code deplace = {j1:.6f}   Delta J = {j1 - j0:.6f}   "
          f"predit {pred:.6f}   s[5,19] apres = {S1[5, 19].item():.6f}  s[8,23] apres = {S1[8, 23].item():.6f}")


def partie_iii():
    from replay_mur23_referent3 import replay
    for graine, k in ((12345, 3), (77777, 1)):
        e, r = replay(graine, k, 30000)
        with torch.no_grad():
            S, R = e.loi(), r.loi()
            dec = R.argmax(1)
            compte = torch.bincount(dec, minlength=N)
            zeros = [i for i in range(N) if compte[i] == 0]
            extras = int(sum(max(0, c - 1) for c in compte.tolist()))
            print(f"(iii) graine={graine} k={k} : messages en trop (synonymes) = {extras}, "
                  f"referents sans message decode = {zeros}")
            for m in range(N):
                top2 = R[m].topk(2)
                if top2.values[1].item() > 0.1:
                    a, b = top2.indices.tolist()
                    print(f"      message {m} : recepteur {a}:{top2.values[0].item():.6f} / {b}:{top2.values[1].item():.6f}   "
                          f"emetteur s[{a},{m}]={S[a, m].item():.6f} s[{b},{m}]={S[b, m].item():.6f}")
            for i in zeros:
                H = -(S[i] * S[i].clamp_min(1e-300).log()).sum().item()
                print(f"      referent {i} sans message : H(ligne emettrice)={H:.3e}  message dominant {S[i].argmax().item()}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    quoi = sys.argv[1] if len(sys.argv) > 1 else "i"
    if "i" == quoi:
        partie_i()
    if "ii" == quoi:
        partie_ii()
    if "iii" == quoi:
        partie_iii()
