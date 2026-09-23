"""agent (dipankar role), 23/09 : census of the pigeonhole in several trained
codes -- synonyms (K messages -> 1 referent), orphans (no message hears the
referent), collisions (a message heard as 2 referents) -- and what the
sender-rows-only census of verifier_autres_murs_systeme.py (H(s_i) > 1e-6)
can and cannot see.
Usage: python agent_t58_l_recensement.py <config> [steps]
config : mur23_0 | mur23_reel | <seed>_<k> (standard replay, Adam default)"""
import sys
sys.path.insert(0, '.')
import torch
from replay_mur23_referent3 import replay, BETA
from representable_atteignable_stable import activer, parametres, objectif

torch.set_num_threads(1)
N = 27
cfg = sys.argv[1]
if cfg.startswith("mur23"):
    from sauver_checkpoints_mur23_tour58 import reprendre
    e, r, opt, poids = reprendre(0.0 if cfg == "mur23_0" else 0.013026615)
else:
    graine, k = (int(a) for a in cfg.split("_"))
    pas = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    e, r = replay(graine, k, 0)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05)
    for _ in range(pas):
        j, _ = objectif(e, r, BETA)
        opt.zero_grad()
        (-j).backward()
        opt.step()
with torch.no_grad():
    S, R = e.loi(), r.loi()
    Hs = -(S * S.clamp_min(1e-300).log()).sum(1)
    Hr = -(R * R.clamp_min(1e-300).log()).sum(1)
    heard = {}                      # referent -> list of (message, r[m,i]) with r > 0.05
    for m in range(N):
        for i in range(N):
            if R[m, i] > 0.05:
                heard.setdefault(i, []).append((m, round(R[m, i].item(), 4)))
    syn = {i: ms for i, ms in heard.items() if len(ms) > 1}
    orph = [i for i in range(N) if i not in heard]
    coll = [m for m in range(N) if (R[m] > 0.05).sum() > 1]
    extra = sum(len(ms) - 1 for ms in syn.values())
    print(f"{cfg}: synonym groups {syn}")
    print(f"   orphans {orph}   collision messages {[(m, [(i, round(R[m, i].item(), 4)) for i in range(N) if R[m, i] > 0.05]) for m in coll]}")
    print(f"   pigeonhole : extra synonym messages {extra} = orphans {len(orph)} + collisions {len(coll)} ? {extra == len(orph) + len(coll)}")
    print(f"   sender rows with H > 1e-6 (the 21/09 census) : {[(i, f'{Hs[i].item():.3e}') for i in range(N) if Hs[i] > 1e-6]}")
    print(f"   receiver rows with H > 1e-6                  : {[(m, f'{Hr[m].item():.3e}') for m in range(N) if Hr[m] > 1e-6]}")
    for m in coll:
        senders = [(i, f'{Hs[i].item():.1e}') for i in range(N) if S[i, m] > 0.5]
        print(f"   collision on message {m}: sender rows (H) {senders}")
