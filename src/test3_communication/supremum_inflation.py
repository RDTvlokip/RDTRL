import sys, time, itertools, numpy as np
sys.path.insert(0,'src/test3_communication')
from loi_nulle_longue import N, matrices_information, statistiques

def objectif(lot):
    cm,ca,_=statistiques(matrices_information(lot,verifier_bijectivite=False))
    return cm-ca

PAIRES=[(i,j) for i in range(N) for j in range(i+1,N)]
# mouvement enrichi : transpositions ET 3-cycles, pour sortir des optima locaux
# que le voisinage par transpositions seul ne peut pas quitter
TRIPLES=[(i,j,k) for i,j,k in itertools.combinations(range(N),3)]

def voisins_transposition(code):
    v=np.repeat(code[None,:],len(PAIRES),axis=0)
    for n,(i,j) in enumerate(PAIRES): v[n,i],v[n,j]=code[j],code[i]
    return v

def voisins_3cycle(code, echantillon, g):
    idx=g.choice(len(TRIPLES), size=min(echantillon,len(TRIPLES)), replace=False)
    v=np.repeat(code[None,:],len(idx),axis=0)
    for n,t in enumerate(idx):
        i,j,k=TRIPLES[t]
        v[n,i],v[n,j],v[n,k]=code[j],code[k],code[i]
    return v

def monter(code, g, avec_3cycles):
    valeur=float(objectif(code[None,:])[0])
    for _ in range(300):
        cand=voisins_transposition(code)
        if avec_3cycles:
            cand=np.concatenate([cand, voisins_3cycle(code,1200,g)])
        vals=objectif(cand); k=int(vals.argmax())
        if vals[k]<=valeur+1e-12: break
        code,valeur=cand[k].copy(),float(vals[k])
    return valeur,code

for avec in (False,True):
    g=np.random.default_rng(12345)
    meilleur=-1.0; hist=[]; t0=time.time()
    n_departs=1500
    for d in range(n_departs):
        v,_=monter(g.permutation(N), g, avec)
        if v>meilleur: meilleur=v
        if (d+1) in (24,96,384,750,1500): hist.append((d+1,meilleur))
    nom="transpositions + 3-cycles" if avec else "transpositions seules"
    print(f"{nom}  ({time.time()-t0:.0f} s)")
    for d,m in hist: print(f"    {d:>5} departs : {m:.6f}")
