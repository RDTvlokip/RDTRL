"""Tour 57 (22/09/2026, VRAIE critique de dipankarsarkar, pas simulee) :
verifie sa these centrale -- ni la ligne 3 ni la ligne 10 n'ont besoin
d'une somme complete a 27 termes. Les deux sont des sigmoides a DEUX
sorties EXACTES une fois le bon complement identifie (dbar, pas 0) ;
mes residus de 0,65%/0,88% ne sont que l'artefact de LINEARISATION
(premier ordre / R(1-R)*delta) contre la sigmoide exacte (non
linearisee).

Sa formule generale, pour une ligne a 2 issues dominantes (a,b) avec
dbar = deplacement partage du "reste" (le complement) :

  a_exact_nouveau = sigmoid(logit_a_nouveau - logit_b_nouveau)
  a_exact_ancien  = sigmoid(logit_a_ancien - logit_b_ancien)
  d_a_exact = a_exact_nouveau - a_exact_ancien   (PAS lineraise)

contre l'approximation au premier ordre deja publiee :

  d_a_lineaire = a(1-a) * (dlogit_a - dlogit_b)

Teste sur la ligne 10 (r4 vs r3) ET la ligne 3 (s3 vs dbar, le mode
commun des 26 autres messages).
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from verifier_precommis_dipankar_sigma_row10 import tracer as tracer_ligne10
from verifier_partenaire_miroir_s3 import tracer as tracer_ligne3

P0, P1 = 59000, 61000
PAS_TESTS = (59989, 60432)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def verifier_ligne10():
    print("=== Ligne 10 (r4 vs r3), sigmoide exacte sur le gap logit ===")
    logits = tracer_ligne10(0.013026615)
    for pas_test in PAS_TESTS:
        _, _, _, _, ligne_0, s3p_0, R4p_0 = logits[P0]
        _, _, _, _, ligne_1, s3p_1, R4p_1 = logits[P1]
        _, _, _, _, ligne_t, s3p_t, R4p_t = logits[pas_test]
        t = (pas_test - P0) / (P1 - P0)
        base_logit = ligne_0 + t * (ligne_1 - ligne_0)
        base_R4p = R4p_0 + t * (R4p_1 - R4p_0)

        gap_base = (base_logit[4] - base_logit[3]).item()
        gap_t = (ligne_t[4] - ligne_t[3]).item()

        r4_exact_base = sigmoid(gap_base)
        r4_exact_t = sigmoid(gap_t)
        dR_exact = r4_exact_t - r4_exact_base

        d4 = (ligne_t[4] - base_logit[4]).item()
        d3 = (ligne_t[3] - base_logit[3]).item()
        R_lin = r4_exact_base  # R(1-R) au point de base
        dR_lin = R_lin * (1 - R_lin) * (d4 - d3)

        dR_mesure = R4p_t - base_R4p

        print(f"pas={pas_test}")
        print(f"  dR mesure directement          = {dR_mesure:.6e}")
        print(f"  dR sigmoide EXACTE (sur le gap) = {dR_exact:.6e}  "
              f"ecart={100*abs(dR_exact-dR_mesure)/abs(dR_mesure):.4f}%")
        print(f"  dR premier ordre (lineaire)     = {dR_lin:.6e}  "
              f"ecart={100*abs(dR_lin-dR_mesure)/abs(dR_mesure):.4f}%")


def verifier_ligne3():
    print("\n=== Ligne 3 (s3 vs dbar, mode commun des 26 autres) ===")
    logits = tracer_ligne3()
    for pas_test in PAS_TESTS:
        ligne3_0 = logits[P0]
        ligne3_1 = logits[P1]
        ligne3_t = logits[pas_test]
        t = (pas_test - P0) / (P1 - P0)
        base_logit = ligne3_0 + t * (ligne3_1 - ligne3_0)

        autres_idx = [j for j in range(27) if j != 10]
        K = len(autres_idx)  # 26, le K de H7
        # dbar = moyenne des logits des 26 autres au point de base et au point test
        dbar_base = base_logit[autres_idx].mean().item()
        dbar_t = ligne3_t[autres_idx].mean().item()

        # CORRECTION : reduction a 2 sorties d'un softmax a 27 entrees ou 26
        # sont quasi-egales a dbar necessite le facteur de multiplicite ln(K)
        # -- exp(logit_10) / (exp(logit_10) + K*exp(dbar)), pas
        # exp(logit_10)/(exp(logit_10)+exp(dbar)). Meme "26" que H7.
        gap_base = (base_logit[10] - dbar_base) - math.log(K)
        gap_t = (ligne3_t[10] - dbar_t) - math.log(K)

        s3_exact_base = sigmoid(gap_base)
        s3_exact_t = sigmoid(gap_t)
        ds3_exact = s3_exact_t - s3_exact_base

        d10 = (ligne3_t[10] - base_logit[10]).item()
        ddbar = dbar_t - dbar_base
        s3_lin = s3_exact_base
        ds3_lin = s3_lin * (1 - s3_lin) * (d10 - ddbar)

        # ds3 mesure directement (probabilite reelle, via softmax complet)
        s3_prob_base = torch.softmax(base_logit, dim=0)[10].item()
        s3_prob_t = torch.softmax(ligne3_t, dim=0)[10].item()
        ds3_mesure = s3_prob_t - s3_prob_base

        print(f"pas={pas_test}")
        print(f"  ds3 mesure directement (softmax complet, 27 entrees) = {ds3_mesure:.6e}")
        print(f"  ds3 sigmoide EXACTE (s3 vs dbar)                     = {ds3_exact:.6e}  "
              f"ecart={100*abs(ds3_exact-ds3_mesure)/abs(ds3_mesure):.4f}%")
        print(f"  ds3 premier ordre (lineaire, s3(1-s3)*(d10-ddbar))    = {ds3_lin:.6e}  "
              f"ecart={100*abs(ds3_lin-ds3_mesure)/abs(ds3_mesure):.4f}%")
        print(f"  dbar={dbar_t-dbar_base:.6e} (attendu ~+8.3007e-3 selon dipankar)")


if __name__ == "__main__":
    verifier_ligne10()
    verifier_ligne3()
