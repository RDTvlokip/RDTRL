"""Relance sur le chemin float64 tout ce qui passait par l'ancien.

Bascule decidee le 31/07/2026 : la soustraction recompense - baseline se fait
desormais en Python (donc en double) puis on arrondit une seule fois. Plus juste,
et mesure 4x plus rapide sur cette ligne.

Cinq scripts passaient par `entrainer` et doivent donc etre refaits :

  rl_grammaire.py          balayage d'entropie, tout-ou-rien, grammaire longue,
                           tests de generalisation
  balayage_graines.py      8 coefficients x 3 graines
  balayage_70_graines.py   DEJA FAIT en float64
  sonde_ordre1.py          partie B, branche de la grammaire longue
  produit_et_saturation.py partie B2, les deux runs echantillonnes

Les six autres avaient deja leur propre ligne float64, et six de plus n'ont aucun
entrainement echantillonne (gradient exact, enumerations, formes closes) : rien a
y refaire, et ce sont eux qui portent les resultats les plus forts.

Ce lanceur borne le parallelisme. Sans borne, onze processus torch se battent
pour douze coeurs et tout ralentit ensemble ; avec une borne, chaque tache va a
sa vitesse nominale.
"""

import argparse
import os
import subprocess
import sys
import time

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.abspath(os.path.join(ICI, "..", ".."))
SORTIE = os.path.join(RACINE, "results_test2")

# (nom, commande, fichier temoin qui prouve que c'est fait)
TACHES = [
    ("balayage_graines",
     [sys.executable, "-u", "balayage_graines.py"],
     "balayage_graines_float64.json"),
    ("sonde_ordre1_B",
     [sys.executable, "-u", "sonde_ordre1.py", "--graines", "0", "1", "2", "3", "4"],
     "sonde_ordre1.json"),
    # --sans-gradient-exact : la partie B1 est du gradient exact, sans ligne
    # d'avantage, donc insensible au chemin numerique. La relancer couterait six
    # optimisations sur 8 000 sequences pour un resultat identique.
    ("produit_et_saturation_B2",
     [sys.executable, "-u", "produit_et_saturation.py", "--sans-gradient-exact"],
     "produit_et_saturation.json"),
    ("rl_grammaire",
     [sys.executable, "-u", "rl_grammaire.py"],
     "rapport.json"),
]


def lancer(nom, commande, dossier_log):
    log = open(os.path.join(dossier_log, f"{nom}.log"), "w", encoding="utf-8")
    p = subprocess.Popen(commande, cwd=ICI, stdout=log, stderr=subprocess.STDOUT,
                         env={**os.environ, "OMP_NUM_THREADS": "1",
                              "MKL_NUM_THREADS": "1"})
    p._nom, p._log = nom, log
    return p


def archiver(source, cible):
    """Met les resultats float32 de cote AVANT de les ecraser.

    Les scripts reecrivent rapport.json, balayage_graines.json et les CSV sous
    les memes noms. Renommer chaque sortie serait invasif ; copier le dossier une
    fois ne l'est pas, et garde de quoi comparer les deux chemins.
    """
    import shutil
    if os.path.exists(cible):
        print(f"  archive deja presente, on n'ecrase pas : {cible}")
        return False
    shutil.copytree(source, cible)
    n = sum(len(f) for _, _, f in os.walk(cible))
    print(f"  {n} fichiers archives dans {os.path.basename(cible)}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archiver", action="store_true",
                   help="copie results_test2 vers results_test2_float32 d'abord")
    p.add_argument("--max-paralleles", type=int, default=4,
                   help="taches simultanees ; laisser de la marge au systeme")
    p.add_argument("--dossier-log", default=SORTIE)
    p.add_argument("--taches", nargs="+", default=None,
                   help="sous-ensemble a relancer")
    p.add_argument("--liste", action="store_true", help="affiche et sort")
    args = p.parse_args()

    taches = [t for t in TACHES if args.taches is None or t[0] in args.taches]
    if args.liste:
        for nom, cmd, temoin in taches:
            print(f"  {nom:>26} -> {' '.join(cmd[2:]) or 'rl_grammaire.py'}")
        return

    if args.archiver:
        archiver(SORTIE, os.path.join(RACINE, "results_test2_float32"))

    os.makedirs(args.dossier_log, exist_ok=True)
    en_cours, restantes, debut = [], list(taches), time.time()
    print(f"{len(restantes)} taches, au plus {args.max_paralleles} en parallele\n")

    while restantes or en_cours:
        while restantes and len(en_cours) < args.max_paralleles:
            nom, commande, _ = restantes.pop(0)
            en_cours.append(lancer(nom, commande, args.dossier_log))
            print(f"  [{time.time()-debut:6.0f}s] demarre  {nom}")
        time.sleep(5)
        for proc in list(en_cours):
            if proc.poll() is not None:
                en_cours.remove(proc)
                proc._log.close()
                etat = "OK" if proc.returncode == 0 else f"ECHEC {proc.returncode}"
                print(f"  [{time.time()-debut:6.0f}s] fini     {proc._nom} : {etat}")

    print(f"\nTermine en {(time.time()-debut)/60:.1f} min. "
          f"Journaux dans {args.dossier_log}")


if __name__ == "__main__":
    main()
