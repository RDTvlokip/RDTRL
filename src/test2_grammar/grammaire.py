"""Grammaire formelle ecrite a la main : le juge de grammaticalite du test 2.

Aucun modele, aucune IA, aucune phrase cible. Uniquement des regles.

Difference fondamentale avec le test 1 : le parser ne compare jamais la sortie
de l'agent a une reponse attendue. Il verifie des contraintes structurelles,
quelle que soit la phrase produite. Il existe donc plusieurs dizaines de phrases
valides differentes, pas une cible unique.

Les traits grammaticaux ci-dessous servent UNIQUEMENT au parser. L'agent ne les
voit jamais : pour lui un token est un index sans structure interne.
"""

from itertools import product

# Lexique : token -> (categorie, genre, nombre). None = trait non pertinent
# ou neutre (ex. 'les' s'accorde avec les deux genres).

LEXIQUE_COURT = {
    "le":  ("det", "m", "sg"),
    "la":  ("det", "f", "sg"),
    "un":  ("det", "m", "sg"),
    "une": ("det", "f", "sg"),
    "les": ("det", None, "pl"),
    "des": ("det", None, "pl"),

    "chat":   ("nom", "m", "sg"),
    "chats":  ("nom", "m", "pl"),
    "chien":  ("nom", "m", "sg"),
    "chiens": ("nom", "m", "pl"),
    "table":  ("nom", "f", "sg"),
    "tables": ("nom", "f", "pl"),
    "fleur":  ("nom", "f", "sg"),
    "fleurs": ("nom", "f", "pl"),

    "dort":     ("verbe", None, "sg"),
    "dorment":  ("verbe", None, "pl"),
    "mange":    ("verbe", None, "sg"),
    "mangent":  ("verbe", None, "pl"),
    "chante":   ("verbe", None, "sg"),
    "chantent": ("verbe", None, "pl"),
}

# Extension pour la grammaire longue : adjectifs (accord genre + nombre) et
# adverbes (aucun accord, ils ne servent qu'a agrandir l'espace de recherche).
LEXIQUE_LONG = {
    "petit":   ("adj", "m", "sg"),
    "petits":  ("adj", "m", "pl"),
    "petite":  ("adj", "f", "sg"),
    "petites": ("adj", "f", "pl"),
    "grand":   ("adj", "m", "sg"),
    "grands":  ("adj", "m", "pl"),
    "grande":  ("adj", "f", "sg"),
    "grandes": ("adj", "f", "pl"),

    "vite":    ("adv", None, None),
    "souvent": ("adv", None, None),
    "bien":    ("adv", None, None),
}


# Variante RENVERSEE, pour le test de renversement du plafond (carnet 7.12).
#
# Le plafond de produit d'un coin est le plus grand ensemble produit
# entierement valide qu'il contient. Dans le lexique standard il vaut 24 au
# pluriel et 12 au singulier, parce que 'les' et 'des' sont neutres en genre :
# ils suppriment la contrainte de genre, donc le coin pluriel EST un produit,
# alors que le coin singulier est l'union de deux produits (masculin, feminin).
#
# Ici on deplace la neutralite de genre vers le singulier, et on marque les
# pluriels. MEMES 20 tokens, meme espace de 8 000 sequences, memes 48 phrases
# valides, memes deux coins de 24 : seule la structure de produit est echangee.
# Le plafond doit donc devenir 24 au singulier et 12 au pluriel.
#
# Si le plafond suit ce renversement, c'est une propriete de la structure de la
# recompense. S'il ne le suit pas, c'etait une coincidence de mon vocabulaire.
LEXIQUE_COURT_RENVERSE = dict(LEXIQUE_COURT)
LEXIQUE_COURT_RENVERSE.update({
    "le":  ("det", None, "sg"),
    "la":  ("det", None, "sg"),
    "un":  ("det", "m", "pl"),
    "une": ("det", "f", "pl"),
    "les": ("det", "m", "pl"),
    "des": ("det", "f", "pl"),
})
#
# ATTENTION : cette variante-la ne teste RIEN, et c'est instructif.
# Les noms et les verbes sont deja symetriques en nombre (2 par genre et par
# nombre, 3 verbes de chaque). Echanger le nombre des determinants EST donc le
# renommage sg <-> pl, et les deux grammaires sont isomorphes. Le resultat
# serait le miroir du standard par construction. Conservee comme contre-exemple
# documente : un controle peut etre parfaitement symetrique et parfaitement
# vide. Voir carnet 7.13.
#
# Un renommage permute, il ne peut pas changer un RAPPORT. Le vrai test fait
# donc varier la VALEUR du plafond. Avec trois genres au lieu de deux :
#
#   coin singulier : determinants neutres en genre -> 2 det x 6 noms x 3 verbes
#                    = 36, et c'est un seul produit, donc plafond 36 ;
#   coin pluriel   : determinants marques -> il faut fixer le genre
#                    -> 2 det x 2 noms x 3 verbes = 12.
#
# Les deux coins contiennent 36 phrases valides, mais leurs plafonds sont dans
# un rapport de 3 et non de 2. Aucun renommage de la grammaire a deux genres ne
# peut produire ce rapport : le plus grand produit est un invariant.
LEXIQUE_TROIS_GENRES = {
    "le":  ("det", None, "sg"),
    "un":  ("det", None, "sg"),
    "les": ("det", "m", "pl"),
    "des": ("det", "m", "pl"),
    "ces": ("det", "f", "pl"),
    "mes": ("det", "f", "pl"),
    "ses": ("det", "n", "pl"),
    "tes": ("det", "n", "pl"),

    "chat":   ("nom", "m", "sg"),  "chats":   ("nom", "m", "pl"),
    "chien":  ("nom", "m", "sg"),  "chiens":  ("nom", "m", "pl"),
    "table":  ("nom", "f", "sg"),  "tables":  ("nom", "f", "pl"),
    "fleur":  ("nom", "f", "sg"),  "fleurs":  ("nom", "f", "pl"),
    "arbre":  ("nom", "n", "sg"),  "arbres":  ("nom", "n", "pl"),
    "livre":  ("nom", "n", "sg"),  "livres":  ("nom", "n", "pl"),

    "dort":     ("verbe", None, "sg"),  "dorment":  ("verbe", None, "pl"),
    "mange":    ("verbe", None, "sg"),  "mangent":  ("verbe", None, "pl"),
    "chante":   ("verbe", None, "sg"),  "chantent": ("verbe", None, "pl"),
}


class Grammaire:
    """Parser deterministe : structure + accords, sans aucune phrase de reference."""

    def __init__(self, longue=False, variante="standard"):
        self.longue = longue
        self.variante = variante
        self.lexique = dict({"renverse": LEXIQUE_COURT_RENVERSE,
                             "trois_genres": LEXIQUE_TROIS_GENRES}
                            .get(variante, LEXIQUE_COURT))
        if longue:
            self.lexique.update(LEXIQUE_LONG)

        self.tokens = sorted(self.lexique)
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.taille = len(self.tokens)

        # Ordre des categories impose par la grammaire
        self.structure = (["det", "adj", "nom", "verbe", "adv"] if longue
                          else ["det", "nom", "verbe"])
        self.longueur = len(self.structure)

        # Accords obligatoires : (categorie, categorie de reference, traits verifies)
        # Tous les accords se font par rapport au nom, qui porte le genre et le nombre.
        self.accords = [("det", "nom", ("genre", "nombre"))]
        if longue:
            self.accords.append(("adj", "nom", ("genre", "nombre")))
        self.accords.append(("verbe", "nom", ("nombre",)))

        # Pour chaque categorie, les traits qui doivent s'accorder avec le nom
        self.traits_a_accorder = {cat: traits for cat, _, traits in self.accords}

        self.positions = {cat: i for i, cat in enumerate(self.structure)}
        self.tokens_par_categorie = {
            cat: [t for t in self.tokens if self.lexique[t][0] == cat]
            for cat in set(self.structure)
        }

    def traits(self, token):
        categorie, genre, nombre = self.lexique[token]
        return {"categorie": categorie, "genre": genre, "nombre": nombre}

    @staticmethod
    def _compatible(valeur_a, valeur_b):
        """None = neutre, s'accorde avec tout (ex. 'les' en genre)."""
        return valeur_a is None or valeur_b is None or valeur_a == valeur_b

    def analyser(self, ids):
        """Renvoie les sous-scores de grammaticalite d'une sequence d'index.

        - structure    : fraction de positions occupees par la bonne categorie
        - accord_X_nom : fraction des traits correctement accordes avec le nom
        - valide       : True seulement si tous les sous-scores valent 1
        """
        mots = [self.tokens[i] for i in ids]
        traits = [self.traits(m) for m in mots]

        positions_correctes = [traits[i]["categorie"] == self.structure[i]
                               for i in range(self.longueur)]
        score_structure = sum(positions_correctes) / self.longueur

        i_nom = self.positions["nom"]
        scores = {"structure": score_structure}

        for categorie, _, noms_traits in self.accords:
            i_cat = self.positions[categorie]
            cle = f"accord_{categorie}_nom"
            # Un accord n'est evaluable que si les deux positions portent bien
            # la categorie attendue ; sinon le sous-score est nul.
            if not (positions_correctes[i_cat] and positions_correctes[i_nom]):
                scores[cle] = 0.0
                continue
            nb_ok = sum(1 for nom_trait in noms_traits
                        if self._compatible(traits[i_cat][nom_trait], traits[i_nom][nom_trait]))
            scores[cle] = nb_ok / len(noms_traits)

        scores["valide"] = all(v == 1.0 for k, v in scores.items() if k != "valide")
        return scores

    def recompense_graduee(self, ids):
        """Moyenne des sous-scores : signal dense, sans oracle sur une phrase."""
        scores = self.analyser(ids)
        sous_scores = [v for k, v in scores.items() if k != "valide"]
        return sum(sous_scores) / len(sous_scores)

    def recompense_tout_ou_rien(self, ids):
        """Controle : 1 seulement si les regles sont toutes respectees."""
        return 1.0 if self.analyser(ids)["valide"] else 0.0

    def taille_espace(self):
        return self.taille ** self.longueur

    def compter_phrases_valides(self):
        """Compte exact, calcule analytiquement : toutes les contraintes passent
        par le nom, donc on enumere les noms et on compte les options compatibles
        pour chaque autre position."""
        total = 0
        for nom in self.tokens_par_categorie["nom"]:
            traits_nom = self.traits(nom)
            options = 1
            for categorie in self.structure:
                if categorie == "nom":
                    continue
                traits_requis = self.traits_a_accorder.get(categorie, ())
                compatibles = [
                    t for t in self.tokens_par_categorie[categorie]
                    if all(self._compatible(self.traits(t)[nt], traits_nom[nt])
                           for nt in traits_requis)
                ]
                options *= len(compatibles)
            total += options
        return total

    def enumerer_valides(self):
        """Liste explicite des phrases valides. Ne doit etre appelee que sur la
        grammaire courte : l'espace de la grammaire longue est trop grand."""
        valides = []
        for combinaison in product(range(self.taille), repeat=self.longueur):
            if self.analyser(combinaison)["valide"]:
                valides.append(" ".join(self.tokens[i] for i in combinaison))
        return valides

    def probabilite_hasard(self):
        """Taux de validite exact d'un generateur uniforme."""
        return self.compter_phrases_valides() / self.taille_espace()

    def resume(self):
        lignes = [
            f"Grammaire {'longue' if self.longue else 'courte'} : "
            f"{' + '.join(self.structure)}",
            f"  vocabulaire      : {self.taille} tokens "
            f"({', '.join(f'{len(v)} {k}' for k, v in sorted(self.tokens_par_categorie.items()))})",
            f"  espace           : {self.taille}^{self.longueur} = {self.taille_espace():,}".replace(",", " "),
            f"  phrases valides  : {self.compter_phrases_valides()}",
            f"  validite hasard  : {100 * self.probabilite_hasard():.4f} %",
            f"  accords verifies : " + ", ".join(f"{c}-{r} sur {'/'.join(t)}"
                                                 for c, r, t in self.accords),
        ]
        return "\n".join(lignes)


if __name__ == "__main__":
    for longue in (False, True):
        g = Grammaire(longue=longue)
        print(g.resume())
        # Verification du compte analytique par force brute (grammaire courte
        # uniquement : 8 000 sequences enumerables, 28 M ne le sont pas).
        if not longue:
            valides = g.enumerer_valides()
            assert len(valides) == g.compter_phrases_valides(), "compte analytique faux"
            print(f"  verifie par force brute : {len(valides)} phrases valides")
            print(f"  exemples : {valides[:3]} ... {valides[-2:]}")
        print()

    # Quelques cas de test du parser
    g = Grammaire()
    for phrase in ["le chat dort", "la chat dort", "les chat dort",
                   "le chat dorment", "des fleurs chantent", "dort le chat"]:
        ids = [g.index[m] for m in phrase.split()]
        s = g.analyser(ids)
        print(f"{phrase:22s} -> valide={str(s['valide']):5s} "
              f"reward={g.recompense_graduee(ids):.3f}  {  {k: v for k, v in s.items() if k != 'valide'} }")
