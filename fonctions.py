"""Fonctions utilisées dans les notebooks : préparation des données, K-medoids, indice de Dunn."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

GRAINE = 42
TAILLE_ECHANTILLON = 4000

# variables retirées (voir notebook 01)
A_RETIRER = ["duration", "month", "day_of_week", "pdays", "emp.var.rate", "nr.employed"]


CHEMIN_DONNEES = Path(__file__).parent / "data" / "bank-additional-full.csv"


def charger_donnees(chemin=CHEMIN_DONNEES):
    df = pd.read_csv(chemin, sep=";")
    df = df.drop_duplicates()                  # on enlève les doublons
    y = (df["y"] == "yes").astype(int)         # cible : 1 si le client a souscrit
    df = df.drop(columns="y")                  # la cible ne sert pas à former les groupes
    return df, y


def nettoyer(df):
    df = df.drop(columns=A_RETIRER)
    limite = df["campaign"].quantile(0.99)     # 14 appels
    df["campaign"] = df["campaign"].clip(upper=limite)
    return df


def echantillonner(df, y):
    # échantillon stratifié : même part de "yes" que dans toutes les données
    X, _, y_ech, _ = train_test_split(df, y, train_size=TAILLE_ECHANTILLON,
                                      stratify=y, random_state=GRAINE)
    return X.reset_index(drop=True), y_ech.reset_index(drop=True)


def types_de_variables(X):
    numeriques = list(X.select_dtypes("number").columns)
    categorielles = [c for c in X.columns if c not in numeriques]
    return numeriques, categorielles


def representation_a(X):
    """(a) numériques standardisées + catégorielles en one-hot."""
    numeriques, categorielles = types_de_variables(X)
    transfo = ColumnTransformer([
        ("num", StandardScaler(), numeriques),
        ("cat", OneHotEncoder(sparse_output=False), categorielles),
    ])
    return transfo.fit_transform(X)


def representation_b(X):
    """(b) données mixtes brutes, pour Gower et K-prototypes."""
    numeriques, categorielles = types_de_variables(X)
    Xb = X.copy()
    Xb[categorielles] = Xb[categorielles].astype(object)   # format attendu par gower et kmodes
    return Xb


def preparer():
    """Toute la préparation : renvoie X, y, Xa, Xb."""
    df, y = charger_donnees()
    df = nettoyer(df)
    X, y = echantillonner(df, y)
    Xa = representation_a(X)
    Xb = representation_b(X)
    return X, y, Xa, Xb


# K-medoids et indice de Dunn


def kmedoids(D, k, graine=0):
    """K-medoids sur une matrice de distances D."""
    rng = np.random.default_rng(graine)
    medoides = rng.choice(len(D), k, replace=False)   # k clients tirés au hasard

    for etape in range(100):
        # chaque client va vers le médoïde le plus proche
        groupes = np.argmin(D[:, medoides], axis=1)

        # dans chaque groupe, le client le plus central devient le nouveau médoïde
        nouveaux = medoides.copy()
        for j in range(k):
            membres = np.where(groupes == j)[0]
            if len(membres) > 0:
                total = D[membres][:, membres].sum(axis=1)
                nouveaux[j] = membres[np.argmin(total)]

        # si les médoïdes ne bougent plus, on s'arrête
        if np.array_equal(nouveaux, medoides):
            break
        medoides = nouveaux

    groupes = np.argmin(D[:, medoides], axis=1)
    cout = D[np.arange(len(D)), medoides[groupes]].sum()   # somme des distances aux médoïdes
    return groupes, medoides, cout


def kmedoids_meilleur(D, k, essais=10):
    """Lance K-medoids plusieurs fois et garde le plus petit coût."""
    meilleur = None
    for graine in range(essais):
        resultat = kmedoids(D, k, graine)
        if meilleur is None or resultat[2] < meilleur[2]:
            meilleur = resultat
    return meilleur


def dunn(D, groupes):
    """Indice de Dunn = plus petite séparation / plus grand diamètre."""
    ids = np.unique(groupes)

    # diamètre : plus grande distance à l'intérieur d'un groupe
    diametre = 0
    for a in ids:
        dans_a = groupes == a
        diametre = max(diametre, D[dans_a][:, dans_a].max())

    # séparation : plus petite distance entre deux groupes différents
    separation = np.inf
    for a in ids:
        for b in ids:
            if a < b:
                separation = min(separation, D[groupes == a][:, groupes == b].min())

    return separation / diametre
