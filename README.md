# Segmentation de clientèle bancaire

Projet final du module Apprentissage non supervisé (Master BDA/IA, UN-CHK).

Une banque veut mieux connaître ses clients pour cibler ses campagnes d'appels. J'ai cherché des groupes de clients
homogènes dans le jeu de données Bank Marketing, sans utiliser la variable « a souscrit ou non », puis j'ai évalué
ces groupes.

## Structure

```
.
├── notebooks/
│   ├── 01_preparation.ipynb           nettoyage, exploration, PCA, t-SNE, UMAP
│   ├── 02_partitionnement.ipynb       distances, K-means, K-medoids, K-prototypes, GMM
│   ├── 03_hierarchique_densite.ipynb  CAH (4 liens), DBSCAN, limites de K-means
│   ├── 04_validation.ipynb            indices internes/externes, stabilité, choix de la solution
│   └── 05_interpretation.ipynb        profil des segments et recommandations
├── data/bank-additional-full.csv      données (UCI)
├── fonctions.py                       préparation réutilisable, K-medoids, indice de Dunn
└── requirements.txt
```

## Lancer le projet

```
pip install -r requirements.txt
jupyter notebook notebooks/
```

Ouvrir les notebooks dans l'ordre. L'ensemble tourne en 3 minutes environ.
La graine aléatoire est fixée à 42 : les résultats sont reproductibles.

## Résultat

La meilleure solution est une classification hiérarchique de Ward en 3 segments, dont les taux de souscription
sont de 5,1 %, 18,7 % et 41,6 % (11,3 % en moyenne).

## Remarques

- La cible `y` n'est jamais utilisée pour former les groupes, seulement pour les évaluer.
- `scikit-learn-extra` ne fonctionne pas avec NumPy 2 : K-medoids est donc écrit à la main dans `fonctions.py`.

## Données

S. Moro, P. Rita et P. Cortez (2014), Bank Marketing, UCI Machine Learning Repository (licence CC BY 4.0).
