# TP1 - Pretraitement des donnees (Titanic)

Ce projet contient un notebook de Data Science qui montre, pas a pas, le pretraitement du dataset Titanic.

## Objectif du TP

L'objectif est de preparer les donnees avant la modelisation machine learning :

- comprendre la structure du dataset,
- nettoyer les valeurs manquantes,
- traiter les valeurs extremes,
- transformer les variables categorielle/numerique,
- corriger le desequilibre de la variable cible.

## Fichier principal

- `titanic_preprocessing.ipynb` : notebook complet du TP.

## Contenu du notebook

### Partie 1 - Introduction au dataset

- Chargement du jeu de donnees Titanic.
- Affichage des types de variables (`dtypes`).
- Apercu des 5 premieres lignes (`head()`).
- Comptage des valeurs manquantes (`isnull().sum()`).

### Partie 2 - Nettoyage des donnees

1. Gestion des valeurs manquantes
- Colonnes avec plus de 40 % de valeurs manquantes : suppression.
- Variables numeriques : imputation par la mediane.
- Variables categorielles : imputation par le mode.

2. Gestion des valeurs extremes
- Detection visuelle avec des boxplots.
- Traitement avec la regle IQR : remplacement par la mediane.

### Partie 3 - Transformation des donnees

1. Encodage
- Encodage One-Hot pour les variables nominales (ex: Sex, Embarked).
- `Pclass` est conservee comme variable ordinale numerique.

2. Mise a l'echelle
- Normalisation Min-Max (valeurs entre 0 et 1).
- Standardisation Z-score (moyenne 0, ecart-type 1).
- Comparaison des distributions avant/apres avec histogrammes.

### Partie 4 - Desequilibre des classes

- Verification de la repartition de `Survived` avec `value_counts()`.
- Sous-echantillonnage de la classe majoritaire.
- Sur-echantillonnage avec SMOTE.
- Si SMOTE n'est pas disponible (incompatibilite de versions), le notebook applique un repli automatique pour continuer l'execution.

## Prerequis

Installer les bibliotheques Python necessaires :

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
```

## Comment executer

1. Ouvrir `titanic_preprocessing.ipynb` dans VS Code/Jupyter.
2. Redemarrer le kernel.
3. Executer les cellules dans l'ordre, du debut vers la fin.

## Resultat attendu

A la fin du TP, vous obtenez un dataset propre et transforme, pret pour entrainer des modeles de classification.
