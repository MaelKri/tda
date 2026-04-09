# TDA & Deep Learning

Ce dépôt rassemble les expérimentations et le code accompagnant notre rapport sur l'Analyse Topologique des Données (TDA) appliquée au Machine Learning.

## Contenu

* **Concepts & Visualisations** : `diagramme_persistance.py`, `nerve_theorem.py` et `rayon_reconstruction.py` génèrent des figures illustrant la théorie de base de la TDA.
* **Notebook** : `TDA_ML.ipynb` pour tester les concepts pas à pas.
* **Réseaux de Neurones** : `topological_regularization.py` implémente un Autoencodeur avec une fonction de perte topologique sous PyTorch.
* **Benchmarks (ML classique vs TDA)** : `benchmark_simple.py` et `benchmark_crossval.py` comparent les performances de pipelines standards avec des approches topologiques (PersLay, Betti Curves, Landscapes).

## Installation

Librairies : 

```bash
pip install -r requirements.txt
```

Exécution : 

```bash
python benchmark_crossval.py
```
