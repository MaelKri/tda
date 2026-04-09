"""
diagramme de persistance et code-barres


"""

import numpy as np
import gudhi
import matplotlib.pyplot as plt
from sklearn import datasets

def main():

    # génération de données (Topologie connue : cercles concentriques)
    print("génération du nuage de points")
    n_samples = 150
    X, _ = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.title("Nuage de points (Noisy Circles)")
    plt.axis('equal')

    # construction du complexe de Vietoris-Rips
    print("construction du Rips complex")
    rips_complex = gudhi.RipsComplex(points=X, max_edge_length=2.0)
    
    #creation du Simplex Tree
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2) 
    # max_dimension=2 car nous voulons voir les triangles (pour trouver les trous H1)
    
    print(f"   Nombre de simplexes : {simplex_tree.num_simplices()}")
    print(f"   Dimension du complexe : {simplex_tree.dimension()}")

    # calcul de l'Homologie Persistante
    print("calcul de la persistance")
    diag = simplex_tree.persistence(min_persistence=0.01)

    
    # Affichage & sauvegarde
    plt.subplot(132)
    gudhi.plot_persistence_diagram(diag, axes=plt.gca(), legend=True)
    plt.title("Diagramme de Persistance")

    plt.subplot(133)
    gudhi.plot_persistence_barcode(diag, axes=plt.gca(), legend=True)
    plt.title("Code-barres de Persistance")

    plt.tight_layout()
    
    output_filename = "persistence_results.png"
    plt.savefig(output_filename)
    print(f"graphiques sauvegardés dans '{output_filename}'")

if __name__ == "__main__":
    main()
