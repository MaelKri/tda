import time
import gudhi
import gudhi.representations

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


def generate_complex_shapes(n_per_class=80):
    X = []
    y = []
    
    # Nombre de points constituant la forme (avant le bruit de fond)
    n_points_shape = 100 

    for class_label in range(3):
        for _ in range(n_per_class):
            
            # génération de la forme de base
            if class_label == 0:
                # classe 0 : amas gaussien
                points = np.random.randn(n_points_shape, 2) * 0.5
                
            elif class_label == 1:
                # classe 1 : anneau
                r = np.random.uniform(0.5, 1.5)
                angles = np.random.uniform(0, 2 * np.pi, n_points_shape)
                points = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
                
            elif class_label == 2:
                # classe 2 : 2 anneaux disjoints
                r1 = np.random.uniform(0.5, 1.0)
                r2 = np.random.uniform(0.5, 1.0)
                
                angles1 = np.random.uniform(0, 2 * np.pi, n_points_shape // 2)
                angles2 = np.random.uniform(0, 2 * np.pi, n_points_shape // 2)
                
                # decalage des centres pour garantir qu'ils soient disjoints
                c1 = np.array([-1.5, 0])
                c2 = np.array([1.5, 0])
                
                pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1))) + c1
                pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2))) + c2
                points = np.vstack((pts1, pts2))

            # dégradation des données
            # rotation aléatoire
            theta = np.random.uniform(0, 2 * np.pi)
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            points = points.dot(rot_matrix.T)

            # translation aléatoire
            tx = np.random.uniform(-3, 3)
            ty = np.random.uniform(-3, 3)
            points += np.array([tx, ty])

            # bruit gaussien local
            points += np.random.normal(0, 0.05, points.shape)

            # bruit de fond extreme
            noise_points = np.random.uniform(-5, 5, (100, 2))
            
            # concat finale
            final_cloud = np.vstack((points, noise_points))
            
            X.append(final_cloud)
            y.append(class_label)

    return X, np.array(y)


def compute_h1_persistence_diagrams(point_clouds):
    """
    calcul des diagrammes de persistance H1 pour une liste de nuages de points.
    """
    h1_diagrams = []
    
    for pc in point_clouds:
        # creation du complexe de Vietoris-Rips à partir du nuage de points
        rips = gudhi.RipsComplex(points=pc)
        
        # création de l'arbre simplicial (max_dimension=2 requis pour calculer H1)
        simplex_tree = rips.create_simplex_tree(max_dimension=2)
        
        # calcul de l'homologie persistante
        persistence = simplex_tree.persistence()
        
        # on ne conserver que la dimension 1 (les cycles H1)
        h1 = [diag[1] for diag in persistence if diag[0] == 1]
        
        if len(h1) > 0:
            h1_diagrams.append(np.array(h1))
        else:
            h1_diagrams.append(np.empty((0, 2)))
            
    return h1_diagrams

# Séparation des données

class TrueGaussianPersLay(nn.Module):
    def __init__(self, num_gaussians=60):
        super().__init__()
        
        # 1. Paramètres ENTRAINABLES du PersLay
        # Les centres t_j des gaussiennes en 2D (naissance, mort)
        self.centers = nn.Parameter(torch.rand(num_gaussians, 2)) 
        
        # Les variances (sigma) de chaque gaussienne
        # On stocke le log de la variance pour s'assurer que la variance reste positive
        self.log_variances = nn.Parameter(torch.zeros(num_gaussians)) 
        
    def forward(self, diagrams, masks):
        """
        diagrams : Tenseur de forme (Batch_size, Max_points, 2) contenant les coordonnées (birth, death).
        masks : Tenseur de forme (Batch_size, Max_points) contenant 1.0 pour un vrai point, 0.0 pour du padding.
        """
        # Transformation phi(p) via les gaussiennes
        # broadcasting pour calculer la distance entre chaque point et chaque centre
        # diagrams shape -> (B, N, 1, 2)
        # centers shape  -> (1, 1, q, 2)
        d_exp = diagrams.unsqueeze(2)
        c_exp = self.centers.unsqueeze(0).unsqueeze(0)
        
        # distance au carré ||p - t_j||^2 : shape -> (B, N, q)
        dist_sq = torch.sum((d_exp - c_exp)**2, dim=-1)
        
        # récupération des variances : shape -> (q,)
        variances = torch.exp(self.log_variances)
        
        # application de la gaussienne : shape -> (B, N, q)
        phi = torch.exp(-dist_sq / (2 * variances))
        
        # pondération w(p)
        # On utilise la persistance (mort - naissance) comme poids naturel
        # diagrams[:, :, 1] = mort , diagrams[:, :, 0] = naissance
        persistence = diagrams[:, :, 1] - diagrams[:, :, 0] # (B, N)
        weights = persistence.unsqueeze(-1) # (B, N, 1)
        
        # application des poids et du masque de padding
        masks = masks.unsqueeze(-1) # (B, N, 1)
        weighted_phi = phi * weights * masks # (B, N, q)
        
        # agrégation (Sum Pooling)
        # On somme sur la dimension des points (dim=1)
        # Résultat final : shape -> (B, q)
        out = torch.sum(weighted_phi, dim=1)
        
        return out
    
class FullTopologicalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Couche d'extraction topologique (apprend à vectoriser)
        self.perslay = TrueGaussianPersLay(num_gaussians=60)
        
        # classifieur standard (apprend à séparer les vecteurs)
        self.classifier = nn.Sequential(
            nn.Linear(60, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
    def forward(self, diagrams, masks):
        # le diagramme brut devient un vecteur de taille 60 optimisé
        topo_features = self.perslay(diagrams, masks)
        
        # classification
        out = self.classifier(topo_features)
        return out

def prepare_data_and_run_pipelines(X_point_clouds, H1_diagrams, y_labels):
    X_train, X_test, H1_train, H1_test, y_train, y_test = train_test_split(
        X_point_clouds, H1_diagrams, y_labels, test_size=0.3, random_state=42
    )
    
    results = {}

    # PIPELINE 0 : Baseline DeepSets (Non Topologique)
    print("Pipeline 0 (DeepSets)")
    start_time = time.time()
    
    class DeepSet(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
            self.rho = nn.Linear(16, 3)
            
        def forward(self, x):
            # x shape: (N, 2) - N points for one point cloud
            out = self.phi(x)
            out = torch.max(out, dim=0)[0] # Max pooling symétrique
            out = self.rho(out)
            return out.unsqueeze(0) # (1, 3) pour la CrossEntropy

    model_0 = DeepSet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_0.parameters(), lr=0.01)

    for epoch in range(150):
        model_0.train()
        for pc, label in zip(X_train, y_train):
            pc_tensor = torch.tensor(pc, dtype=torch.float32)
            label_tensor = torch.tensor([label], dtype=torch.long)
            
            optimizer.zero_grad()
            output = model_0(pc_tensor)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()

    model_0.eval()
    preds_0 = []
    with torch.no_grad():
        for pc in X_test:
            pc_tensor = torch.tensor(pc, dtype=torch.float32)
            output = model_0(pc_tensor)
            preds_0.append(torch.argmax(output).item())
            
    results['0: Baseline DeepSets'] = {
        'accuracy': accuracy_score(y_test, preds_0),
        'time': time.time() - start_time
    }


    # PIPELINE A : Features manuelles + Random Forest
    print("Pipeline A (Features manuelles)")
    start_time = time.time()
    
    def extract_manual_features(diagrams):
        features = []
        for diag in diagrams:
            if len(diag) == 0:
                features.append([0.0, 0.0, 0.0])
            else:
                pers = diag[:, 1] - diag[:, 0]
                features.append([np.sum(pers), np.max(pers), np.sum(pers > 0.15)])
        return np.array(features)

    feat_train_A = extract_manual_features(H1_train)
    feat_test_A = extract_manual_features(H1_test)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(feat_train_A, y_train)
    preds_A = rf_model.predict(feat_test_A)
    
    results['A: Features Manuelles'] = {
        'accuracy': accuracy_score(y_test, preds_A),
        'time': time.time() - start_time
    }


    # PIPELINE B : Persistence Landscapes + SVC
    print("Pipeline B (Landscapes)")
    start_time = time.time()
    
    landscape = gudhi.representations.Landscape(num_landscapes=3, resolution=80)
    feat_train_B = landscape.fit_transform(H1_train)
    feat_test_B = landscape.transform(H1_test)

    scaler_B = StandardScaler()
    feat_train_B = scaler_B.fit_transform(feat_train_B)
    feat_test_B = scaler_B.transform(feat_test_B)

    svc_B = SVC(kernel='rbf', random_state=42)
    svc_B.fit(feat_train_B, y_train)
    preds_B = svc_B.predict(feat_test_B)
    
    results['B: Landscapes'] = {
        'accuracy': accuracy_score(y_test, preds_B),
        'time': time.time() - start_time
    }

    # PIPELINE C : Betti Curves + SVC
    print("Pipeline C (Betti Curves)")
    start_time = time.time()
    
    betti = gudhi.representations.BettiCurve(resolution=80)
    feat_train_C = betti.fit_transform(H1_train)
    feat_test_C = betti.transform(H1_test)

    scaler_C = StandardScaler()
    feat_train_C = scaler_C.fit_transform(feat_train_C)
    feat_test_C = scaler_C.transform(feat_test_C)

    svc_C = SVC(kernel='rbf', random_state=42)
    svc_C.fit(feat_train_C, y_train)
    preds_C = svc_C.predict(feat_test_C)
    
    results['C: Betti Curves'] = {
        'accuracy': accuracy_score(y_test, preds_C),
        'time': time.time() - start_time
    }

    # PIPELINE D : Pseudo-PersLay (Gaussian 1D) + MLP PyTorch
    print("Pipeline D (Pseudo-PersLay)")
    start_time = time.time()
    
    def diagram_to_1d_gaussian(diagrams, resolution=60, sigma=0.1, max_pers=2.0):
        grid = np.linspace(0, max_pers, resolution)
        features = []
        for diag in diagrams:
            sig_1d = np.zeros(resolution)
            if len(diag) > 0:
                pers = diag[:, 1] - diag[:, 0]
                for p in pers:
                    sig_1d += np.exp(-((grid - p)**2) / (2 * sigma**2))
            features.append(sig_1d)
        return np.array(features, dtype=np.float32)

    feat_train_D = diagram_to_1d_gaussian(H1_train)
    feat_test_D = diagram_to_1d_gaussian(H1_test)

    class PseudoPersLayMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(60, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 3)
            )
            
        def forward(self, x):
            return self.net(x)

    model_D = PseudoPersLayMLP()
    optimizer_D = optim.Adam(model_D.parameters(), lr=0.01)

    X_tensor_train = torch.tensor(feat_train_D)
    y_tensor_train = torch.tensor(y_train, dtype=torch.long)
    X_tensor_test = torch.tensor(feat_test_D)

    for epoch in range(150):
        model_D.train()
        optimizer_D.zero_grad()
        output = model_D(X_tensor_train)
        loss = criterion(output, y_tensor_train)
        loss.backward()
        optimizer_D.step()

    model_D.eval()
    with torch.no_grad():
        output = model_D(X_tensor_test)
        preds_D = torch.argmax(output, dim=1).numpy()
        
    results['D: Pseudo-PersLay'] = {
        'accuracy': accuracy_score(y_test, preds_D),
        'time': time.time() - start_time
    }

    # PIPELINE E : PersLay
    print("Pipeline E (PersLay End-to-End)")
    start_time = time.time()
    
    # préparation des données (padding et masques)
    max_pts = max(max([len(diag) for diag in H1_train]), max([len(diag) for diag in H1_test]))
    max_pts = max(1, max_pts) # Sécurité au cas où tous les diagrammes seraient vides
    
    def pad_and_mask_diagrams(diagrams, max_pts):
        N = len(diagrams)
        padded_diags = np.zeros((N, max_pts, 2), dtype=np.float32)
        masks = np.zeros((N, max_pts), dtype=np.float32)
        for i, diag in enumerate(diagrams):
            pts_count = len(diag)
            if pts_count > 0:
                padded_diags[i, :pts_count, :] = diag
                masks[i, :pts_count] = 1.0
        return torch.tensor(padded_diags), torch.tensor(masks)

    train_diags, train_masks = pad_and_mask_diagrams(H1_train, max_pts)
    test_diags, test_masks = pad_and_mask_diagrams(H1_test, max_pts)
    y_tensor_train = torch.tensor(y_train, dtype=torch.long)

    # initialisation du modèle
    model_true_perslay = FullTopologicalNetwork()
    criterion_true = nn.CrossEntropyLoss()
    optimizer_true = optim.Adam(model_true_perslay.parameters(), lr=0.01)

    # boucle d'entraînement
    for epoch in range(150):
        model_true_perslay.train()
        optimizer_true.zero_grad()
        
        output = model_true_perslay(train_diags, train_masks)
        loss = criterion_true(output, y_tensor_train)
        
        loss.backward()
        optimizer_true.step()

    # evaluation
    model_true_perslay.eval()
    with torch.no_grad():
        output_test = model_true_perslay(test_diags, test_masks)
        preds_true = torch.argmax(output_test, dim=1).numpy()
        
    results['E: PersLay'] = {
        'accuracy': accuracy_score(y_test, preds_true),
        'time': time.time() - start_time
    }

    return results


def generate_visualizations(X_point_clouds, y_labels, results):

    # figure 1 : dataset_bruit.png (Visualisation des 3 classes)
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle("Échantillons du Dataset : Structures noyées dans le bruit", fontsize=16)

    classes_names = ['Classe 0 (Amas)', 'Classe 1 (1 Anneau)', 'Classe 2 (2 Anneaux)']
    for i in range(3):
        idx = np.random.choice(np.where(y_labels == i)[0])
        pc = X_point_clouds[idx]
        
        axes[i].scatter(pc[:, 0], pc[:, 1], s=10, alpha=0.6, c='blue')
        axes[i].set_title(classes_names[i])
        axes[i].set_xlim(-5, 5)
        axes[i].set_ylim(-5, 5)
        axes[i].set_aspect('equal')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('dataset_bruit.png', dpi=300)
    plt.close()
    print("Sauvegardé : dataset_bruit.png")

    #extraction des données du dictionnaire results
    pipelines = list(results.keys())
    accuracies = [results[p]['accuracy'] * 100 for p in pipelines] # En pourcentage
    times = [results[p]['time'] for p in pipelines]


    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6']

    # figure 2 : benchmark_accuracies.png
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars = ax2.barh(pipelines, accuracies, color=colors)
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_title('Comparaison des Accuracies par Pipeline')
    ax2.set_xlim(0, 110)
    
    for bar in bars:
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.1f}%', 
                 va='center', ha='left')

    plt.tight_layout()
    plt.savefig('benchmark_accuracies.png', dpi=300)
    plt.close()
    print("Sauvegardé : benchmark_accuracies.png")

    #figure 3 : benchmark_times.png
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bars = ax3.barh(pipelines, times, color=colors)
    ax3.set_xlabel('Temps d\'exécution (secondes)')
    ax3.set_title('Comparaison des Temps de Calcul par Pipeline')
    
    for bar in bars:
        ax3.text(bar.get_width() + (max(times)*0.01), bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.2f}s', 
                 va='center', ha='left')

    plt.tight_layout()
    plt.savefig('benchmark_times.png', dpi=300)
    plt.close()
    print("Sauvegardé : benchmark_times.png")

    # Affichage d'un récapitulatif dans la console
    print("\nPerformances :")
    for p in pipelines:
        print(f"{p:<25} | Accuracy: {results[p]['accuracy']*100:>6.2f}% | Temps: {results[p]['time']:>6.2f}s")

if __name__=="__main__":
    X, y = generate_complex_shapes(n_per_class=80)
    H1_diagrams = compute_h1_persistence_diagrams(X)
    results = prepare_data_and_run_pipelines(X, H1_diagrams, y)
    generate_visualizations(X, y, results)