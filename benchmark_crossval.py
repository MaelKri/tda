import time
import warnings
import gudhi
import gudhi.representations

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'

warnings.filterwarnings('ignore')

# DONNÉES

def generate_complex_shapes(n_per_class=80, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = []
    y = []
    n_points_shape = 100

    for class_label in range(3):
        for _ in range(n_per_class):
            if class_label == 0:
                points = np.random.randn(n_points_shape, 2) * 0.5
            elif class_label == 1:
                r = np.random.uniform(0.5, 1.5)
                angles = np.random.uniform(0, 2 * np.pi, n_points_shape)
                points = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
            elif class_label == 2:
                r1 = np.random.uniform(0.5, 1.0)
                r2 = np.random.uniform(0.5, 1.0)
                angles1 = np.random.uniform(0, 2 * np.pi, n_points_shape // 2)
                angles2 = np.random.uniform(0, 2 * np.pi, n_points_shape // 2)
                c1 = np.array([-1.5, 0])
                c2 = np.array([1.5, 0])
                pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1))) + c1
                pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2))) + c2
                points = np.vstack((pts1, pts2))

            theta = np.random.uniform(0, 2 * np.pi)
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            points = points.dot(rot_matrix.T)
            tx = np.random.uniform(-3, 3)
            ty = np.random.uniform(-3, 3)
            points += np.array([tx, ty])
            points += np.random.normal(0, 0.05, points.shape)
            noise_points = np.random.uniform(-5, 5, (100, 2))
            final_cloud = np.vstack((points, noise_points))

            X.append(final_cloud)
            y.append(class_label)

    return X, np.array(y)


def compute_h1_persistence_diagrams(point_clouds):
    h1_diagrams = []
    for pc in point_clouds:
        rips = gudhi.RipsComplex(points=pc)
        simplex_tree = rips.create_simplex_tree(max_dimension=2)
        persistence = simplex_tree.persistence()
        h1 = [diag[1] for diag in persistence if diag[0] == 1]
        if len(h1) > 0:
            h1_diagrams.append(np.array(h1))
        else:
            h1_diagrams.append(np.empty((0, 2)))
    return h1_diagrams


#MODÈLES

class DeepSet(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.rho = nn.Linear(16, 3)

    def forward(self, x):
        out = self.phi(x)
        out = torch.max(out, dim=0)[0]
        out = self.rho(out)
        return out.unsqueeze(0)


class TrueGaussianPersLay(nn.Module):
    def __init__(self, num_gaussians=60):
        super().__init__()
        self.centers = nn.Parameter(torch.rand(num_gaussians, 2))
        self.log_variances = nn.Parameter(torch.zeros(num_gaussians))

    def forward(self, diagrams, masks):
        d_exp = diagrams.unsqueeze(2)
        c_exp = self.centers.unsqueeze(0).unsqueeze(0)
        dist_sq = torch.sum((d_exp - c_exp)**2, dim=-1)
        variances = torch.exp(self.log_variances)
        phi = torch.exp(-dist_sq / (2 * variances))
        persistence = diagrams[:, :, 1] - diagrams[:, :, 0]
        weights = persistence.unsqueeze(-1)
        masks = masks.unsqueeze(-1)
        weighted_phi = phi * weights * masks
        out = torch.sum(weighted_phi, dim=1)
        return out


class FullTopologicalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.perslay = TrueGaussianPersLay(num_gaussians=60)
        self.classifier = nn.Sequential(
            nn.Linear(60, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, diagrams, masks):
        topo_features = self.perslay(diagrams, masks)
        return self.classifier(topo_features)


class PseudoPersLayMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(60, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


#utilitaires

def extract_manual_features(diagrams):
    features = []
    for diag in diagrams:
        if len(diag) == 0:
            features.append([0.0, 0.0, 0.0])
        else:
            pers = diag[:, 1] - diag[:, 0]
            features.append([np.sum(pers), np.max(pers), np.sum(pers > 0.15)])
    return np.array(features)


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

# evaluation d'un fold : 

def evaluate_fold(X_clouds, H1_diags, y_all, train_idx, test_idx):
    """evalue les 6 pipelines sur un fold donné."""

    X_train = [X_clouds[i] for i in train_idx]
    X_test = [X_clouds[i] for i in test_idx]
    H1_train = [H1_diags[i] for i in train_idx]
    H1_test = [H1_diags[i] for i in test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    fold_results = {}

    # Pipeline 0 : Baseline DeepSets
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
    fold_results['0: Baseline'] = accuracy_score(y_test, preds_0)

    #Pipeline A : Features manuelles + Random Forest
    feat_train_A = extract_manual_features(H1_train)
    feat_test_A = extract_manual_features(H1_test)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(feat_train_A, y_train)
    fold_results['A: Manuelles'] = accuracy_score(y_test, rf.predict(feat_test_A))

    # Pipeline B : Landscapes + SVM
    landscape = gudhi.representations.Landscape(num_landscapes=3, resolution=80)
    feat_train_B = landscape.fit_transform(H1_train)
    feat_test_B = landscape.transform(H1_test)
    scaler_B = StandardScaler()
    feat_train_B = scaler_B.fit_transform(feat_train_B)
    feat_test_B = scaler_B.transform(feat_test_B)
    svc_B = SVC(kernel='rbf', random_state=42)
    svc_B.fit(feat_train_B, y_train)
    fold_results['B: Landscapes'] = accuracy_score(y_test, svc_B.predict(feat_test_B))

    #pipeline C : Betti Curves + SVM 
    betti = gudhi.representations.BettiCurve(resolution=80)
    feat_train_C = betti.fit_transform(H1_train)
    feat_test_C = betti.transform(H1_test)
    scaler_C = StandardScaler()
    feat_train_C = scaler_C.fit_transform(feat_train_C)
    feat_test_C = scaler_C.transform(feat_test_C)
    svc_C = SVC(kernel='rbf', random_state=42)
    svc_C.fit(feat_train_C, y_train)
    fold_results['C: Betti'] = accuracy_score(y_test, svc_C.predict(feat_test_C))

    # pipeline D : Pseudo-PersLay + MLP 
    feat_train_D = diagram_to_1d_gaussian(H1_train)
    feat_test_D = diagram_to_1d_gaussian(H1_test)

    model_D = PseudoPersLayMLP()
    optimizer_D = optim.Adam(model_D.parameters(), lr=0.01)
    criterion_D = nn.CrossEntropyLoss()
    X_t_D = torch.tensor(feat_train_D)
    y_t_D = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(150):
        model_D.train()
        optimizer_D.zero_grad()
        loss = criterion_D(model_D(X_t_D), y_t_D)
        loss.backward()
        optimizer_D.step()

    model_D.eval()
    with torch.no_grad():
        preds_D = torch.argmax(model_D(torch.tensor(feat_test_D)), dim=1).numpy()
    fold_results['D: Pseudo-PL'] = accuracy_score(y_test, preds_D)

    # pipeline E : True PersLay end to end
    all_diags_fold = H1_train + H1_test
    max_pts = max(max(len(d) for d in all_diags_fold), 1)

    train_diags, train_masks = pad_and_mask_diagrams(H1_train, max_pts)
    test_diags, test_masks = pad_and_mask_diagrams(H1_test, max_pts)
    y_t_E = torch.tensor(y_train, dtype=torch.long)

    model_E = FullTopologicalNetwork()
    optimizer_E = optim.Adam(model_E.parameters(), lr=0.01)
    criterion_E = nn.CrossEntropyLoss()

    for epoch in range(150):
        model_E.train()
        optimizer_E.zero_grad()
        loss = criterion_E(model_E(train_diags, train_masks), y_t_E)
        loss.backward()
        optimizer_E.step()

    model_E.eval()
    with torch.no_grad():
        preds_E = torch.argmax(model_E(test_diags, test_masks), dim=1).numpy()
    fold_results['E: PersLay'] = accuracy_score(y_test, preds_E)

    return fold_results



#VALIDATION croisée


def run_cross_validation(n_folds=10, n_per_class=80, seed=42):
    print(f"Cross validation : {n_folds} FOLDS")

    # génération du dataset
    X_clouds, y = generate_complex_shapes(n_per_class=n_per_class, seed=seed)

    # Calcul des diagrammes de persistance
    print("calcul des diagrammes de persistance H1")
    H1_diags = compute_h1_persistence_diagrams(X_clouds)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # stockage : dict pipeline_name -> liste de n_folds accuracies
    all_scores = {}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"Fold {fold_idx+1}/{n_folds} — train: {len(train_idx)}, test: {len(test_idx)}")
        fold_res = evaluate_fold(X_clouds, H1_diags, y, train_idx, test_idx)

        for pipeline_name, acc in fold_res.items():
            if pipeline_name not in all_scores:
                all_scores[pipeline_name] = []
            all_scores[pipeline_name].append(acc)

        summary = " | ".join([f"{k}: {v*100:.1f}%" for k, v in fold_res.items()])
        print(f"{summary}")

    return all_scores



# VISUALISATIONS

def generate_statistical_figures(all_scores):
    pipeline_names = list(all_scores.keys())

    # boxplot des accuracies par fold
    fig, ax = plt.subplots(figsize=(10, 6))

    data_for_box = [np.array(all_scores[p]) * 100 for p in pipeline_names]
    bp = ax.boxplot(data_for_box, labels=pipeline_names, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22']
    for patch, c in zip(bp['boxes'], colors[:len(pipeline_names)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Distribution des Accuracies — Validation Croisée Stratifiée '
                 f'({len(all_scores[pipeline_names[0]])} folds)', fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.axhline(y=33.33, color='grey', linestyle=':', linewidth=1, label='Hasard (33.3%)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('cv_boxplot_accuracies.png', dpi=300)
    plt.close()
    print("Sauvegardé : cv_boxplot_accuracies.png")


if __name__ == "__main__":
    all_scores = run_cross_validation(n_folds=10, n_per_class=80, seed=42)

    generate_statistical_figures(all_scores)

