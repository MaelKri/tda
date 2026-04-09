"""
Régularisation Topologique d'un Autoencodeur
"""

import numpy as np
import gudhi
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#génération des données (anneau 2D)
def generate_annulus(n_points=300, r_inner=0.7, r_outer=1.3):
    """
    genere un anneau 2D avec un trou
    topologie attendue : 1 composante connexe (H0), 1 cycle (H1).
    """
    angles = np.random.uniform(0, 2 * np.pi, n_points)
    radii = np.random.uniform(r_inner, r_outer, n_points)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y)).astype(np.float32)


# loss topo avec pénalisation de l'absence de persistance (dans H1)
def topological_loss(latent_points, target_n_cycles=1, max_edge=3.0):
    points = latent_points.detach().cpu().numpy()

    if len(points) > 80:
        idx = np.random.choice(len(points), 80, replace=False)
        points = points[idx]

    rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge)
    st = rips.create_simplex_tree(max_dimension=2)
    st.persistence()

    h1 = st.persistence_intervals_in_dimension(1)
    h1_pers = [d - b for (b, d) in h1 if d != float('inf') and (d - b) > 0.1]

    # pénalité 1 : écart au nombre de cycles attendu
    n_cycles = len(h1_pers)
    cycle_penalty = (n_cycles - target_n_cycles) ** 2

    # pénalité 2 : si on a des cycles --> récompenser la persistance forte
    if len(h1_pers) > 0:
        max_pers = max(h1_pers)
        persistence_bonus = max(0, 1.0 - max_pers)
    else:
        persistence_bonus = 2.0  # pénalité forte si aucun cycle

    return float(cycle_penalty + persistence_bonus)


#architecture Autoencodeur
class TopoAutoEncoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2, hidden_dim=32):
        super(TopoAutoEncoder, self).__init__()
        # encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # decodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# entrainement
def train_autoencoder(data_tensor, use_topo_loss=False, lambda_topo=0.5,
                      epochs=200, lr=0.005):
    """
    entraine l'autoencodeur avec ou sans régularisation topologique

    """
    model = TopoAutoEncoder()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    mode_str = "AVEC" if use_topo_loss else "SANS"
    print(f"\nEntrainement {mode_str} régularisation topologique")

    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_hat, z = model(data_tensor)
        loss_recon = mse_loss(x_hat, data_tensor)

        if use_topo_loss and (epoch % 5 == 0):  # Calcul topo toutes les 5 époques
            l_topo = topological_loss(z, target_n_cycles=1)
            total_loss = loss_recon + lambda_topo * torch.tensor(l_topo)
        else:
            total_loss = loss_recon

        total_loss.backward()
        optimizer.step()
        history.append(total_loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")

    return model, history


def main():

    # 1 : données
    np.random.seed(42)
    data = generate_annulus(n_points=300)
    data_tensor = torch.tensor(data)

    # 2 : entrainement sans regularisation
    model_no_topo, hist_no = train_autoencoder(data_tensor, use_topo_loss=False)

    # 3 : entraînement avec régularisation
    model_with_topo, hist_with = train_autoencoder(data_tensor, use_topo_loss=True)

    # 4: visualisation comparative
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Ligne 1 : SANS topo
    axes[0, 0].scatter(data[:, 0], data[:, 1], s=5, c='steelblue', alpha=0.6)
    axes[0, 0].set_title("Données originales (Anneau)")
    axes[0, 0].axis('equal')

    with torch.no_grad():
        _, z_no = model_no_topo(data_tensor)
        x_hat_no, _ = model_no_topo(data_tensor)
    z_no = z_no.numpy()
    x_hat_no = x_hat_no.numpy()

    axes[0, 1].scatter(z_no[:, 0], z_no[:, 1], s=5, c='tomato', alpha=0.6)
    axes[0, 1].set_title("Espace latent SANS topo-loss")
    axes[0, 1].axis('equal')

    axes[0, 2].scatter(x_hat_no[:, 0], x_hat_no[:, 1], s=5, c='green', alpha=0.6)
    axes[0, 2].set_title("Reconstruction SANS topo-loss")
    axes[0, 2].axis('equal')

    axes[1, 0].plot(hist_no, label='Sans topo', color='tomato')
    axes[1, 0].plot(hist_with, label='Avec topo', color='steelblue')
    axes[1, 0].set_title("Courbes de loss")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)

    with torch.no_grad():
        _, z_with = model_with_topo(data_tensor)
        x_hat_with, _ = model_with_topo(data_tensor)
    z_with = z_with.numpy()
    x_hat_with = x_hat_with.numpy()

    axes[1, 1].scatter(z_with[:, 0], z_with[:, 1], s=5, c='steelblue', alpha=0.6)
    axes[1, 1].set_title("Espace latent AVEC topo-loss")
    axes[1, 1].axis('equal')

    axes[1, 2].scatter(x_hat_with[:, 0], x_hat_with[:, 1], s=5, c='green', alpha=0.6)
    axes[1, 2].set_title("Reconstruction AVEC topo-loss")
    axes[1, 2].axis('equal')

    plt.suptitle("Comparaison : Autoencodeur avec vs sans régularisation topologique",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_filename = "topological_regularization.png"
    plt.savefig(output_filename)
    print(f"   resultats sauvegardé : '{output_filename}'")

    # for name, z_data in [("SANS topo-loss", z_no), ("AVEC topo-loss", z_with)]:
    #     rips = gudhi.RipsComplex(points=z_data[:80], max_edge_length=3.0)
    #     st = rips.create_simplex_tree(max_dimension=2)
    #     st.persistence()
    #     h1 = st.persistence_intervals_in_dimension(1)
    #     h1_sig = [(b, d) for (b, d) in h1 if d != float('inf') and (d - b) > 0.1]
    #     print(f"  {name} : {len(h1_sig)} cycles H1 significatifs")


if __name__ == "__main__":
    main()
