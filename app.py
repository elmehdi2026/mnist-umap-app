import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import fetch_openml
import umap

# Configuration de la page Streamlit
st.set_page_config(
    page_title="TP-UMAP MNIST", layout="centered", initial_sidebar_state="expanded"
)

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white; padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 25px;">
        <div style="font-size: 11pt; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; opacity: 0.85;">EL MEHDI - Master IAENG</div>
        <h1 style="margin: 10px 0 0 0; font-size: 24pt;">TP-UMAP</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Réduction de Dimension non linéaire & Visualisation des clusters MNIST</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Chargement optimisé des données MNIST converties directement en tableaux NumPy
@st.cache_data
def load_mnist_data():
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    X = mnist.data.to_numpy().astype("float32") / 255.0
    y = mnist.target.to_numpy().astype("int")
    return X, y


with st.spinner("Chargement du dataset MNIST en cours (veuillez patienter)..."):
    X, y = load_mnist_data()

# Panneau de configuration (Sidebar)
st.sidebar.header("Paramètres UMAP")
sample_size = st.sidebar.slider(
    "Taille de l'échantillon (vitesse)",
    min_value=1000,
    max_value=7000,
    value=3000,
    step=1000,
)
n_neighbors = st.sidebar.slider(
    "Nombre de voisins (n_neighbors)", min_value=2, max_value=50, value=15
)
min_dist = st.sidebar.slider(
    "Distance minimale (min_dist)", min_value=0.0, max_value=0.9, value=0.1
)

# Réduction d'échantillon sécurisée avec NumPy
indices = np.random.choice(len(X), sample_size, replace=False)
X_subset = X[indices]
y_subset = y[indices]


# Application de l'algorithme UMAP avec mise en cache pour la vitesse
@st.cache_resource
def compute_umap(X_data, n_neigh, m_dist):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neigh,
        min_dist=m_dist,
        random_state=42,
    )
    return reducer.fit_transform(X_data)


st.subheader("1. Projection UMAP en 2D")
with st.spinner("Exécution de la projection UMAP..."):
    embedding = compute_umap(X_subset, n_neighbors, min_dist)

# Affichage du graphique interactif Matplotlib (avec légende décalée et ajustement anti-coupure)
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=y_subset,
    cmap="tab10",
    s=10,
    alpha=0.7,
)
legend = ax.legend(
    *scatter.legend_elements(),
    title="Chiffres",
    loc="upper right",
    bbox_to_anchor=(1.25, 1),
)
ax.add_artist(legend)
ax.set_title(
    "Projection UMAP des chiffres MNIST (Espace 2D)",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel("Dimension UMAP 1")
ax.set_ylabel("Dimension UMAP 2")
ax.grid(True, linestyle="--", alpha=0.5)

# Utilisation de bbox_inches='tight' pour empêcher Streamlit de rogner les éléments extérieurs (légende)
st.pyplot(fig, bbox_inches="tight")

st.markdown("---")
st.markdown("### 💡 Ce que montre ce graphique :")
st.markdown(
    "- Contrairement à la PCA qui est linéaire, **UMAP** modélise les voisinages locaux pour regrouper les chiffres similaires (ex: tous les '0' ensemble, tous les '1' ensemble) tout en respectant la structure globale du dataset[cite: 2]."
)
