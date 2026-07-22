import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_openml, load_digits
import umap

# 1. Configuration de la page Streamlit
st.set_page_config(
    page_title="TP-UMAP MNIST - EL MEHDI",
    layout="centered",
    initial_sidebar_state="expanded",
)

# En-tête personnalisé avec design soigné (Bleu pour UMAP)
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


# 2. Chargement sécurisé avec repli anti-plantage réseau
@st.cache_data(show_spinner=False)
def load_mnist_data():
    try:
        mnist = fetch_openml("mnist_784", version=1, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        y = mnist.target.astype(np.int64)
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        return X, y
    except Exception:
        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target.astype(np.int64)
        return X, y


try:
    with st.spinner("Chargement des données MNIST..."):
        X, y = load_mnist_data()
except Exception as e:
    st.error(f"Erreur critique lors du chargement : {e}")
    st.stop()


# 3. Panneau de configuration (Sidebar épurée)
st.sidebar.header("Paramètres UMAP")

max_samples = min(len(X), 5000)
sample_size = st.sidebar.slider(
    "Taille de l'échantillon (vitesse)",
    min_value=500,
    max_value=max_samples,
    value=2000,
    step=500,
)

n_neighbors = st.sidebar.slider(
    "Nombre de voisins (n_neighbors)", min_value=2, max_value=50, value=15
)
min_dist = st.sidebar.slider(
    "Distance minimale (min_dist)", min_value=0.0, max_value=0.9, value=0.1
)

# Graine fixée en arrière-plan pour garantir la stabilité sans encombrer l'interface
RANDOM_SEED = 42


# 4. Sous-échantillonnage reproductible et mis en cache
@st.cache_data(show_spinner=False)
def get_subset(X, y, sample_size, seed):
    np.random.seed(seed)
    actual_size = min(sample_size, len(X))
    indices = np.random.choice(len(X), actual_size, replace=False)
    return X[indices], y[indices]


X_subset, y_subset = get_subset(X, y, sample_size, RANDOM_SEED)


# 5. Mise en cache de l'algorithme UMAP
@st.cache_data(show_spinner=False)
def compute_umap(X_data, n_neigh, m_dist, seed):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neigh,
        min_dist=m_dist,
        random_state=seed,
    )
    return reducer.fit_transform(X_data)


# 6. Exécution et affichage de la projection
st.subheader("1. Projection UMAP en 2D")

with st.spinner("Exécution de la projection UMAP (calcul ultra-rapide)..."):
    try:
        embedding = compute_umap(
            X_subset, n_neighbors, min_dist, RANDOM_SEED
        )
    except Exception as e:
        st.error(f"Erreur durant l'exécution d'UMAP : {e}")
        st.stop()

# 7. Création du graphique Matplotlib haut de gamme
fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=y_subset,
    cmap="tab10",
    s=12,
    alpha=0.75,
    edgecolors="none",
)

legend = ax.legend(
    *scatter.legend_elements(),
    title="Chiffres",
    loc="upper right",
    bbox_to_anchor=(1.22, 1),
    frameon=True,
    facecolor="#f8fafc",
    edgecolor="#cbd5e1",
)
ax.add_artist(legend)

ax.set_title(
    "Projection UMAP des chiffres MNIST (Espace 2D)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Dimension UMAP 1", fontsize=10)
ax.set_ylabel("Dimension UMAP 2", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

st.pyplot(fig, bbox_inches="tight")

# 8. Section explicative
st.markdown("---")
st.markdown("### 💡 Ce que montre ce graphique :")
st.markdown(
    """
* **Performance et fluidité :** L'algorithme UMAP s'exécute très rapidement, soutenu par un cache intelligent pour éviter les recalculs superflus.
* **Structure topologique :** Contrairement à la PCA linéaire, **UMAP** modélise les voisinages locaux pour regrouper les chiffres similaires tout en préservant la structure globale du dataset.
"""
)
