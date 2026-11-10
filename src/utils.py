import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve

def check_data_quality(df):
    """Check data quality.

Parameters
----------
df
    DataFrame a examiner.

Returns
-------
missing (Series), duplicates (int)."""
    missing = df.isna().sum()
    duplicates = df.duplicated().sum()
    return (missing, duplicates)

def plot_data(X, title='Distribution des observations'):
    """Plot data.

Parameters
----------
X
    DataFrame des observations ; colonnes x et y pour les graphiques.
title
    Titre du graphique.

Returns
-------
None : affiche le graphique."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X['x'], X['y'], s=15, alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

def prepare_labels(n_samples, n_anomalies):
    """Prepare labels.

Parameters
----------
n_samples
    Nombre total de lignes.
n_anomalies
    Nombre de dernieres lignes a etiqueter 1, entre 0 et n_samples.

Returns
-------
Vecteur entier : 0 normale, 1 anomalie."""
    if not 0 <= n_anomalies <= n_samples:
        raise ValueError("Invalid anomaly count")
    y = np.zeros(n_samples, dtype=int)
    if n_anomalies:
        y[-n_anomalies:] = 1
    return y


def fit_model(model, X, y=None):
    """
    Ajuste un modèle et mesure le temps d'apprentissage.

    Parameters
    ----------
    model : estimator
        Modèle scikit-learn.
    X : array-like
        Variables explicatives.
    y : array-like, optional
        Étiquettes pour les modèles supervisés.

    Returns
    -------
    tuple
        Modèle entraîné et temps d'apprentissage (secondes).
    """
    start = time.perf_counter()

    if y is None:
        model.fit(X)
    else:
        model.fit(X, y)

    temps_fit = time.perf_counter() - start
    return model, temps_fit


def compute_anomaly_scores(model, X):
    """Compute anomaly scores.

Parameters
----------
model
    Estimateur scikit-learn compatible avec la methode appelee.
X
    DataFrame des observations ; colonnes x et y pour les graphiques.

Returns
-------
Tuple : scores, temps de calcul en secondes.

Notes
-----
LOF outlier : X doit etre exactement les donnees du fit, dans le meme ordre.
Le temps de score mesure extraction et inversion de negative_outlier_factor_.
Le calcul des facteurs est inclus dans le temps de fit.
LOF novelty : utiliser de nouvelles observations pour score_samples."""
    start = time.perf_counter()
    if hasattr(model, 'negative_outlier_factor_') and not getattr(model, 'novelty', False):
        if len(X) != len(model.negative_outlier_factor_):
            raise ValueError('Use the training observations in their original order')
        scores = -np.asarray(model.negative_outlier_factor_)
    elif hasattr(model, "decision_function") and not hasattr(model, "score_samples"):
        # ECOD : score élevé = anomalie
        scores = np.asarray(model.decision_function(X))

    else:
        scores = -np.asarray(model.score_samples(X))
    score_time = time.perf_counter() - start
    return (scores, score_time)

def predict_anomalies(model, X):
    """Predict anomalies.

Parameters
----------
model
    Estimateur scikit-learn compatible avec la methode appelee.
X
    DataFrame des observations ; colonnes x et y pour les graphiques.

Returns
-------
Vecteur entier : 0 normale, 1 anomalie."""
    predictions = model.predict(X)
    return (predictions == -1).astype(int)

def plot_score_distribution(scores, title='Distribution des scores', threshold=None, ax=None):
    """Plot score distribution.

Parameters
----------
scores
    Vecteur de scores : valeur elevee = plus anormal.
title
    Titre du graphique.
threshold
    Seuil optionnel a afficher.
ax
    Axe matplotlib optionnel ; cree si None.

Returns
-------
Axe matplotlib contenant le graphique."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=30, edgecolor='black', alpha=0.7)
    if threshold is not None:
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Seuil = {threshold:.4f}')
        ax.legend()
    ax.set_xlabel("Score d'anomalie")
    ax.set_ylabel("Nombre d'observations")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax

def plot_detected_anomalies(X, predictions, title='Anomalies détectées', ax=None):
    """Plot detected anomalies.

Parameters
----------
X
    DataFrame des observations ; colonnes x et y pour les graphiques.
predictions
    Vecteur binaire : 0 normale, 1 anomalie.
title
    Titre du graphique.
ax
    Axe matplotlib optionnel ; cree si None.

Returns
-------
Axe matplotlib contenant le graphique."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    predictions = np.asarray(predictions)
    normales = predictions == 0
    anomalies = predictions == 1
    ax.scatter(X.loc[normales, 'x'], X.loc[normales, 'y'], s=15, alpha=0.7, label='Normales (0)')
    ax.scatter(X.loc[anomalies, 'x'], X.loc[anomalies, 'y'], s=40, alpha=0.9, marker='x', color='red', label='Anomalies (1)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax

def plot_score_scatter(X, scores, title="Scores d'anomalie", ax=None):
    """Plot score scatter.

Parameters
----------
X
    DataFrame des observations ; colonnes x et y pour les graphiques.
scores
    Vecteur de scores : valeur elevee = plus anormal.
title
    Titre du graphique.
ax
    Axe matplotlib optionnel ; cree si None.

Returns
-------
Axe matplotlib contenant le graphique."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X['x'], X['y'], c=scores, cmap='viridis', s=20, alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.figure.colorbar(scatter, ax=ax, label="Score d'anomalie")
    return ax

def iqr_threshold(scores, factor=1.5):
    """Iqr threshold.

Parameters
----------
scores
    Vecteur de scores : valeur elevee = plus anormal.
factor
    Multiplicateur de IQR.

Returns
-------
Seuil Q3 + factor * IQR (float)."""
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    threshold = q3 + factor * iqr
    return threshold

def plot_scores_points(scores, title="Scores d'anomalie", ax=None):
    """Plot scores points.

Parameters
----------
scores
    Vecteur de scores : valeur elevee = plus anormal.
title
    Titre du graphique.
ax
    Axe matplotlib optionnel ; cree si None.

Returns
-------
Axe matplotlib contenant le graphique."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(np.arange(len(scores)), scores, s=15, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Indice de l'observation")
    ax.set_ylabel("Score d'anomalie")
    ax.grid(alpha=0.3)
    return ax

def kmeans_score_clustering(scores, n_clusters=2, random_state=42):
    """Kmeans score clustering.

Parameters
----------
scores
    Vecteur de scores : valeur elevee = plus anormal.
n_clusters
    Nombre de groupes ; celui de moyenne maximale est anormal.
random_state
    Graine aleatoire.

Returns
-------
Tuple : modele ajuste, predictions binaires."""
    scores_2d = np.asarray(scores).reshape(-1, 1)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = model.fit_predict(scores_2d)
    cluster_means = [scores_2d[clusters == i].mean() for i in range(n_clusters)]
    anomaly_cluster = np.argmax(cluster_means)
    predictions = (clusters == anomaly_cluster).astype(int)
    return (model, predictions)

def plot_clustered_scores(scores, predictions, title='', ax=None):
    """Plot clustered scores.

Parameters
----------
scores
    Vecteur de scores : valeur elevee = plus anormal.
predictions
    Vecteur binaire : 0 normale, 1 anomalie.
title
    Titre du graphique.
ax
    Axe matplotlib optionnel ; cree si None.

Returns
-------
Axe matplotlib contenant le graphique."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    scores = np.asarray(scores)
    indices = np.arange(len(scores))
    predictions = np.asarray(predictions)
    normales = predictions == 0
    anomalies = predictions == 1
    ax.scatter(indices[normales], scores[normales], s=15, alpha=0.7, label='Normales (0)')
    ax.scatter(indices[anomalies], scores[anomalies], s=20, color='red', alpha=0.8, label='Anomalies (1)')
    ax.set_title(title)
    ax.set_xlabel("Indice de l'observation")
    ax.set_ylabel("Score d'anomalie")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax

def dbscan_score_clustering(scores, eps=0.05, min_samples=5):
    """Dbscan score clustering.

Parameters
----------
scores
    Vecteur de scores : valeur elevee = plus anormal.
eps
    Rayon DBSCAN sur les scores non normalises.
min_samples
    Effectif minimal de voisinage, point inclus.

Returns
-------
Tuple : modele ajuste, predictions binaires (bruit = 1).

Notes
-----
Le bruit DBSCAN peut concerner aussi les scores faibles."""
    scores_2d = np.asarray(scores).reshape(-1, 1)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(scores_2d)
    predictions = (clusters == -1).astype(int)
    return (model, predictions)

def evaluate_anomalies(y_true, y_pred):
    """Evaluate anomalies.

Parameters
----------
y_true
    Etiquettes reelles binaires.
y_pred
    Etiquettes predites binaires.

Returns
-------
Dictionnaire : TN, FP, FN, TP, F1, Matrice, Indices anomalies."""
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    return {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'F1': f1, 'Matrice': cm, 'Indices anomalies': np.flatnonzero(y_pred == 1)}

def plot_confusion_matrix(cm, title='', ax=None):
    """Plot confusion matrix.

Parameters
----------
cm
    Matrice 2 x 2, lignes reelles et colonnes predites, ordre [0, 1].
title
    Titre du graphique.
ax
    Axe matplotlib optionnel ; cree si None.

Returns
-------
Axe matplotlib contenant la matrice."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normale (0)', 'Anomalie (1)'], yticklabels=['Normale (0)', 'Anomalie (1)'], ax=ax)
    ax.set_xlabel('Classe prédite')
    ax.set_ylabel('Classe réelle')
    ax.set_title(title)
    return ax

def evaluate_scores(y_true, scores):
    """Evaluate scores.

Parameters
----------
y_true
    Etiquettes reelles binaires.
scores
    Vecteur de scores : valeur elevee = plus anormal.

Returns
-------
Tuple : precision moyenne AP, aire ROC AUC. Deux classes requises."""
    ap = average_precision_score(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    return (ap, auc)

def split_novelty_data(X, y, test_size=0.2, random_state=42):
    """Split novelty data.

Parameters
----------
X
    DataFrame des observations ; colonnes x et y pour les graphiques.
y
    Etiquettes binaires dans le meme ordre que X.
test_size
    Part ou nombre de normales reservees au test.
random_state
    Graine aleatoire.

Returns
-------
Tuple : X_train, X_test, y_test. Apprentissage normal uniquement."""
    y = np.asarray(y)
    X_normal = X[y == 0]
    X_anomaly = X[y == 1]
    X_train, X_test_normal = train_test_split(X_normal, test_size=test_size, random_state=random_state)
    X_test = pd.concat([X_test_normal, X_anomaly])
    y_test = np.concatenate([np.zeros(len(X_test_normal), dtype=int), np.ones(len(X_anomaly), dtype=int)])
    return (X_train, X_test, y_test)

def plot_novelty_split(X_train, X_test, y_test):
    """Plot novelty split.

Parameters
----------
X_train
    DataFrame des normales utilisees pour le fit.
X_test
    DataFrame de test, normales puis anomalies.
y_test
    Etiquettes binaires dans le meme ordre que X_test.

Returns
-------
None : affiche les deux graphiques."""
    y_test = np.asarray(y_test)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(X_train['x'], X_train['y'], s=15, alpha=0.7, label=f"Normales ({len(X_train)})")
    axes[0].set_title('Apprentissage — Normales uniquement')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    normales = y_test == 0
    anomalies = y_test == 1
    axes[1].scatter(X_test.loc[normales, 'x'], X_test.loc[normales, 'y'], s=15, alpha=0.7, label=f"Normales ({np.count_nonzero(y_test == 0)})")
    axes[1].scatter(X_test.loc[anomalies, 'x'], X_test.loc[anomalies, 'y'], s=40, color='red', marker='x', label=f"Anomalies ({np.count_nonzero(y_test == 1)})")
    axes[1].set_title('Test — Normales et anomalies')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_class_distribution(y, title="Répartition des classes"):
    """
    Affiche la répartition des classes avec leurs effectifs.

    Parameters
    ----------
    y : array-like
        Étiquettes des observations (0 = normal, 1 = anomalie).
    title : str
        Titre du graphique.

    Returns
    -------
    None
    """
    counts = pd.Series(y).value_counts().reindex([0, 1], fill_value=0)

    ax = counts.plot(kind="bar", figsize=(7, 4), color=["steelblue", "salmon"])

    for i, count in enumerate(counts):
        ax.text(i, count, f"{count:,}", ha="center", va="bottom")

    ax.set_xticklabels(["Normales (0)", "Fraudes (1)"], rotation=0)
    ax.set_ylabel("Nombre de transactions")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()





def split_stratified_groups(X, y, random_state=42):
    """
    Découpage 60/20/20 par groupes de prédicteurs identiques.
    Stratification sur les classes des groupes.
    """
    # Identifier les groupes de prédicteurs identiques
    groups = pd.factorize(pd.MultiIndex.from_frame(X))[0]

    group_df = pd.DataFrame({
        "group": groups,
        "Class": np.asarray(y)
    })

    # Une ligne par groupe
    unique_groups = group_df.drop_duplicates("group")

    # 60 % train, 40 % temporaire
    g_train, g_temp = train_test_split(
        unique_groups["group"],
        test_size=0.4,
        random_state=random_state,
        stratify=unique_groups["Class"]
    )

    # 20 % validation, 20 % test
    temp = unique_groups.set_index("group").loc[g_temp]

    g_val, g_test = train_test_split(
        temp.index,
        test_size=0.5,
        random_state=random_state,
        stratify=temp["Class"]
    )

    # Reconstituer les partitions
    train_mask = np.isin(groups, g_train)
    val_mask = np.isin(groups, g_val)
    test_mask = np.isin(groups, g_test)

    return (
        X.loc[train_mask],
        X.loc[val_mask],
        X.loc[test_mask],
        y.loc[train_mask],
        y.loc[val_mask],
        y.loc[test_mask]
    )



def standardize_data(X_train, X_val, X_test):
    """
    Standardise les variables à partir du train uniquement.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Partitions des données.

    Returns
    -------
    tuple
        X_train_scaled, X_val_scaled, X_test_scaled, scaler.
    """
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler




def plot_roc_pr_curves(y_true, scores_dict):
    """
    Trace les courbes ROC et précision-rappel de plusieurs modèles.

    Parameters
    ----------
    y_true : array-like
        Étiquettes réelles (0 = normale, 1 = fraude).
    scores_dict : dict
        Dictionnaire {nom_modele: scores continus}.
        Score élevé = observation plus suspecte.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for nom, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)

        ap, auc = evaluate_scores(y_true, scores)

        axes[0].plot(fpr, tpr, label=f"{nom} (AUC={auc:.3f})")
        axes[1].plot(recall, precision, label=f"{nom} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set(
        title="Courbes ROC",
        xlabel="Taux de faux positifs",
        ylabel="Rappel (TPR)"
    )

    axes[1].axhline(
        np.mean(y_true), color="gray", linestyle="--",
        label="Prévalence des fraudes"
    )
    axes[1].set(
        title="Courbes précision-rappel",
        xlabel="Rappel",
        ylabel="Précision"
    )

    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



def plot_metrics_threshold(y_true, scores_dict, title=""):
    """
    Trace F1 et balanced accuracy en fonction du seuil.

    Parameters
    ----------
    y_true : array-like
        Étiquettes réelles (0 = normale, 1 = fraude).
    scores_dict : dict
        Dictionnaire {nom: scores continus}.
        Score élevé = observation plus suspecte.
    title : str
        Titre de la figure.

    Returns
    -------
    None
    """
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for nom, scores in scores_dict.items():
        fpr, tpr, seuils = roc_curve(
            y_true, scores, drop_intermediate=False
        )

        # Ignorer le seuil infini
        fpr, tpr, seuils = fpr[1:], tpr[1:], seuils[1:]

        # Calcul des métriques
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        tp = tpr * n_pos
        fp = fpr * n_neg
        fn = n_pos - tp

        f1 = np.divide(
            2 * tp, 2 * tp + fp + fn,
            out=np.zeros_like(tp),
            where=(2 * tp + fp + fn) > 0
        )

        balanced_acc = (tpr + 1 - fpr) / 2

        axes[0].plot(seuils, f1, label=nom)
        axes[1].plot(seuils, balanced_acc, label=nom)

    axes[0].set(
        title="F1-score selon le seuil",
        xlabel="Seuil",
        ylabel="F1-score"
    )

    axes[1].set(
        title="Balanced accuracy selon le seuil",
        xlabel="Seuil",
        ylabel="Balanced accuracy"
    )

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()



def select_threshold(y_true, scores, min_precision=0.80):
    """
    Sélectionne le seuil maximisant le rappel sous
    une contrainte minimale de précision.

    Parameters
    ----------
    y_true : array-like
        Étiquettes réelles (0 = normale, 1 = fraude).
    scores : array-like
        Scores continus, élevés pour les anomalies.
    min_precision : float
        Précision minimale exigée.

    Returns
    -------
    dict
        Seuil, précision, rappel et nombre d'alertes.
        None si aucun seuil ne satisfait la contrainte.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true, scores
    )

    # Le dernier point n'a pas de seuil associé
    precision = precision[:-1]
    recall = recall[:-1]

    # Seuils admissibles avec au moins une fraude détectée
    valid = np.where(
        (precision >= min_precision) & (recall > 0)
    )[0]

    if len(valid) == 0:
        return None

    # Maximiser le rappel, puis la précision
    best = sorted(
        valid,
        key=lambda i: (recall[i], precision[i], thresholds[i]),
        reverse=True
    )[0]

    seuil = thresholds[best]
    predictions = np.asarray(scores) >= seuil

    return {
        "Seuil": seuil,
        "Précision": precision[best],
        "Rappel": recall[best],
        "Alertes": int(predictions.sum())
    }



def compute_supervised_scores(model, X):
    """
    Calcule les scores de fraude et mesure le temps de calcul.

    Parameters
    ----------
    model : estimator
        Modèle supervisé entraîné.
    X : pd.DataFrame
        Données à évaluer.

    Returns
    -------
    tuple
        Scores de la classe 1 et temps de calcul en secondes.
    """
    start = time.perf_counter()
    scores = model.predict_proba(X)[:, 1]
    temps_score = time.perf_counter() - start

    return scores, temps_score


def contaminate_training_data(X_train, contamination_rate, random_state=42):
    """
    Remplace des normales par des anomalies synthétiques.

    Les anomalies sont générées autour de Mouse, dans
    des zones peu denses, sans utiliser les données de test.
    """
    from sklearn.neighbors import NearestNeighbors

    if not 0 <= contamination_rate < 1:
        raise ValueError("La contamination doit être entre 0 et 1 exclu.")

    rng = np.random.default_rng(random_state)
    X_contaminated = X_train.copy()

    n_samples = len(X_train)
    n_anomalies = round(n_samples * contamination_rate)

    if n_anomalies == 0:
        return X_contaminated

    # Référence normale
    values = X_train.to_numpy()
    center = values.mean(axis=0)
    ranges = np.ptp(values, axis=0)

    if values.shape[1] != 2 or np.any(ranges <= 0):
        raise ValueError("Deux variables non constantes sont nécessaires.")

    # Distance locale entre observations normales
    knn = NearestNeighbors(n_neighbors=6).fit(values)
    distances, _ = knn.kneighbors(values)

    # Seuil de distance basé uniquement sur l'apprentissage
    seuil = np.quantile(distances[:, -1], 0.95)

    # Générer des candidats autour de la distribution
    anomalies = []
    max_attempts = 100000
    attempts = 0

    while len(anomalies) < n_anomalies and attempts < max_attempts:
        attempts += 1

        angle = rng.uniform(0, 2 * np.pi)
        rayon = rng.uniform(0.45, 0.85)

        candidat = center + rayon * ranges * np.array([
            np.cos(angle), np.sin(angle)
        ])

        # Éviter les zones proches des normales
        distance, _ = knn.kneighbors(candidat.reshape(1, -1))

        if distance[0, -1] > seuil:
            anomalies.append(candidat)

    if len(anomalies) < n_anomalies:
        raise RuntimeError("Impossible de générer assez d'anomalies.")

    # Remplacer les normales sélectionnées
    indices = rng.choice(n_samples, size=n_anomalies, replace=False)
    X_contaminated.iloc[indices] = np.asarray(anomalies)

    return X_contaminated
