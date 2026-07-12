import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib
import os



CLASS_NAMES = {
    0: 'Normal',
    1: 'Inner Race',
    2: 'Ball',
    3: 'Outer Race',
}



def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    model_name: str = 'Model') -> dict:
    """
    Compute accuracy, precision, recall, and F1-score.

    Parameters
    ----------
    y_true      : true labels
    y_pred      : predicted labels
    model_name  : name shown in the printed output

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n── {model_name} Results ──────────────────────")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"─────────────────────────────────────────────")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def print_classification_report(y_true: np.ndarray,
                                  y_pred: np.ndarray) -> None:
    """
    Print a detailed per-class classification report.
    Shows precision, recall, F1 and support for each fault class.
    """
    n_classes = len(np.unique(y_true))
    target_names = [CLASS_NAMES.get(i, str(i)) for i in range(n_classes)]

    print("\n── Per-Class Classification Report ──────────")
    print(classification_report(y_true, y_pred,
                                  target_names=target_names,
                                  zero_division=0))


def compare_models(results: dict) -> pd.DataFrame:
    """
    Build a comparison table from a dict of model results.

    Parameters
    ----------
    results : dict — {model_name: metrics_dict}
              where metrics_dict comes from compute_metrics()

    Returns
    -------
    pd.DataFrame sorted by F1-Score descending
    """
    rows = []
    for name, m in results.items():
        rows.append({
            'Model':     name,
            'Accuracy':  f"{m['accuracy']*100:.2f}%",
            'Precision': f"{m['precision']:.4f}",
            'Recall':    f"{m['recall']:.4f}",
            'F1-Score':  f"{m['f1']:.4f}",
        })

    df = pd.DataFrame(rows).sort_values('F1-Score', ascending=False)
    df = df.reset_index(drop=True)

    print("\n── Model Comparison ──────────────────────────")
    print(df.to_string(index=False))
    print("─────────────────────────────────────────────\n")

    return df



def plot_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           model_name: str = 'Model',
                           save_path: str = None) -> None:
    """
    Plot a normalized confusion matrix as a heatmap.

    Normalized = each row divided by total true samples in that class,
    so values represent recall per class (0.0 – 1.0).

    Parameters
    ----------
    y_true     : true labels
    y_pred     : predicted labels
    model_name : title shown on the plot
    save_path  : if provided, saves the figure (e.g. 'reports/cm_rf.png')
    """
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    labels    = list(range(n_classes))
    names     = [CLASS_NAMES.get(i, str(i)) for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f',
        xticklabels=names, yticklabels=names,
        cmap='Blues', linewidths=0.5,
        vmin=0, vmax=1, ax=ax,
        annot_kws={'size': 11}
    )

    ax.set_title(f'Confusion Matrix — {model_name}\n(normalized by true class)',
                 fontsize=12, fontweight='bold', pad=14)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved confusion matrix → {save_path}")

    plt.show()



def plot_roc_curves(model,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    model_name: str = 'Model',
                    save_path: str = None) -> None:
    """
    Plot One-vs-Rest ROC curves for all classes.

    Requires the model to have a predict_proba() method
    (Logistic Regression, Random Forest, XGBoost all do).

    Parameters
    ----------
    model      : fitted sklearn-compatible classifier
    X_test     : test features
    y_test     : true labels
    model_name : title shown on the plot
    save_path  : optional path to save the figure
    """
    classes  = sorted(np.unique(y_test))
    n_classes = len(classes)
    names    = [CLASS_NAMES.get(i, str(i)) for i in classes]

    
    y_bin   = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    colors = ['#3d6aff', '#00d4aa', '#f0a147', '#c17cff', '#ff6b7a']
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (cls, name) in enumerate(zip(classes, names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                color=colors[i % len(colors)],
                lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curves (One-vs-Rest) — {model_name}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved ROC curves → {save_path}")

    plt.show()



def plot_shap_summary(model,
                      X_test: np.ndarray,
                      feature_names: list,
                      save_path: str = None) -> None:
    """
    Generate SHAP feature importance plots.

    Produces two plots:
      1. Bar chart — global mean importance across all classes
      2. Beeswarm — shows importance magnitude AND direction per class

    Requires: pip install shap
    Works best with tree-based models (Random Forest, XGBoost).

    Parameters
    ----------
    model         : fitted tree-based classifier
    X_test        : test feature matrix (numpy array)
    feature_names : list of feature name strings
    save_path     : optional folder path to save both figures
    """
    try:
        import shap
    except ImportError:
        print("[ERROR] shap not installed. Run: pip install shap")
        return

    print("  Computing SHAP values (this may take 30–60 seconds)...")

    X_df     = pd.DataFrame(X_test, columns=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_df,
                      feature_names=feature_names,
                      plot_type='bar',
                      show=False)
    plt.title('SHAP Feature Importance (Global)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fpath = os.path.join(save_path, 'shap_bar.png')
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"  Saved SHAP bar chart → {fpath}")

    plt.show()

    # ── Plot 2: Beeswarm for class 1 (Inner Race) ─
    if isinstance(shap_values, list) and len(shap_values) > 1:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1], X_df,
                          feature_names=feature_names,
                          show=False)
        plt.title('SHAP Beeswarm — Inner Race Fault (Class 1)',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fpath = os.path.join(save_path, 'shap_beeswarm_class1.png')
            plt.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f"  Saved SHAP beeswarm → {fpath}")

        plt.show()




def save_model(model, model_name: str, models_dir: str = 'models') -> str:
    """
    Save a trained model to disk as a .pkl file.

    Parameters
    ----------
    model      : fitted sklearn-compatible model
    model_name : name used for the filename (e.g. 'random_forest')
    models_dir : folder to save into (default 'models/')

    Returns
    -------
    str : path where the model was saved
    """
    os.makedirs(models_dir, exist_ok=True)
    fpath = os.path.join(models_dir, f'{model_name}.pkl')
    joblib.dump(model, fpath)
    print(f"  ✓ Model saved → {fpath}")
    return fpath


def load_model(model_name: str, models_dir: str = 'models'):
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    model_name : name used when saving (e.g. 'random_forest')
    models_dir : folder where models are stored

    Returns
    -------
    fitted sklearn-compatible model
    """
    fpath = os.path.join(models_dir, f'{model_name}.pkl')

    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Model file not found: '{fpath}'. "
            f"Make sure you trained and saved the model first."
        )

    model = joblib.load(fpath)
    print(f"  ✓ Model loaded ← {fpath}")
    return model
