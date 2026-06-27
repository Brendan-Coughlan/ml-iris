from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, X, y):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        model,
        X,
        y,
        cv=kfold,
        scoring={
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average="macro"),
            "recall": make_scorer(recall_score, average="macro"),
            "f1": make_scorer(f1_score, average="macro"),
        },
    )
    print(
        f"Accuracy: {scores['test_accuracy'].mean():.2f} ± {scores['test_accuracy'].std():.2f}"
    )
    print(
        f"Precision: {scores['test_precision'].mean():.2f} ± {scores['test_precision'].std():.2f}"
    )
    print(
        f"Recall: {scores['test_recall'].mean():.2f} ± {scores['test_recall'].std():.2f}"
    )
    print(f"F1 Score: {scores['test_f1'].mean():.2f} ± {scores['test_f1'].std():.2f}")


def get_confusion_matrix(model, X, y, model_name):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=kfold)

    cm = confusion_matrix(y, y_pred)
    print(cm)

    target_names = [
        "setosa",
        "versicolor",
        "virginica",
    ]

    sns.heatmap(
        cm, cmap="Blues", annot=True, xticklabels=target_names, yticklabels=target_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    
    plt.savefig(f"images/confusion_matrix_{model_name.lower()}.png")
