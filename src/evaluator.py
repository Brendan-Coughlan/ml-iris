from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

def evaluate_model(model, X, y):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")

    print(f"Mean Accuracy: {np.round(np.mean(scores)*100, 2)}%")