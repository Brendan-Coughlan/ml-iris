from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

iris = load_iris()

X = iris.data
y = iris.target

pipeline = Pipeline(
    [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))]
)

kfold = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(pipeline, X, y, cv=kfold, scoring="accuracy")

print("Accuracy scores for each fold:", scores)
print("Mean Accuracy:", np.mean(scores))