from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from evaluator import evaluate_model, get_confusion_matrix

iris = load_iris()

X = iris.data
y = iris.target

pipeline = Pipeline(
    [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))]
)

evaluate_model(pipeline, X, y)
get_confusion_matrix(pipeline, X, y)