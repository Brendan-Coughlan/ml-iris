from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from evaluator import evaluate_model

iris = load_iris()

X = iris.data
y = iris.target

pipeline = Pipeline(
    [("scaler", StandardScaler()), ("svm", SVC())]
)

evaluate_model(pipeline, X, y)