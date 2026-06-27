from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from evaluator import evaluate_model, get_confusion_matrix

iris = load_iris()

X = iris.data
y = iris.target

pipeline = Pipeline(
    [("scaler", StandardScaler()), ("svm", SVC(random_state=42))]
)

evaluate_model(pipeline, X, y)
get_confusion_matrix(pipeline, X, y, "SVM")