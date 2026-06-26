import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

df = pd.concat([X, y], axis=1)

# sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='target')
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='target')
plt.show()