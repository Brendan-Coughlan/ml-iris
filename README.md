# Iris Machine Learning Project

| Setosa | Versicolor | Virginica |
|--------|------------|-----------|
| ![](images/iris_setosa.jpg) | ![](images/iris_versicolor.jpg) | ![](images/iris_virginica.jpg) |

## About The Project
The Iris dataset is one of the most well-known datasets in machine learning. This project provides a comprehensive exploration of the dataset through exploratory data analysis (EDA), data visualization, feature engineering, and machine learning models to classify iris flower species.


## Dataset
This project makes use of the famous Iris dataset, first used by Sir R.A. Fisher in his 1936 paper.

The dataset contains 150 instances of iris flowers across three distinct species:
- Setosa
- Versicolor
- Virginica

Each sample contains four features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Results

*Results are reported as mean ± standard deviation across 5-fold cross-validation*

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|---------|---------|---------|---------|
| Logistic Regression | 95.33% ± 4.52% | 95.49% ± 4.43% | 95.33% ± 4.52% | 95.32% ± 4.53% |
| Decision Tree | 95.33% ± 3.40% | 95.72% ± 3.26% | 95.33% ± 3.40% | 95.31% ± 3.41% |
| Random Forest | 95.33% ± 3.40% | 95.72% ± 3.26% | 95.33% ± 3.40% | 95.31% ± 3.41% |
| K-Nearest Neighbors | 97.33% ± 2.49% | 97.45% ± 2.47% | 97.33% ± 2.49% | 97.33% ± 2.50% |
| Support Vector Machine | 96.00% ± 3.89% | 96.11% ± 3.83% | 96.00% ± 3.89% | 95.99% ± 3.89% |

## Figures
### Petal Measurements

![Petal Size Scatter Plot](images/scatter_petal_size.png)

Petal length and width provide strong separation between the three species.

### Sepal Measurements

![Sepal Size Scatter Plot](images/scatter_sepal_size.png)

Sepal measurements show greater overlap, especially between Versicolor and Virginica.

### Pair Plot

![Pair Plot](images/pairplot.png)

The pair plot visualizes every feature combination and highlights that petal measurements are the most informative features.

### Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)

### Feature Distributions

![Distribution](images/feature_distributions.png)

### Confusion Matrices

![Logistic Regression Confusion Matrix](images/confusion_matrix_logistic_regression.png)
![Decision Tree Confusion Matrix](images/confusion_matrix_decision_tree.png)
![Random Forest Confusion Matrix](images/confusion_matrix_random_forest.png)
![KNN Confusion Matrix](images/confusion_matrix_knn.png)
![SVM Confusion Matrix](images/confusion_matrix_svm.png)

Most misclassifications occur between versicolor and virginica, which is expected given the overlap observed in the pair plot.