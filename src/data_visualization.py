import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

def generate_scatter_plot(iris_df):
    sns.scatterplot(data=iris_df, x="sepal length (cm)", y="sepal width (cm)", hue="species", palette="tab20")
    plt.savefig("images/scatter_sepal_size.png")

    plt.close()

    sns.scatterplot(data=iris_df, x="petal length (cm)", y="petal width (cm)", hue="species", palette="tab20")
    plt.savefig("images/scatter_petal_size.png")

    plt.close()

def generate_pair_plot(iris_df):
    sns.pairplot(iris_df, hue="species", palette="Set2")
    plt.savefig("images/pairplot.png")

    plt.close()

def generate_correlation_heatmap(iris_df):
    sns.heatmap(iris_df.drop(columns=["species"]).corr(), cmap="coolwarm", annot=True)
    plt.savefig("images/correlation_heatmap.png")

    plt.close()

def generate_distributions(iris_df):
    target_names = {
        0: "setosa",
        1: "versicolor",
        2: "virginica",
    }
        
    features = {
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
    }

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    for col, (target, species_name) in enumerate(target_names.items()):
            species = iris_df[iris_df["target"] == target]

            for row, feature in enumerate(features):
                ax = axes[row, col]
                sns.histplot(
                    data=species,
                    x=feature,
                    kde=True,
                    ax=ax
                )
                ax.set_title(species_name)
                ax.set_xlabel(feature)

    plt.tight_layout()
    plt.savefig("images/distributions_4x3.png")
    plt.close()

iris = load_iris(as_frame=True)
df = iris.frame

df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

generate_scatter_plot(df)
generate_pair_plot(df)
generate_correlation_heatmap(df)
generate_distributions(df)