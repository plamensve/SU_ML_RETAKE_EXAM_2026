from matplotlib import pyplot as plt
import seaborn as sns


def plot_correlation_matrix(correlation_matrix, method) -> None:
    """
    :param correlation_matrix: pd.Dataframe
    :param method: String
    :return: None
    """
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Correlation"}
    )

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.title(f"Correlation Matrix [{method}]")
    plt.show()