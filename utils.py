import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()


def plot(df, x, y, hue=None, title='Ticker', x_label=None, y_label=None, save_name=None):
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue)
    ax.set_title(title)
    plt.xticks(rotation=45)

    if x_label:
        ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if save_name:
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plt.savefig(f"plots/{save_name}.svg", bbox_inches="tight")
    plt.show()
