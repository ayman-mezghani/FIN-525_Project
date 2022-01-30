import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()


def plot(df, x, y, hue=None, title='Ticker', x_label=None, y_label=None, save_name=None, outside_legend=False):
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue)
    ax.set_title(title)
    plt.xticks(rotation=45)
    
    if outside_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if x_label:
        ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if save_name:
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plt.savefig(f"plots/{save_name}.svg", bbox_inches="tight")
    plt.show()

    
def plot_scatter(df, x, y, hue=None, title='Ticker', x_label=None, y_label=None, save_name=None):
    ax = sns.scatterplot(data=df, x=x, y=y, hue=hue, palette='muted')
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if x_label:
        ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if save_name:
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plt.savefig(f"plots/{save_name}.svg", bbox_inches="tight")
    plt.show()
