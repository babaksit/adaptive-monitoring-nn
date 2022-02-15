import os

import pandas as pd
import matplotlib.pyplot as plt


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def plot_each_column(df: pd.DataFrame, plot_constant_cols: bool = False) -> None:
    """
    Plot a dataframe by creating subplot for each column

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    plot_constant_cols : bool
        plot columns which their values are constant


    Returns
    -------
    None
    """
    plot_df = df.copy()
    if not plot_constant_cols:
        plot_df = plot_df.loc[:, (plot_df != plot_df.iloc[0]).any()].copy()

    values = plot_df.values

    cols = plot_df.columns

    n_subplot = 5
    icols = list(range(len(cols)))
    icols = list(chunks(icols, n_subplot))
    print(icols)
    for cols in icols:
        plt.figure()
        for i, col in enumerate(cols):
            plt.subplot(n_subplot, 1, i + 1)
            plt.plot(values[:, col])
            plt.subplots_adjust(hspace=2)
            plt.title(plot_df.columns[col], y=1, loc='right')
        plt.savefig(os.path.join("saved_plots", (str(col) + ".png")))
        plt.show()
        plt.close()


if __name__ == '__main__':
    df = pd.read_csv("/home/bsi/thesis/Adaptive_Monitoring_NN/data/rabbitmq_prometheus_24_jan.csv",
                     index_col=0, header=0)
    plot_each_column(df)
    # icols = list(range(13))
    # icols = list(chunks(icols, 5))
    # print(icols)
