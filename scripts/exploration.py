from utils import *
import matplotlib.pyplot as plt


def plot_var(df, var_name):
    areas = df["area"].unique()
    fig, axs = plt.subplots(5, 9, sharex=True, sharey=True)
    axs = axs.reshape(-1)
    start_date = sorted(df["date"])[0]
    for area, ax in zip(areas, axs):
        area_data = get_area(df, area)
        ax.plot(area_data["date"] - start_date, area_data[var_name])
        ax.set_title(area)
    fig.tight_layout()
    plt.show()
    fig.savefig("../plots/{}_vs_time_by_area.png".format(var_name),
                bbox_inches="tight")


if __name__ == "__main__":
    monthly_data, yearly_data = load_dataset()
    plot_var(monthly_data, "average_price")