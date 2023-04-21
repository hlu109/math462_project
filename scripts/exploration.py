from utils import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    monthly_data, yearly_data = load_dataset()
    boroughs = monthly_data["area"].unique()
    fig, axs = plt.subplots(5, 9, sharex=True, sharey=True)
    axs = axs.reshape(-1)
    start_date = sorted(monthly_data["date"])[0]
    for borough, ax in zip(boroughs, axs):
        borough_data = get_area(monthly_data, borough)
        ax.plot(borough_data["date"] - start_date, borough_data["average_price"])
        ax.set_title(borough)
    fig.tight_layout()
    plt.show()
    fig.savefig("../plots/avg_price_vs_time_by_borough.png", bbox_inches="tight")