import utils
import numpy as np
import matplotlib.pyplot as plt

def plot_var(df, var_name):
    areas = df["area"].unique()
    fig, axs = plt.subplots(5, 9)
    axs = axs.reshape(-1)
    start_date = sorted(df["date"])[0]
    for area, ax in zip(areas, axs):
        area_data = utils.get_area(df, area)
        ax.plot(area_data["date"] - start_date, area_data[var_name])
        ax.plot(area_data["date"] - start_date, np.full(len(area_data), np.mean(area_data[var_name])))
        ax.set_title(area)
    fig.tight_layout()
    plt.show()
    # fig.savefig("../plots/{}_vs_time_by_area.png".format(var_name),
    #             bbox_inches="tight")

def plot_var_area(df, area, var_name, **kwargs):
    # start_date = sorted(df["date"])[0]
    area_data = utils.get_area(df, area)
    plt.plot(area_data["date"], area_data[var_name])
    plt.xlabel(kwargs.get("xlabel", "Time"))
    plt.ylabel(kwargs.get("ylabel", var_name))
    # plt.plot(area_data["date"] - start_date, np.full(len(area_data), np.mean(area_data[var_name])))
    plt.title(kwargs.get("title", area))
    plt.show()

def scatter_area(df, area, var_name1, var_name2, **kwargs):
    area_data = utils.get_area(df, area)
    plt.scatter(area_data[var_name1], area_data[var_name2])
    corr = np.corrcoef(area_data[var_name1], area_data[var_name2])[0][1]
    print(f"Correlation: {corr}")
    plt.text(0, 0, f"Correlation: {corr:.4f}", transform=plt.gca().transAxes)
    plt.title(area)
    plt.xlabel(kwargs.get("xlabel", var_name1))
    plt.ylabel(kwargs.get("ylabel", var_name2))
    plt.show()

if __name__ == "__main__":
    monthly_data = utils.load_interpolated_data()
    utils.add_column_derivative(monthly_data, "median_salary", "median_salary_d1")
    # plot_var(monthly_data, "average_price_d1")
    # plot_var_area(monthly_data, "city of london", "average_price", ylabel="Average House Price (\xA3)")
    # plot_var_area(monthly_data, "city of london", "average_price_d1", ylabel="Change in Average House Price (\xA3/month)")
    # plot_var_area(monthly_data, "city of london", "median_salary", ylabel="Median Salary (\xA3)")
    # plot_var_area(monthly_data, "city of london", "population_size", ylabel="Population Size")
    # plot_var_area(monthly_data, "city of london", "number_of_jobs", ylabel="Number of Jobs")
    # plot_var_area(monthly_data, "city of london", "no_of_houses", ylabel="Number of Houses")
    # plot_var_area(monthly_data, "city of london", "no_of_crimes", ylabel="Number of Crimes")
    # scatter_area(monthly_data, "city of london", "median_salary", "average_price", xlabel="Median Salary (\xA3)", ylabel="Average House Price (\xA3)")
    # scatter_area(monthly_data, "city of london", "population_size", "average_price", xlabel="Population Size", ylabel="Average House Price (\xA3)")
    # scatter_area(monthly_data, "city of london", "number_of_jobs", "average_price", xlabel="Number of Jobs", ylabel="Average House Price (\xA3)")
    # scatter_area(monthly_data, "city of london", "no_of_houses", "average_price", xlabel="Number of Houses", ylabel="Average House Price (\xA3)")
    # scatter_area(monthly_data, "city of london", "median_salary_d1", "average_price_d1")
    # scatter_area(monthly_data, "city of london", "life_satisfaction", "average_price")
    # scatter_area(monthly_data, "city of london", "life_satisfaction", "average_price_d1")

    # Plot correlations
    # vars = ["average_price", "median_salary", "population_size", "number_of_jobs", "no_of_houses"]
    # area_data = utils.get_area(monthly_data, "city of london")
    # fig, axs = plt.subplots(len(vars), len(vars), figsize=(1.5 * len(vars), 1.5 * len(vars)))
    # for i, (row_var, ax_row) in enumerate(zip(vars, axs)):
    #     for j, (col_var, ax) in enumerate(zip(vars, ax_row)):
    #         if i == j:
    #             ax.hist(area_data[col_var], color="lightgray", edgecolor="black")
    #             ax.text(0.5, 0.9, col_var, horizontalalignment="center", verticalalignment="top", transform=ax.transAxes)
    #         elif i > j:
    #             ax.scatter(area_data[col_var], area_data[row_var], color="black", marker=".")
    #             if i == len(vars) - 1:
    #                 ax.set_xlabel(col_var)
    #             if j == 0:
    #                 ax.set_ylabel(row_var)
    #         else:
    #             corr = np.corrcoef(area_data[col_var], area_data[row_var])[0][1]
    #             ax.text(0.5, 0.5, f"{corr:.2f}", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
    #         ax.set_xticklabels([])
    #         ax.set_yticklabels([])
    # axs[0, 0].annotate("City of London Correlations", (0.5, 0.9), xycoords="figure fraction", ha="center", fontsize=12)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    # utils.savefig(fig, "../plots/city_of_london/correlations.png")

    # Plot data shifted one month forward
    # area_data = utils.get_area(monthly_data, "city of london")
    # plt.plot(area_data["date"], area_data["average_price"], label="true prices")
    # plt.plot(area_data["date"][1:], area_data["average_price"][:-1], label="shifted 1 month forward")
    # plt.title("city of london")
    # plt.xlabel("Time")
    # plt.ylabel("Average House Price (\xA3)")
    # plt.legend()
    # plt.show()