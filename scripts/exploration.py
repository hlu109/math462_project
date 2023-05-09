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

def scatter_area(df, area, var_name1, var_name2):
    area_data = utils.get_area(df, area)
    plt.scatter(area_data[var_name1], area_data[var_name2])
    corr = np.corrcoef(area_data[var_name1], area_data[var_name2])[0][1]
    print(f"Correlation: {corr}")
    plt.title(area)
    plt.xlabel(var_name1)
    plt.ylabel(var_name2)
    plt.show()

if __name__ == "__main__":
    monthly_data = utils.load_interpolated_data()
    utils.add_column_derivative(monthly_data, "median_salary", "median_salary_d1")
    plot_var_area(monthly_data, "city of london", "average_price", ylabel="Average House Price (\xA3)")
    plot_var_area(monthly_data, "city of london", "median_salary", ylabel="Median Salary (\xA3)")
    # plot_var(monthly_data, "average_price_d1")
    # plot_var_area(monthly_data, "city of london", "median_salary")
    # scatter_area(monthly_data, "city of london", "life_satisfaction", "average_price")
    # scatter_area(monthly_data, "city of london", "life_satisfaction", "average_price_d1")
    # scatter_area(monthly_data, "city of london", "median_salary_d1", "average_price")
    # scatter_area(monthly_data, "city of london", "median_salary_d1", "average_price_d1")