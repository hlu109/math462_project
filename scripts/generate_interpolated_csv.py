# code to generate a new csv containing the yearly data, interpolated at a
# monthly frequency
# this only needs to be run once to create and save the csv.

from utils import *

if __name__ == "__main__":
    monthly_data, yearly_data = load_dataset()

    dfs = []
    for a in yearly_data.area.unique():
        if a not in monthly_data.area.unique():
            continue

        print(a)
        interpolated_df = get_area(yearly_data, a).copy()
        interpolated_df = interpolated_df[[
            'date', 'median_salary', 'life_satisfaction', 'mean_salary',
            'recycling_pct', 'population_size', 'number_of_jobs', 'area_size',
            'no_of_houses'
        ]].resample('M', on='date').mean(numeric_only=True)
        interpolated_df = interpolated_df.interpolate()
        interpolated_df['area'] = a
        cols = ['area'] + interpolated_df.columns.tolist()[:-1]
        interpolated_df = interpolated_df[cols]
        print(interpolated_df)
        dfs.append(interpolated_df)
    yearly_interpolated = pd.concat(dfs)
    yearly_interpolated.head(243)

    yearly_interpolated.to_csv(
        '../dataset/housing_in_london_yearly_variables_resampled.csv')
