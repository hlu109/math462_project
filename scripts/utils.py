import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import itertools
import numpy as np
from datetime import datetime
from itertools import cycle

import statsmodels.tsa.api as smt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Default matplotlib colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = cycle(prop_cycle.by_key()['color'])

def find_files(*sources, name=None, extension=None):
    '''
    Recursively retrieve files from the given source directories whose names and/or extensions (full)match the given patterns.
    name: string or regex pattern
    extension: string or regex pattern
    Returns a DirEntry generator
    '''
    # Compile regexes if needed
    if name is None:
        name = re.compile(r'.*')
    elif type(name) is not re.Pattern:
        name = re.compile(name)
    if extension is None:
        extension = re.compile(r'.*')
    elif type(extension) is not re.Pattern:
        extension = re.compile(extension)

    # Keep track of the sources already scanned and the files already found
    memo_table = {}

    def find_files_helper(*sources):
        # Search through each source directoty
        for source in sources:
            # Get all of the contents of the source directory and search them
            entries = os.scandir(source)
            for entry in entries:
                # Check if the entry has already been scanned or matched
                normed = os.path.normpath(entry.path)
                if normed not in memo_table:
                    memo_table[normed] = True
                    # If the current entry is itself a directory, search it recursively
                    if entry.is_dir():
                        yield from find_files_helper(entry)

                    # Otherwise yield entries whose name matches the name pattern and whose extension matches the extension pattern
                    else:
                        # Return only entries that have not already been found
                        filename, fileext = os.path.splitext(entry.name)
                        if name.fullmatch(filename) is not None and \
                        extension.fullmatch(fileext) is not None:
                            yield entry
            entries.close()

    return find_files_helper(*sources)


def imerge(*its, enum=False):
    '''Merge iterators end to end'''
    for i, it in enumerate(its):
        for el in it:
            if enum:
                yield i, el
            else:
                yield el


def pairwise(iterable):
    '''s -> (s0,s1), (s1,s2), (s2, s3), ...'''
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def mkdir(path):
    '''Create a directory if it does not exist'''
    path, ext = os.path.splitext(path)
    if ext != '':
        mkdir(os.path.split(path)[0])
    elif not os.path.isdir(path):
        parent, _ = os.path.split(path)
        mkdir(parent)
        os.mkdir(path)


def savefig(fig, savepath, *args, **kwargs):
    '''Save a given figure to the given path, creating a directory as needed and making sure not to overwrite files'''
    savedir, _ = os.path.split(savepath)
    mkdir(savedir)
    fig.savefig(savepath, *args, **kwargs)


def add_version(savepath, replace=False):
    '''Appends a version number to the end of a path if that path already exists with the given name and if replace is False'''
    savefile, saveext = os.path.splitext(savepath)
    if replace or not os.path.exists(savepath):
        return savepath
    else:
        version = 1
        new_savepath = '{} ({}){}'.format(savefile, version, saveext)
        while os.path.exists(new_savepath):
            version += 1
            new_savepath = '{} ({}){}'.format(savefile, version, saveext)
        return new_savepath


def clean_load_dataset(file_path):
    """ Loads a csv file into a dataframe. 
    
        Sort by area and then date. Adds columns for year and month. 
    """
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["area", "date"], inplace=True)
    df.reset_index()

    # add column for year and month
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month

    return df


def load_dataset():
    '''
    Load in both the monthly and yearly csv files.
    Sort by area and then date, and add the time derivative of average price under the column average_price_d1.

    Adds columns for year and month. Also, add a column that records seconds since the earliest data sample (?).
    '''
    monthly_data = clean_load_dataset(
        "../dataset/housing_in_london_monthly_variables.csv")
    add_column_derivative(monthly_data, "average_price", "average_price_d1")

    yearly_data = clean_load_dataset(
        "../dataset/housing_in_london_yearly_variables.csv")

    start_time = monthly_data["date"][0]

    def make_column(area_data):
        return [dt.total_seconds() for dt in area_data["date"] - start_time]

    add_column_by_area(monthly_data, "seconds", make_column)
    add_column_by_area(yearly_data, "seconds", make_column)

    return monthly_data, yearly_data


def load_interpolated_data(interpolate_out_of_range=True):
    """ Returns a single dataframe containing monthly data, along with yearly 
        data that has been interpolated at a monthly frequency. 
    """
    monthly_data, yearly_data = load_dataset()
    yearly_resampled = clean_load_dataset(
        "../dataset/housing_in_london_yearly_variables_resampled.csv")
    for col in [
            'median_salary', 'life_satisfaction', 'population_size',
            'number_of_jobs', 'area_size', 'no_of_houses'
    ]:
        monthly_data = interpolate_yearly(monthly_data, yearly_resampled, col,
                                          interpolate_out_of_range)
    return monthly_data


def add_column_by_area(data, new_column, make_column):
    data[new_column] = np.full(data.shape[0], np.nan)
    for area in data["area"].unique():
        mask = data["area"] == area
        area_data = data[mask]
        idxs = data.index[mask]
        data.loc[idxs, new_column] = make_column(area_data)
    return data


def add_column_derivative(data, column, new_column):

    def make_column(area_data):
        # dcol = np.diff(area_data[column])
        # dt = [delta / np.timedelta64(1, "s") for delta in np.diff(area_data["date"])]
        # return np.concatenate([[0], dcol / dt])
        return np.concatenate([[0], np.diff(area_data[column])])

    return add_column_by_area(data, new_column, make_column)


def get_area(data, area):
    return data[data["area"] == area]


def filter_time(df,
                start_time=datetime(1995, 1, 1, 0, 0, 0),
                end_time=datetime(2020, 1, 1, 0, 0, 0)):
    '''Filter out rows whose time does not fall between the given start and end times.'''
    dates = pd.to_datetime(df["date"])
    if start_time == None and end_time == None:
        return df
    elif start_time == None:
        return df[(dates <= end_time)]
    elif end_time == None:
        return df[(start_time <= dates)]
    else:
        return df[(start_time <= dates) & (dates <= end_time)]


def project_time_series(ref, ts, key=None, reversed=False):
    '''Function to project a time series onto a reference time series'''
    # Create key function if it does not exist yet
    if key is None:
        key = lambda x: x

    if reversed:
        # Set up some book keeping
        it = iter(ref)
        r = next(it)
        t_last = None

        for t, t_next in pairwise(ts):
            t_last = t_next
            try:
                while key(r) < key(t):
                    yield (r, None)
                    r = next(it)

                while key(t) <= key(r) and key(r) < key(t_next):
                    yield (r, t)
                    r = next(it)
            except StopIteration:
                return

        try:
            while True:
                yield (r, t_last)
                r = next(it)
        except StopIteration:
            return
    else:
        # Set up some book keeping
        it = iter(ts)
        counter = next(it)
        r_last = None

        def get_next(r, start):
            # Try getting the next value of the iterator thats larger than r
            counter = start
            try:
                while counter is not None and key(counter) < key(r):
                    counter = next(it)
            except StopIteration:
                counter = None
            return counter

        # Iterate through the reference timeseries
        for r, r_next in pairwise(ref):
            r_last = r_next

            counter = get_next(r, counter)
            if counter is not None and key(counter) < key(r_next):
                yield (r, counter)
            else:
                yield (r, None)

        counter = get_next(r_last, counter)
        yield (r_last, counter)


def interpolate_yearly(monthly_data,
                       yearly_resampled,
                       col,
                       interpolate_out_of_range=True):
    assert col in yearly_resampled.columns

    for a in yearly_resampled.area.unique():
        if a not in monthly_data.area.unique():
            continue

        idx_start = yearly_resampled.loc[yearly_resampled.area == a,
                                         col].first_valid_index()
        idx_end = yearly_resampled.loc[yearly_resampled.area == a,
                                       col].last_valid_index()

        if idx_start is None:
            # the area does not contain any data for the given col, so skip
            continue
        min_year = yearly_resampled.loc[idx_start].year
        min_month = yearly_resampled.loc[idx_start].month

        max_year = yearly_resampled.loc[idx_end].year
        max_month = yearly_resampled.loc[idx_end].month

        # copy interpolated data between min and max year
        monthly_data.loc[(monthly_data.area == a) & (
            (monthly_data.year > min_year) |  # after min year
            (((monthly_data.year == min_year)) &  # or, min year + 
             (monthly_data.month >= min_month))  # after min month
        ) & ((monthly_data.year < max_year) |  # before max year
             (((monthly_data.year == max_year)) &  # or, max yr +
              (monthly_data.month <= max_month))  # before max month
             ), col] = yearly_resampled.loc[(yearly_resampled.area == a) & (
                 (yearly_resampled.year > min_year) |  # after min year
                 (((yearly_resampled.year == min_year)) &  # or, min year + 
                  (yearly_resampled.month >= min_month))  # after min month
             ) & ((yearly_resampled.year < max_year) |  # before max year
                  (((yearly_resampled.year == max_year)) &  # or, max yr +
                   (yearly_resampled.month <= max_month))  # before max month
                  ), col].values
        # this might copy some empty values but thats ok since we will extrapolate those next

        if interpolate_out_of_range:
            # copy constant-filled data before min year & month
            monthly_data.loc[(monthly_data.area == a) & (
                (monthly_data.year < min_year) |  # before min year
                (((monthly_data.year == min_year)) &  # or, min year + 
                 (monthly_data.month <= min_month))  # before min month
            ), col] = yearly_resampled.loc[
                (yearly_resampled.area == a) &
                (yearly_resampled.year == min_year)
                &  # min year
                (yearly_resampled.month == min_month),  # min month
                col].values[0]

            # copy constant-filled data after max year
            monthly_data.loc[
                (monthly_data.area == a)
                & ((monthly_data.year > max_year) |  # after max year
                   (((monthly_data.year == max_year)) &  # or, max yr +
                    (monthly_data.month >= max_month))  # after max month
                   ), col] = yearly_resampled.loc[
                       (yearly_resampled.area == a)
                       & (yearly_resampled.year == max_year) &  # max yr
                       (yearly_resampled.month == max_month),  # max month
                       col].values[0]
        else:
            monthly_data.loc[(monthly_data.area == a) & (
                (monthly_data.year < min_year) |  # before min year
                (((monthly_data.year == min_year)) &  # or, min year + 
                 (monthly_data.month <= min_month))  # before min month
            ), col] = np.nan

            monthly_data.loc[
                (monthly_data.area == a)
                & ((monthly_data.year > max_year) |  # after max year
                   (((monthly_data.year == max_year)) &  # or, max yr +
                    (monthly_data.month >= max_month))  # after max month
                   ), col] = np.nan

    return monthly_data


def plot_multi_acf(data, lags, titles, suptitle='', ylim=None, partial=False):
    """ Plot autocorrelation at a variety of frequencies. 
    
        Copied from https://www.ethanrosenthal.com/2018/03/22/time-series-for-scikit-learn-people-part2/
        
        Args: 
            lags: list of frequencies 
    """
    num_plots = len(lags)
    fig, ax = plt.subplots(len(lags), 1, figsize=(8, 2 * num_plots))
    fig.suptitle(suptitle)
    if num_plots == 1:
        ax = [ax]
    acf_func = smt.graphics.plot_pacf if partial else smt.graphics.plot_acf
    for idx, (lag, title) in enumerate(zip(lags, titles)):
        ax[idx] = acf_func(data, lags=lag, ax=ax[idx], title=title)
        if ylim is not None:
            ax[idx].set_ylim(ylim)

    fig.tight_layout()


def create_windowed_dataset(dataset, cols, look_back=1, look_forward=1):
    """ creates new time series data with past datapoints as input features. 
    
    Copied from https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    
        Args:
            dataset: pd dataframe
            look_back (int): num of previous time steps to use as input variables to predict the next time period
    """
    dataset = dataset.reset_index(drop=True)
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_forward):
        # recall loc is end-inclusive
        x = dataset.loc[i:(i + look_back) - 1, cols].to_numpy()
        y = dataset.loc[(i + look_back):(i + look_back + look_forward) - 1,
                        cols].to_numpy()
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX).astype(float), np.array(dataY).astype(float)


def format_data_for_forecasting(area_data, cols, lookback, lookforward):
    """ modified from """
    train_df, test_df = train_test_split(area_data[cols],
                                         test_size=0.3,
                                         shuffle=False)

    # normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df[cols] = scaler.fit_transform(train_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])

    X_train, y_train = create_windowed_dataset(train_df, cols, lookback,
                                                     lookforward)
    X_test, y_test = create_windowed_dataset(test_df, cols, lookback,
                                                   lookforward)
    return train_df, test_df, X_train, y_train, X_test, y_test, scaler
