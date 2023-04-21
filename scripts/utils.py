import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import itertools

# Default matplotlib colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

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

def load_dataset():
    monthly_data = pd.read_csv("../dataset/housing_in_london_monthly_variables.csv")
    monthly_data["date"] = pd.to_datetime(monthly_data["date"])
    yearly_data = pd.read_csv("../dataset/housing_in_london_yearly_variables.csv")
    yearly_data["date"] = pd.to_datetime(yearly_data["date"])
    return monthly_data, yearly_data

def get_area(data, area):
    return data[data["area"] == area]