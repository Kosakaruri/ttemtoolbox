import pandas as pd
import pathlib
import re
def keyword_search(fname, pattern):
    """
    Try to find keyword that matching the pre-defined lithology keywords
    :param fname:
    :param pattern:
    :return:
    """
    if isinstance(fname, dict):
        keys = [str(key).strip() for key in list(fname.keys())]
        match = [key for key in keys if key.lower in pattern]
        return match
    elif isinstance(fname, pd.DataFrame):
        columns = [str(key).strip() for key in list(fname.columns)]
        match = [column for column in columns if column.lower in pattern]
        return match
    else:
        raise TypeError('fname must be dict or DataFrame')

def skip_metadata(fname: (pathlib.PurePath, str),
                  keyword_pattern: str) -> list:
    """
    Use given keyword pattern to skip any metadata above the file in tTEM xyz file
    :return:
    """
    with open(str(fname), 'r') as file:
        lines = file.readlines()
    regex = re.compile(keyword_pattern)
    match_index = []
    for index, line in enumerate(lines):
        if regex.search(line):
            match_index.append(index)
    if len(match_index) == 0:
        raise ValueError('No keywords pattern matched "{}" in file {}'.format(keyword_pattern, str(fname)))
    elif len(match_index) > 1:
        raise ValueError('Found multiple keywords pattern matched "{}" in file {}'. format(keyword_pattern, str(fname)))
    data = [line[1::].strip().split() for line in lines[match_index[0]::]]
    return data