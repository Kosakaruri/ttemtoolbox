#!/usr/bin/env python
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
        match = [key for key in keys if key.lower() in pattern]
        return match
    elif isinstance(fname, pd.DataFrame):
        columns = [str(key).strip() for key in list(fname.columns)]
        match = [column for column in columns if column.lower() in pattern]
        return match
    else:
        raise TypeError('fname must be dict or DataFrame')

def skip_metadata(fname: pathlib.PurePath | str,
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

def type_convert(config_str: str) :
    config_str = config_str.strip()
    if config_str.isdigit(): 
        return int(config_str)
    if config_str.replace('.', '', 1).isdigit():
        return float(config_str)
    if config_str == 'T' or config_str == 'True':
        return True
    if config_str == 'F' or config_str == 'False':
        return False
    if config_str == 'NA' or config_str == 'NAN':
        return float('nan')
    if config_str.startswith('"') and config_str.endswith('"'):
        return config_str.replace('"', '')
    if config_str.replace('e', '', 1).isdigit():
        return int(float(config_str))
    if config_str.replace('e-', '', 1).isdigit():
        return float(config_str)
    if config_str.startswith('[') and config_str.endswith(']'):
        config_str = config_str[1:-1].split(',')
        config_str = [item.strip() for item in config_str]
        return config_str
    return config_str

def parse_config(config_path: str | pathlib.PurePath) -> dict:
    """
    This function takes a pathlik object point to CONFIG file and parse the config file into a dictionary
    :param config_path: pathlike object, the path to the CONFIG file
    :return: dictoionary, the parsed config file in a dictionary format
    """
    config = {}
    with open(config_path, 'r') as file:
        lines = [line.strip() for line in file if not line.startswith(("#", " "))]
    param_list = [param for param in lines if param != '']
    for p in param_list:
        key, value = p.split("=", 1)
        key = key.strip()
        value = type_convert(value)
        config[key] = value
    return config
    
    