import pathlib
from pathlib import Path
import re
import pandas as pd 
import requests
import numpy as np


def dl_usgs_water(wellname, save_path=Path.cwd()):
    '''
    Downloads water data from the USGS website for a given well and saves it to a specified path.

    Parameters:
    - wellname (str): The name or ID of the well. It should be in the USGS well name format, e.g., '375006112554801'.
    - save_path (str): The path where the downloaded data will be saved.

    Returns:
    - report (pd.DataFrame): The downloaded water data in a pandas DataFrame.
    - metadata (pd.Series): Metadata about the well, including well name, latitude, longitude, datum, well depth, and altitude.

    Raises:
    - Exception: If the wellname is not in the correct format.
    - Exception: If the download fails due to internet connection issues or the specified URL is not reachable.

    '''

    if wellname.isdigit():
        well_no = wellname
    else:
        try:
            well_no = re.findall(r"\d+",wellname)[0]
        except:
            raise("{} is not a usgs well name format, e.g.:'375006112554801'".format(wellname))
    url_1 = r'https://nwis.waterdata.usgs.gov/nwis/gwlevels?site_no=' + \
             well_no + r'&agency_cd=USGS&format=rdb'
    try:
        report = requests.get(url_1, stream=True)
    except:
        raise("Download failed! Check the Internet connection or {} is not reachable anymore".format(url_1))
    fail_pattern1 = r'Incorrectly formatted USGS site number'
    fail_pattern2 = r'No sites/data found using the selection criteria specified'
    with open(Path(save_path, well_no), 'wb') as f:
        for ch in report:
            if re.search(fail_pattern1, ch.decode('utf-8')) or re.search(fail_pattern2, ch.decode('utf-8')):
                raise Exception('Not able to find input USGS well number "{}"!'.format(well_no))
            f.write(ch)

    url_2 = r'https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=' + well_no
    pattern = re.compile(
        r'<title>USGS (.*\d)  .*?</title>.*?Latitude  (.*?), &nbsp; Longitude (.*?) &nbsp; (.*?)<br />.*?Well depth: (.*?) .*?Land surface altitude:  (.*?)'
    'feet above')
    pattern_coor = r'[&#;\\\'" ]'
    try:
        source = requests.get(url_2)
    except:
        raise("Download failed! Check the Internet connection or {} is not reachable anymore".format(url_1))
    match = re.findall(pattern, str(source.content))
    sitename = match[0][0]
    prelat = re.split(pattern_coor, match[0][1])
    lat = float(prelat[0]) + float(prelat[3]) / 60 + float(prelat[5]) / (60 * 60)
    prelong = re.split(pattern_coor, match[0][2])
    long = -(float(prelong[0]) + float(prelong[3]) / 60 + float(prelong[5]) / (60 * 60))
    datum = match[0][3]
    try:
        well_depth = float(match[0][4])
    except:
        well_depth = np.nan
    try:
        altitude = float(re.split(pattern_coor, match[0][5])[0].replace(',',''))
    except:
        altitude = np.nan
    metadata = pd.Series({'wellname': sitename, 'lon': long, 'lat': lat,  'datum': datum,
                          'well_depth':well_depth, 'altitude':altitude})
    report = pd.read_fwf(Path(save_path, well_no))
    return report, metadata

def format_usgs_water(usgs_well_NO : str, dlpath: str | pathlib.PurePath =Path.cwd() ) -> pd.DataFrame:
    '''
    Formats the downloaded water data from the USGS website for a given well.
    File will contain attributes of the well and the water level data.
    Check attributes use formatdata.attrs
    '''
    report = pd.DataFrame()
    if isinstance(usgs_well_NO, str):
        try:
            exist_file = Path(dlpath, usgs_well_NO)
            report = pd.read_fwf(exist_file)
            _, meta = dl_usgs_water(usgs_well_NO, dlpath)
        except:
            report, meta = dl_usgs_water(usgs_well_NO, dlpath)
    elif isinstance(usgs_well_NO, pd.DataFrame):
        report = usgs_well_NO
    #because all reports are different in column so we need to find the exact column the header start
    row_start = report[report.apply(lambda row: row.astype(str).str.contains('agency_cd\tsite_no').any(), axis=1)].index.values[0]
    #https://datascientyst.com/search-for-string-whole-dataframe-pandas/
    messyinfo = report.iloc[:row_start, :]#other inforotion above header
    data = report.iloc[row_start:, :].copy()#real data
    data.rename(columns=data.iloc[0], inplace=True) #set data header
    data = data.iloc[2:, :]# reset data region
    columns = data.columns[0].split('\t') + data.columns[1:].to_list() #split columns they looks like aaa\tbbb\tccc
    split_data = list(data.iloc[:, 0].str.split('\t').values)#data also the same format
    formatdata = pd.DataFrame(split_data, columns=columns)#recombine the data into dataframe
    formatdata['lev_dt'] = pd.to_datetime(formatdata['lev_dt'])
    water_group = formatdata.groupby(['lev_dt'])
    concat_list = []
    column_data_types = {
        'agent':str,
        'well_no':str,
        'time':'datetime64[ns]',
        'wt_blw_gd':float,
        'wt_abv_ngvd29':float,
        'wt_abv_navd88':float,            
    }
    for time, group in water_group: 
        temp_df = pd.DataFrame(columns=column_data_types.keys(),index=[1]).astype(column_data_types)
        temp_df['agent'] = group['agency_cd'].iloc[0]
        temp_df['well_no'] = group['site_no'].iloc[0]
        temp_df['time'] = time
        temp_df['wt_blw_gd'] = group[group['lev_va'] != '']['lev_va'].astype(float).iloc[0] / 3.28084
        temp_df['wt_abv_ngvd29'] = group[(group['sl_lev_va'] != '') & 
                                            (group['sl_datum_cd'] == 'NGVD29')]['sl_lev_va'].astype(float).iloc[0] /3.28084
        temp_df['wt_abv_navd88'] = group[(group['sl_lev_va'] != '') &
                                            (group['sl_datum_cd'] == 'NAVD88')]['sl_lev_va'].astype(float).iloc[0] /3.28084
        concat_list.append(temp_df)
    formatdata = pd.concat(concat_list)
    formatdata.reset_index(inplace=True, drop=True)
    formatdata.attrs['wellname'] = meta['wellname']
    formatdata.attrs['lon'] = meta['lon']
    formatdata.attrs['lat'] = meta['lat']
    formatdata.attrs['datum'] = meta['datum']
    formatdata.attrs['well_depth'] = float(meta['well_depth']) /3.28084
    formatdata.attrs['altitude'] = float(meta['altitude']) /3.28084
    formatdata.attrs['unit'] = 'meter'
    print('{} downloaded to {}'.format(formatdata.attrs['wellname'], dlpath))
    return formatdata
