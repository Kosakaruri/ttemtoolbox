#!/usr/bin/env python
import pathlib
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import pearsonr
from ttemtoolbox.core.process_well import ProcessWell

def select_closest(ttemdata: pd.DataFrame | gpd.GeoDataFrame,
                   welllog: pd.DataFrame | gpd.GeoDataFrame,
                   search_radius=500,
                   showskip=False,
                   ):
    def get_distance(group1, group2):
        dis = np.sqrt((group1[0] - group2[0]) ** 2 + (group1[1] - group2[1]) ** 2)
        return dis
    concatlist = []
    concatwell = []
    skipname = []
    skipdistace = []
    ori_well = welllog
    groups_well = ori_well.groupby('Bore')
    ttem_location = list(ttemdata.groupby(['X', 'Y']).groups.keys())
    for name, group in groups_well:
        wellxy = list(group[['X','Y']].iloc[0])
        well_ttem_distance = list(map(lambda x: get_distance(wellxy, x), ttem_location))
        minvalue = min(well_ttem_distance)
        if minvalue <= float(search_radius):
            point_match = ttem_location[well_ttem_distance.index(minvalue)]
            matchpoint = ttemdata[(ttemdata['X'] == point_match[0]) & (ttemdata['Y'] == point_match[1])].copy()
            matchpoint.loc[:, 'distance'] = minvalue
            matchpoint.loc[:, 'Bore'] = name
            concatlist.append(matchpoint)
            concatwell.append(group)
        else:
            skipname.append(name)
            skipdistace.append(minvalue)
    try:
        matched_ttem = pd.concat(concatlist).reset_index(drop=True)
    except ValueError:
        matched_ttem = pd.DataFrame()
    try:
        matched_well = pd.concat(concatwell).reset_index(drop=True)
    except ValueError:
        matched_well = pd.DataFrame()
    skipped = pd.DataFrame({'Bore': skipname, 'Distance': skipdistace})
    print('Total of {} well with in radius ({}m), {} skipped'.format(len(concatlist), search_radius, len(skipname)))
    if showskip is False:
        return matched_ttem, matched_well
    else:
        return matched_ttem, matched_well, skipped

def sum_thickness(welllog): # receive single bore well_log and sum up the thickness of fine/mix/coarse material
    output = pd.DataFrame(columns=["Lithology", "Thickness"])
    init_lith = welllog["Keyword"].iloc[0]
    init_elev = welllog["Elevation_top"].iloc[0]
    concatlist=[]
    # TODO make pure ndarry to fit JIT to increase speed
    for index, row in welllog.iterrows():
        if row["Keyword"] == init_lith:
            pass
        elif row["Keyword"] != init_lith:
            pack = [init_lith, init_elev - row["Elevation_top"]]
            tmp = pd.DataFrame(pack,index=('Lithology','Thickness')).T
            concatlist.append(tmp)
            init_lith = row["Keyword"]
            init_elev = row["Elevation_top"]
        else:
            print("error")
        if row["Elevation_top"] == welllog.loc[welllog.index[-1], 'Elevation_top']:
            pack = [init_lith, init_elev - row["Elevation_top"]]
            tmp = pd.DataFrame(pack,index=('Lithology','Thickness')).T
            concatlist.append(tmp)
    output = pd.concat(concatlist)
    output = output.groupby("Lithology")["Thickness"].sum()
    return output

def ttem_well_connect(matched_ttem, matched_well):
    # use ttem data interval to filter out welllog data to make a bootstrap ready dataframe
    matched_ttem[["Fine", "Mix", "Coarse"]] = 0
    #TODO make pure numpy format to increase speed
    concatlist= []
    ttem_group = matched_ttem.groupby('Bore')
    for name, group in ttem_group:
        well_select = matched_well[matched_well['Bore'] == name].copy()
        for index, row in group.iterrows():
            top = row["Elevation_Cell"]
            bott = row["Elevation_End"]
            match_litho = well_select[(well_select['Elevation_top'] >= bott) & (well_select['Elevation_bottom'] < top)]
            if match_litho.empty:
                break
            thickness = dict(match_litho.groupby('Keyword')['Keyword'].count()*match_litho['Thickness'].iloc[0])

            if "fine grain" in thickness:
                row["Fine"] = row["Coarse"]+thickness["fine grain"]
            else:
                row["Fine"] = 0
            if "mix grain" in thickness:
                row["Mix"] = row["Mix"]+thickness["mix grain"]
            else:
                row["Mix"] = 0
            if "coarse grain" in thickness:
                row["Coarse"] = row["Coarse"]+thickness["coarse grain"]
            else:
                row["Coarse"] = 0
            row = row.to_frame().T
            concatlist.append(row)
    df = pd.concat(concatlist).reset_index(drop=True)
    return df

def pre_bootstrap(dataframe,welllog, distance=500):
    matched_ttem, matched_well = select_closest(dataframe, welllog, search_radius=distance, showskip=False)
    stitched_ttem_well = ttem_well_connect(matched_ttem, matched_well)
    Resistivity = stitched_ttem_well["Resistivity"].to_numpy().astype('float64')
    Thickness_ratio = stitched_ttem_well[["Fine", "Mix", "Coarse"]].div(stitched_ttem_well["Thickness"],
                                                                   axis=0).to_numpy().astype('float64')
    return stitched_ttem_well, Resistivity, Thickness_ratio, matched_ttem, matched_well


def bootstrap(resistivity, thickness_ratio):
    """
    bootstrap method, randomly pick from pre_bootstrap dataset to create a new data set with same shape,
    use thenew data set as an over-determined problem to solve the equation. Repeat 1000 times and output the resistivity
    The linear algebra equation check https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/gwat.12656
    """
    print('Bootstraping...')
    fine_Resistivity = np.empty(1000)
    mix_Resistivity = np.empty(1000)
    coarse_Resistivity = np.empty(1000)
    for k in range(1000):
        random_index = np.random.choice(np.arange(len(resistivity)), len(resistivity), replace=True)
        resistivity_sample = resistivity[random_index]
        resistivity_reverse = 1 / resistivity_sample
        thickness_ratio_sample = thickness_ratio[random_index]
        lstsq_result = np.linalg.lstsq(thickness_ratio_sample, resistivity_reverse)
        if lstsq_result[0][0] == 0:
            fine_Resistivity[k] = 0
        else:
            fine_Resistivity[k] = 1/lstsq_result[0][0]
        if lstsq_result[0][1] == 0:
            mix_Resistivity[k] = 0
        else:
            mix_Resistivity[k] = 1/lstsq_result[0][1]
        if lstsq_result[0][2] == 0:
            coarse_Resistivity[k] = 0
        else:
            coarse_Resistivity[k] = 1/lstsq_result[0][2]
    print('Done!')
    return fine_Resistivity, mix_Resistivity, coarse_Resistivity

def confidence(bootstrap_result, confidence=95): #95% condifence interval
    confidence_index = [(100 - confidence)/2, confidence+(100 - confidence)/2]
    confidence_interval = [np.percentile(bootstrap_result,confidence_index[0]),
                           np.percentile(bootstrap_result, confidence_index[1])]
    return confidence_interval

def packup(Fine_Resistivity, Mix_Resistivity, Coarse_Resistivity):

    Resi_conf_df = pd.DataFrame({"Fine_conf": confidence(Fine_Resistivity),
                                 "Mix_conf": confidence(Mix_Resistivity),
                                 "Coarse_conf": confidence(Coarse_Resistivity)})
    return Resi_conf_df