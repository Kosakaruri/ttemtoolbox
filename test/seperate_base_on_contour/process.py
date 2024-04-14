import tTEM_toolbox as tt
import pathlib
from pathlib import Path
import pandas as pd
import re
import numpy as np
workdir = Path.cwd().joinpath(r'data')
welllog = workdir.joinpath('Well_log.xlsx')
tTEM_file_list = list(workdir.glob('*MOD.xyz'))
DOI_file_list = list(workdir.glob('DOI*.xyz'))
shape_path = workdir.joinpath(r'shapefiles')
shapefiles = list(shape_path.glob('*.shp'))
tTEM_lsl = tt.core.process_ttem.ProcessTTEM(tTEM_file_list[1])
tTEM_north = tt.core.process_ttem.ProcessTTEM(tTEM_file_list[0], DOI_file_list)
tTEM_center = tt.core.process_ttem.ProcessTTEM(tTEM_file_list[2], DOI_file_list)
tTEM_north_data = tTEM_north.data()
tTEM_center_data = tTEM_center.data()
tTEM_lsl_data = tTEM_lsl.data()
#load segmented data
segment_north_ttem_list = list(Path(r'/mnt/c/Users/jldz9/Documents/export').glob('ttem_n_*.csv'))
segment_north_east_ttem_list = list(Path(r'/mnt/c/Users/jldz9/Documents/export').glob('ttem_ne_*.csv'))
segment_center_ttem_list = list(Path(r'/mnt/c/Users/jldz9/Documents/export').glob('ttem_ct_*.csv'))
segment_lsl_ttem_list = list(Path(r'/mnt/c/Users/jldz9/Documents/export').glob('ttem_lsl_*.csv'))

def get_wt_from_filename(name:str)-> float:
    # from file name get top depth to water and bottom depth to water
    numbers = re.findall(r'\d+', name)
    number_top = numbers[-2]
    number_bottom = numbers[-1]
    return (int(number_top)+int(number_bottom))/2

def get_ttem_data_piece(name:(str,pathlib.PurePath),ttemdf:pd.DataFrame)-> pd.DataFrame:
    file = pd.read_csv(name)
    match = ttemdf[ttemdf['ID'].isin(file['ID'])]
    return match

def seperate_ttem_base_on_dtw(ttemdf:pd.DataFrame, wt:float) -> tuple:
    group = ttemdf.groupby("ID")
    ttem_above_wt = group.apply(lambda x: x[x['Depth_bottom']<=wt])
    ttem_above_wt.reset_index(inplace=True,drop=True)
    ttem_below_wt = group.apply(lambda x: x[x['Depth_bottom']>wt])
    ttem_below_wt.reset_index(inplace=True, drop=True)
    return ttem_above_wt, ttem_below_wt

def distance_of_two_points(point1,point2):
    distance = np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return distance
def linear_fill(group, factor=100):
    group.reset_index(drop=True, inplace=True)
    newgroup = group.loc[group.index.repeat(group.Thickness * factor)]
    mul_per_gr = newgroup.groupby('Elevation_Cell').cumcount()
    newgroup['Elevation_Cell'] = newgroup['Elevation_Cell'].subtract(mul_per_gr * 1 / factor)
    newgroup['Depth_top'] = newgroup['Depth_top'].add(mul_per_gr * 1 / factor)
    newgroup['Depth_bottom'] = newgroup['Depth_top'].add(1 / factor)
    newgroup['Elevation_End'] = newgroup['Elevation_Cell'].subtract(1 / factor)
    newgroup['Thickness'] = 1 / factor
    newgroup.reset_index(drop=True, inplace=True)
    return newgroup
def closest_rock(rk_trans, welllog):
    rk_trans_group = rk_trans.groupby(['UTMX','UTMY'])
    group_length = len(list(rk_trans_group.groups.keys()))
    welllog_coor_tuple = (welllog['X'].iloc[0],welllog['Y'].iloc[0])
    distance = list(map(distance_of_two_points,[welllog_coor_tuple for _ in range(group_length)],
                   list(rk_trans_group.groups.keys())))
    if min(distance) > 500:
        print('No close well found')
        ttem_data = pd.DataFrame()
        return ttem_data
    closest_coor=list(rk_trans_group.groups.keys())[distance.index(min(distance))]
    ttem_data = rk_trans_group.get_group(closest_coor)
    ttem_data = linear_fill(ttem_data, 100)
    return ttem_data

def value_search(ttem_data_df, welllog, WIN, rho_fine=10, rho_coarse=25,step=1, loop_range=20,correct=False):
    #progress = Bar('Processing', max=100)
    import itertools
    if isinstance(welllog,(str, pathlib.PurePath)):
        welllog_df = tt.core.process_well.ProcessWell(welllog)
        welllog_df = welllog_df.upscale(100)
    elif isinstance(welllog, pd.DataFrame):
        welllog_df = welllog
    welllog_WIN = welllog_df[welllog_df['Bore']==str(WIN)]
    try:
        ttem_data = closest_rock(ttem_data_df, welllog_WIN)
    except ttem_data.empty:
         raise ('No match Well log')
    fine_range = np.arange(rho_fine, rho_fine + (step * loop_range), step)
    coarse_range = np.arange(rho_coarse, rho_coarse + (step * loop_range), step)
    resistivity_list = list(itertools.product(fine_range, coarse_range))
    #Resi_conf_df = pd.DataFrame({'Fine_conf': [0, fine_rho], 'Mix_conf': [fine_rho, coarse_rho], 'Coarse_conf': [coarse_rho, 300]})
    welllog_WIN['Elevation_top'] = welllog_WIN['Elevation_top'].round(2)
    if correct is True:
        elevation_diff = welllog_WIN['Elevation_top'].iloc[0] - ttem_data['Elevation_Cell'].iloc[0]
        welllog_WIN['Elevation_top'] =welllog_WIN['Elevation_top'].subtract(elevation_diff)
        welllog_WIN['Elevation_bottom'] = welllog_WIN['Elevation_bottom'].subtract(elevation_diff)
    reslist = ['']*len(resistivity_list)
    corrlist = ['']*len(resistivity_list)
    i = 0
    for rho_fine, rho_coarse in resistivity_list:
        Resi_conf_df = pd.DataFrame(
            {'Fine_conf': [0, rho_fine],
             'Mix_conf': [rho_fine, rho_coarse],
             'Coarse_conf': [rho_coarse, 300]})
        reslist[i] = [rho_fine, rho_coarse]
        rk_trans = tt.core.Rock_trans.rock_transform(ttem_data, Resi_conf_df)
        merge = pd.merge(welllog_WIN, rk_trans, left_on=['Elevation_top'], right_on=['Elevation_Cell'])
        #corr = merge['Keyword_n'].corr(merge['Identity_n'])
        corr2 = (merge['Keyword_n'] == merge['Identity_n']).sum()/len(merge['Keyword_n'])
        #corrlist.append(corr)
        corrlist[i]=corr2
        i += 1
    def fine_best_corr(reslist, corrlist):
        corrlist = [np.nan_to_num(x) for x in corrlist]
        best_corr = max(corrlist)
        match_list = [i for i, x in enumerate(corrlist) if x == best_corr]
        resistivity_list = [reslist[i] for i in match_list]
        resistivity_coarse_gt_fine = [i for i in resistivity_list if i[1]>i[0]]
        res_bkup_incase_empty = np.array(resistivity_list)
        resistivity_array = np.array(resistivity_coarse_gt_fine)
        if resistivity_array.size > 0:
            fine_grained_rho_mean = resistivity_array[:,0].mean()
            coarse_grained_rho_mean = resistivity_array[:,1].mean()
            export_result = {'similiarity':best_corr,'Fine_conf':fine_grained_rho_mean,'Coarse_conf':coarse_grained_rho_mean}
            return export_result
        else:
            coarse_grained_rho_mean = res_bkup_incase_empty[:, 1].mean()
            fine_grained_rho_mean = coarse_grained_rho_mean

            export_result = {'similiarity': best_corr, 'Fine_conf': fine_grained_rho_mean,
                             'Coarse_conf': coarse_grained_rho_mean}
            return export_result
    #resi_conf_df1, best1 = fine_best_corr(reslist, corrlist)
    best= fine_best_corr(reslist, corrlist)

    #progress.finish()
    return best
def value_search_res(ttem_data_df, welllog, WIN,
                     rho_fine:float=10,
                     rho_mix:float=15,
                     rho_coarse:float=25,
                     step:int=1,
                     loop_range:int=20,correct=False):
    """
    Assign each lithology type as corresponsing resistivity and run pearson correlation to fine the best resistiviry overall
    :param ttem_data_df: tTEM resistivity profile
    :param welllog: well log data
    :param WIN: The WIN number of the well log
    :param rho_fine: resistivity of fine-grained material
    :param rho_mix: resistivity of mix-grained material
    :param rho_coarse: resistivity of coarse-grained material
    :param step: loop of each step
    :param loop_range: the total range of the loop
    :return:
    """
    import itertools
    pd.options.mode.chained_assignment = None
    if isinstance(welllog,(str, pathlib.PurePath)):
        welllog_df = tt.core.process_well.ProcessWell(welllog)
        welllog_df = welllog_df.upscale(100)
    elif isinstance(welllog, pd.DataFrame):
        welllog_df = welllog
    welllog_WIN = welllog_df[welllog_df['Bore']==str(WIN)]
    welllog_WIN.fillna('',inplace=True)
    try:
        ttem_data = closest_rock(ttem_data_df, welllog_WIN)
    except ttem_data.empty:
         raise ('well log expty')
    welllog_WIN['Elevation_top'] = welllog_WIN['Elevation_top'].round(2)
    if correct is True:
        elevation_diff = welllog_WIN['Elevation_top'].iloc[0] - ttem_data['Elevation_Cell'].iloc[0]
        welllog_WIN['Elevation_top'] =welllog_WIN['Elevation_top'].subtract(elevation_diff)
        welllog_WIN['Elevation_bottom'] = welllog_WIN['Elevation_bottom'].subtract(elevation_diff)

    fine_range = np.arange(rho_fine, rho_fine+(step*loop_range), step)
    mix_range = np.arange(rho_mix, rho_mix+(step*loop_range), step)
    coarse_range = np.arange(rho_coarse, rho_coarse+(step*loop_range), step)
    resistivity_list = list(itertools.product(fine_range, mix_range, coarse_range))
    corr_list = ['']*len(resistivity_list)
    i=0
    #total = len(resistivity_list)
    #count = 0

    merge = pd.merge(welllog_WIN, ttem_data, left_on=['Elevation_top'], right_on=['Elevation_Cell'])
    choicelist = [merge['Keyword_n'] == 1, merge['Keyword_n'] == 2, merge['Keyword_n'] == 3]
    for rho_fine, rho_mix, rho_coarse in resistivity_list:

        choicelist2 = [rho_fine, rho_mix, rho_coarse]
        welllog_resistivity = np.select(choicelist, choicelist2)
        corr = np.corrcoef(welllog_resistivity, merge['Resistivity'])[0,1]
        corr_list[i]=corr
        i+=1
        #count = count + 1
        #print('{}/{}'.format(count, total))
    def best_corr(reslist, corrlist):
        corrlist = [np.nan_to_num(x) for x in corrlist]
        best = max(corrlist)
        match_list = [i for i, x in enumerate(corrlist) if x == best]
        resistivity_list = [reslist[i] for i in match_list]
        resistivity_coarse_gt_fine = [i for i in resistivity_list if i[2] > i[0]]
        res_bkup_incase_empty = np.array(resistivity_list)
        resistivity_array = np.array(resistivity_coarse_gt_fine)
        if resistivity_array.size > 0:
            fine_rho_avg = resistivity_array[:,0].mean()
            mix_rho_avg = resistivity_array[:,1].mean()
            coarse_rho_avg = resistivity_array[:,2].mean()
            export_result = {'pearson':best,'Fine_average':fine_rho_avg,'Mix_average':mix_rho_avg,'Coarse_average':coarse_rho_avg,}
            return export_result
        else:

            mix_rho_avg = res_bkup_incase_empty[:, 1].mean()
            coarse_rho_avg = res_bkup_incase_empty[:, 2].mean()
            fine_rho_avg = coarse_rho_avg
            export_result = {'pearson':best,'Fine_average':fine_rho_avg,'Mix_average':mix_rho_avg,'Coarse_average':coarse_rho_avg,}
            return export_result
    resi_conf_df = best_corr(resistivity_list, corr_list)
    return resi_conf_df

def run_value_search(ttem_df, welllog, method = 'all',correct=False):
    if isinstance(welllog, (str, pathlib.PurePath)):
        welllog = tt.core.process_well.ProcessWell(welllog)
        welllog_df = welllog.upscale(100)
    elif isinstance(welllog, pd.DataFrame):
        welllog_df = welllog
    _, welllog_matched = tt.core.bootstrap.select_closest(ttem_df, welllog_df,500)
    if welllog_matched.empty: return []
    well_group = welllog_matched.groupby('Bore')
    result_dict = {}
    for name, group in well_group:
        ttem_single_match = closest_rock(ttem_df, group)
        if ttem_single_match.empty:
            continue
        if method == 1:
            result = value_search_res(ttem_single_match, group, name, rho_fine=5,
                                       rho_mix=10,
                                       rho_coarse=20,
                                       step=1,
                                       loop_range=40,correct=correct)
            result_dict[name] = result
        if method == 2:
            result2 = value_search(ttem_single_match, group, name,rho_fine=5, rho_coarse=20,step=1, loop_range=40,correct=correct)
            result_dict[name] = result2
        #result = value_search(ttem_df, group, name,rho_fine=5, rho_coarse=20,step=1, loop_range=40)
        if method == 'all':
            result = value_search_res(ttem_single_match, group, name, rho_fine=5,
                                       rho_mix=10,
                                       rho_coarse=20,
                                       step=1,
                                       loop_range=40,correct=correct)
            result2 = value_search(ttem_single_match, group, name,rho_fine=5, rho_coarse=20,step=1, loop_range=40,correct=correct)
            result_dict[name] = [result, result2]
        print({'{} is done'.format(name)})
    return result_dict
def main(filelist: list, ttemdf: pd.DataFrame, welllog:pathlib.PurePath) -> list:
    result_list = []
    well_upscaled = tt.core.process_well.ProcessWell(welllog)
    well_upscaled = well_upscaled.upscale(100)
    for segment_file in filelist:
        print(segment_file)
        wt = get_wt_from_filename(str(segment_file))
        print(wt)
        matchdf = get_ttem_data_piece(segment_file, ttemdf)
        ttem_above, ttem_below = seperate_ttem_base_on_dtw(matchdf, wt)
        result_above_water = run_value_search(ttem_above, well_upscaled)
        result_below_water = run_value_search(ttem_below, well_upscaled)
        result_dict = {segment_file.stem:{'result_above_water':result_above_water, 'result_below_water':result_below_water}}
        result_list.append(result_dict)
    return result_list

large_list = [(segment_north_ttem_list,tTEM_north_data), (segment_center_ttem_list,tTEM_center_data), (segment_north_east_ttem_list,tTEM_north_data), (segment_lsl_ttem_list,tTEM_lsl_data)]
large_result_list = []
concat_above_wt = []
concat_below_wt = []

for file in large_list:
    for segment_file in file[0]:
        wt = get_wt_from_filename(str(segment_file))
        matchdf = get_ttem_data_piece(segment_file, file[1])
        ttem_above, ttem_below = seperate_ttem_base_on_dtw(matchdf, wt)
        concat_above_wt.append(ttem_above)
        concat_below_wt.append(ttem_below)

ttem_above_wt = pd.concat(concat_above_wt)
ttem_below_wt = pd.concat(concat_below_wt)
well = tt.core.process_well.ProcessWell(welllog)
welllogs = well.data()
ttem_above_match, well_above_match = tt.core.bootstrap.select_closest(ttem_above_wt, welllogs, distance =500)
ttem_below_match, well_below_match = tt.core.bootstrap.select_closest(ttem_below_wt, welllogs, distance =500)
stitched_ttem_well_above, Resistivity_above, Thickness_ratio_above, matched_ttem_above, matched_well_above = tt.core.bootstrap.pre_bootstrap(ttem_above_wt, welllogs)
stitched_ttem_well_below, Resistivity_below, Thickness_ratio_below, matched_ttem_below, matched_well_below = tt.core.bootstrap.pre_bootstrap(ttem_below_wt, welllogs)
fine_Resistivity_above, mix_Resistivity_above, coarse_Resistivity_above = tt.core.bootstrap.bootstrap(Resistivity_above, Thickness_ratio_above)
fine_Resistivity_below, mix_Resistivity_below, coarse_Resistivity_below = tt.core.bootstrap.bootstrap(Resistivity_below, Thickness_ratio_below)
hist_above_result = pd.DataFrame({'fine':fine_Resistivity_above,'mix':mix_Resistivity_above,'coarse':coarse_Resistivity_above})
hist_below_result = pd.DataFrame({'fine':fine_Resistivity_below,'mix':mix_Resistivity_below,'coarse':coarse_Resistivity_below})
pack_above = tt.core.bootstrap.packup(fine_Resistivity_above,mix_Resistivity_above,coarse_Resistivity_above)
fig_above = plot_bootstrap_result(hist_above_result)
fig_below = plot_bootstrap_result(hist_below_result)
result = main(file[0], file[1], welllog)
large_result_list.append(result)
def plot_bootstrap_result(dataframe):
    import plotly.graph_objects as go
    """
    plot bootstrap result

    :param dataframe:
    :return: plotly fig
    """
    fig_hist = go.Figure()
    fig_hist.data = []
    fig_hist.add_trace(go.Histogram(x=dataframe.fine, name='Fine', marker_color='Blue', opacity=0.75))
    fig_hist.add_trace(go.Histogram(x=dataframe.coarse, name='Coarse', marker_color='Red', opacity=0.75))
    if dataframe.mix.sum() == 0:
        print("skip plot mix because there is no data")
    else:
        fig_hist.add_trace(go.Histogram(x=dataframe.mix, name='Mix', marker_color='Green', opacity=0.75))
    fig_hist.update_layout(paper_bgcolor='white',plot_bgcolor='white',font_color='black')
    fig_hist.update_layout(
        xaxis=dict(
            title='Resistivity (ohm-m)',
            title_font=dict(
                family='Arial',
                size=50,
                #weight='bold'
            ),
            tickfont=dict(
                family='Arial',
                size=45,
                #weight='bold'
            )
            #tickmode='linear',
            #tick0=1774,
            #dtick=100
        ),
        yaxis=dict(
            title='Counts',
            title_font=dict(
                family='Arial',
                size=50,
                #weight='bold'
            ),
            tickfont=dict(
                family='Arial',
                size=45,
                #weight='bold'

            )

            #tickmode='linear',
            #tick0=1774,
            #dtick=100
        ),
        legend=dict(
            font=dict(
                family='Arial',
                size=25
            )
        )
    )
    return fig_hist
#TODO Linear
all_above_water = wt
def bootstrap_seperate()