import ttemtoolbox as tt

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