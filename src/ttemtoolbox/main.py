#!/usr/bin/env python
from ttemtoolbox import process_ttem, process_gamma, process_well, process_water, lithology_connect
from ttemtoolbox import tools
from ttemtoolbox import __version__
from pathlib import Path
import argparse
import shutil
import sys
import geopandas as gpd
import pandas as pd


def create_parser():
    synopsis = 'This is a python interface for ttemtoolbox program'
    name = 'ttemtoolbox'
    parser = argparse.ArgumentParser(
        name, description=synopsis)
    parser.add_argument('-c',"--config_path", metavar="PATH", help = 'Run entire ttem rock physics tranform process')
    parser.add_argument("--get_config",action='store_true', help='Generate default config file')
    parser.add_argument("-f","--force_clean", help="To force remove all files for new program",
                        action="store_true")
    parser.add_argument("--example_data", help="To download example data",
                        action="store_true")
    parser.add_argument("-v", "--version", action='version', version='ttemtoolbox {}'.format(__version__))
    subparser = parser.add_subparsers()
    subparser_ttem = subparser.add_parser('ttem')
    subparser_ttem.add_argument('ttem', metavar='PATH', help = 'Path to config file')
    subparser_ttem.add_argument('--doi_path', metavar='PATH', help='Path to doi file')
    subparser_ttem.add_argument('--layer_exclude', nargs='+', metavar='int(s)', type=int,
                                   help='Specify exclude layers when processing ttem data, \
                                   this can also be done in config file')
    subparser_ttem.add_argument('--line_exclude', nargs='+', metavar='int(s)', type=int,
                                   help='Specify exclude lines when processing ttem data, \
                                   this can also be done in config file')
    subparser_ttem.add_argument('--ID_exclude', nargs='+', metavar='int(s)', type=int,
                                help='Specify exclude ID when processing ttem data, \
                                   this can also be done in config file')
    subparser_ttem.add_argument('--resample', metavar='int', type=int,
                                help='Specify resample factor when processing ttem data, \
                                   this can also be done in config file')
    subparser_ttem.add_argument('--reproject', metavar='str', type=str, help='Reproject ttemdata to a new crs,\
                                e.g: EPSG:4326')
    subparser_ttem.add_argument('--unit', metavar='str', help='Use "meter" or "feet", default is meter')
    subparser_lithology = subparser.add_parser('lithology')
    subparser_lithology.add_argument('lithology', metavar='PATH', help = 'Path to config file')
    subparser_lithology.add_argument('--reproject', metavar='str', type=str, help='Reproject welllog data to a new crs')
    subparser_lithology.add_argument('--resample', metavar='int', type=int, help='Resample welllog data')
    subparser_lithology.add_argument('--unit', metavar='str', help='"meter" or "feet", default is meter')
    subparser_water = subparser.add_parser('water')
    subparser_water.add_argument('water', metavar='PATH', help = 'Path to config file')
    subparser_water.add_argument('-w','--well_no', metavar='str[s]', help='Download specific well number')
    subparser_water = subparser.add_parser('connect')
    subparser_water.add_argument('connect', metavar='PATH', help = 'Path to config file')
    return parser

def cmd_line_parse(iargs=None):
    default_config_path = Path(__file__).parent.joinpath('defaults/CONFIG')
    default_data_path = Path(__file__).parents[2].joinpath('data')
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if inps.config_path:
        inps.config_path = Path(inps.config_path).resolve()
        if inps.config_path.is_dir():
            inps.config_path = inps.config_path.joinpath('CONFIG')
    if inps.force_clean:
        print('All result will be purged')
    if inps.get_config:
        copypath = Path.cwd().joinpath('CONFIG')
        shutil.copyfile(default_config_path, copypath)
        print('Default CONFIG file generated in {}'.format(copypath))
        sys.exit(0)
    if inps.example_data:
        copypath = Path.cwd().joinpath('data')
        data_files = default_data_path.glob('*')
        for file in data_files:
            shutil.copy(file, copypath)
        sys.exit(0)

    return inps

def step_ttem(config: dict, inps: dict) -> gpd.GeoDataFrame:
    print('Step1: Process ttem')
    if inps.get('layer_exclude'):
        config['layer_exclude'] = inps['layer_exclude']
    if inps.get('line_exclude'):
        config['line_exclude'] = inps['line_exclude']
    if inps.get('ID_exclude'):
        config['ID_exclude'] = inps['ID_exclude']
    if inps.get('resample'):
        config['ttem_resample'] = inps['resample']
    else:
        config['ttem_resample'] = None
    if inps.get('doi_path'):
        config['doi_path'] = inps['doi_path']
    if inps.get('reproject'):
        config['ttem_reproject_crs'] = inps['reproject']
    if inps.get('unit'):
        config['ttem_unit'] = inps['unit']

    if Path(config['ttem_path']).is_file():
        
        ttem = process_ttem.ProcessTTEM(
            fname = config['ttem_path'],
            doi_path=config['doi_path'],
            layer_exclude = config['layer_exclude'],
            line_exclude = config['line_exclude'],
            ID_exclude = config['ID_exclude'],
            resample = config['ttem_resample'],
            unit = config['ttem_unit']
        )
    else:
        raise TypeError('TTEM file not found in {}'.format(config['ttem_path']))
    if config['ttem_crs'] is not None:
        ttem.set_crs(config['ttem_crs'])
    if config['ttem_reproject_crs'] is not None:
        ttem.reproject(config['ttem_reproject_crs'])
    
    ttem.summary().to_csv(config['deliver'].joinpath(Path(config['ttem_path']).stem + '_summary.csv'), index=False)
    ttem.to_shp(config['deliver'].joinpath(Path(config['ttem_path']).stem + '.shp'))
    ttem.data.to_csv(config['ttem_temp'].joinpath(Path(config['ttem_path']).stem+ '.csv'), index=False)
    return ttem.data


def step_lithology(config: dict, inps: dict)-> gpd.GeoDataFrame:
    print('Step2: Process lithology')
    if inps.get('reproject'):
        config['lithology_reproject_crs'] = inps['reproject']
    if inps.get('resample'):
        config['lithology_resample'] = inps['resample']
    if inps.get('unit'):
        config['lithology_unit'] = inps['unit']
    if Path(config['well_path']).is_file():
        lithology = process_well.ProcessWell(
            fname = config['well_path'],
            crs=config['lithology_crs'],
            unit = config['lithology_unit']
        )
    else:
        raise TypeError('Lithology file not found in {}'.format(config['well_path']))
    if config['lithology_resample'] is not None:
        lithology.resample(config['lithology_resample'])
    if config['lithology_reproject_crs'] is not None:
        lithology.reproject(config['lithology_reproject_crs'])
    lithology.to_shp(config['deliver'].joinpath(Path(config['well_path']).stem + '.shp'))
    lithology.data.to_csv(config['well_temp'].joinpath(Path(config['well_path']).stem+ '.csv'), index=False)
    return lithology.data


def step_gamma(config, inps):
    return

def step_water(config : dict, inps: dict) -> tuple:
    print('Step4: Process water level data')
    if inps.get('well_no'):
        config['USGS_well_NO'] = inps['well_no']
    concat_list = []
    meta_data_list = []
    for well in config['USGS_well_NO']:
        water = process_water.format_usgs_water(well, config['water_temp'])
        concat_list.append(water)
        meta_data_list.append(pd.DataFrame(water.attrs, index=[0]))
    water = pd.concat(concat_list)
    water.reset_index(drop=True, inplace=True)
    meta = pd.concat(meta_data_list)
    meta.reset_index(drop=True, inplace=True)
    with pd.ExcelWriter(config['deliver'].joinpath('water_level.xlsx')) as writer:
        water.to_excel(writer, sheet_name='water_level', index=False)
        meta.to_excel(writer, sheet_name='metadata', index=False)
    print('Water level data saved in {}'.format(config['deliver'].joinpath('water_level.xlsx')))
    return water, meta


def step_connect(config: dict, inps:dict, 
                 ttem: gpd.GeoDataFrame = None, 
                 lithology: gpd.GeoDataFrame = None):
    if ttem is None and lithology is None:
        ttemlist = Path(config['ttem_temp']).glob('*.csv')
        temp_ttem = pd.concat([pd.read_csv(file) for file in ttemlist])
        ttem =  gpd.GeoDataFrame(temp_ttem, geometry=gpd.points_from_xy(temp_ttem['X'], temp_ttem['Y']), 
                                crs=config['ttem_reproject_crs'])
        lithologylist = Path(config['well_temp']).glob('*.csv')
        temp_lithology = pd.concat([pd.read_csv(file) for file in lithologylist])
        lithology = gpd.GeoDataFrame(temp_lithology, geometry=gpd.points_from_xy(temp_lithology['X'], temp_lithology['Y']), 
                                crs=config['lithology_reproject_crs'])
    matched_ttem, matched_lithology = lithology_connect.select_closest(ttem, lithology,
                                                                       search_radius = config['search_radius'])
    stitched = lithology_connect.ttem_well_connect(matched_ttem, matched_lithology)
    stitched.to_csv(config['deliver'].joinpath('_ttem_well_connect.csv'))
    return stitched
    
def main(iargs=None):
    inps = vars(cmd_line_parse(iargs)) # parse CLI input to dict
    #########run entire ttem rock physics tranform process
    if inps.get('config_path'):
        print('Run entire ttem rock physics tranform process')
        user_config = tools.parse_config(inps['config_path'])
        tools.clean_output(Path(user_config['output']))
        config = tools.create_dir_structure(user_config)
        ttemdata = step_ttem(config, inps)
        
    if inps.get('force_clean'):
        print('All result will be purged')
        tools.clean_output()
        
    #########Step1: Process ttem
    if inps.get('ttem'):
        user_config = tools.parse_config(inps['ttem'])
        tools.clean_output(Path(user_config['output']))
        config = tools.create_dir_structure(user_config)
        step_ttem(config, inps)
        
    #########Step2: Process lithology
    if inps.get('lithology'):
        user_config = tools.parse_config(inps['lithology'])
        tools.clean_output(Path(user_config['output']))
        config = tools.create_dir_structure(user_config)
        step_lithology(config, inps)
        
    #########Step3: Process gamma
    if inps.get('gamma'):
        print('This feature is still under development ')
    #########Step4: Process water level
    if inps.get('water'):
        user_config = tools.parse_config(inps['water'])
        tools.clean_output(Path(user_config['output']))
        config = tools.create_dir_structure(user_config)
        step_water(config, inps)
    #########Step5: ttem well connect
    if inps.get('connect'):
        user_config = tools.parse_config(inps['connect'])
        tools.clean_output(Path(user_config['output']))
        config = tools.create_dir_structure(user_config)
        step_connect(config, inps)

        
'''
class ProcessTTEM():
    def __init__(self, ttem_path, **kwargs):
        self.ttem_path = ttem_path
        self.kwargs = kwargs
        ttem = core.process_ttem.format_ttem(fname=self.ttem_path, **self.kwargs)
        self.ttem = ttem
    def data(self):
        return self.ttem
    def fill(self, factor=100):
        ttem_fill = core.process_ttem.format_ttem(fname=self.ttem, filling=True, factor=factor)
        return ttem_fill
    def ttem_well_connect(self, distance=500, wellupscale=100, check_corr=np.nan, debug=False):
        if np.isin("welllog", list(self.kwargs.keys())):
            if isinstance(self.kwargs['welllog'], (str, pathlib.PurePath)):
                welllog_upscale = core.process_well.format_well(self.kwargs["welllog"], upscale=wellupscale)
            elif isinstance(self.kwargs['welllog'], pd.DataFrame):
                welllog_upscale = self.kwargs['welllog']
            if ~np.isnan(check_corr):
                ttem_match, well_match = core.bootstrap.select_closest(self.ttem, welllog_upscale,
                                                                       distance=distance, showskip=False)
                welllog_upscale = core.bootstrap.corr_well_filter(ttem_match, well_match, corr_thershold=check_corr)
                pre_bootstrap, Resistivity, Thickness_ratio,matched_ttem, matched_well = core.bootstrap.pre_bootstrap(self.ttem, welllog_upscale,
                                                                                                                      distance=distance)
            else:
                pre_bootstrap, Resistivity, Thickness_ratio,matched_ttem, matched_well = core.bootstrap.pre_bootstrap(self.ttem, welllog_upscale,
                                                                                                                      distance=distance)
            fine, mix, coarse = core.bootstrap.bootstrap(Resistivity, Thickness_ratio)
            pack_bootstrap_result = pd.DataFrame({'fine':fine,'mix':mix,'coarse':coarse})
            Resi_conf_df = core.bootstrap.packup(fine, mix, coarse)
            rk_trans= core.Rock_trans.rock_transform(self.ttem, Resi_conf_df)
        else:
            raise ValueError("welllog not found!")
        if debug is True:
            return pre_bootstrap,rk_trans,Resistivity, Thickness_ratio, pack_bootstrap_result, Resi_conf_df, matched_ttem, matched_well
        else:
            return rk_trans, pack_bootstrap_result
class ProcessGamma():
    def __init__(self, gamma, gammaloc, **kwargs):
        self.gamma = gamma
        self.gammaloc = gammaloc
        self.kwargs = kwargs
    def process(self):
        if np.isin('columns', list(self.kwargs.keys())):
            ori = core.process_gamma.load_gamma(self.gamma, columns=self.kwargs["columns"])
        else:
            ori = core.process_gamma.load_gamma(self.gamma)
        ori = core.process_gamma.georeference(ori, self.gammaloc)
        if np.isin("rolling", list(self.kwargs.keys())):
            if self.kwargs["rolling"] is True:
                try:
                    ori = core.process_gamma.rolling_average(ori,
                                                             window=self.kwargs["window"])
                except:
                    ori = core.process_gamma.rolling_average(ori)
        return ori
    def gr_wlg_combine(self,ori):
        try:
            gr_wlg = core.process_gamma.gamma_well_connect(ori, self.kwargs["welllog"])
        except:
            raise("Process gamma data first use .process()")

        return gr_wlg

    def gr_ttem_combine(self, ori, **kwargs):
        ttem_result = pd.DataFrame()
        if np.isin("ttem", list(kwargs.keys())):
            try:
                ttem_result = kwargs["ttem"]
            except:
                try:
                    ttem_result = ProcessTTEM(ttemname=kwargs["ttem"],
                                                    welllog=kwargs['welllog'],
                                                    DOI=kwargs['DOI']).data()
                except:
                    raise ValueError("\nMissing parameters")
        try:
            ori_result = ori
        except:
            raise ValueError("Process gamma data first use .process()")
        gr_ttem = core.process_gamma.gamma_ttem_connect(ori_result, ttem_result)
        return gr_ttem
class GWSurface():
    """
    Parameters
    ----------
    waterwell : could be a single well or path to a file that contains a list of well name.
    elevation_type: Choose to keep which type of data could be 'NAVD88', 'NGVD29', or ''depth'
    time: Select a time period to export, could be a single time or a list of time, format YYYY-MM-DD
    """
    def __init__(self, waterwell,
                 elevation_type='NAVD88',
                 *args, **kwargs):
        self.well = waterwell
        self.elevation_type = elevation_type
        self.args = args
        self.df = pd.DataFrame()
        self.kwargs = kwargs
        if isinstance(self.well, int):
            print('reading wells in file {}'.format(self.well))
            self.well = pd.read_excel(self.well)
            self.well_list = self.well['SiteNumber'].values
        elif isinstance(self.well, (str, pathlib.PurePath)):
            try:
                self.well = pd.read_excel(self.well)
            except:
                try:
                    self.well= pd.read_csv(self.well)
                except:
                    print('{} is not in xls or xlsx format try read as csv'.format(self.well))
            self.well_list = self.well['SiteNumber'].values
        self.ds = xr.Dataset()
        for i in self.well_list:
            tmp_ds = core.process_well.format_usgs_water(str(i), self.elevation_type, **self.kwargs)
            if tmp_ds is None:
                print('{} does not have water level data'.format(i))
                continue
            try:
                self.ds = self.ds.merge(tmp_ds)
                # ds = ds.merge(tmp_ds)
            except:
                print('{} not able to merge, try to solve the problem by drop duplicates.'.format(str(i)))
                try:
                    self.ds = self.ds.merge(tmp_ds[str(i)].drop_duplicates(dim='time').to_dataset())
                except:
                    print('{} failed to merge'.format(str(i)))
        print('All Wells Done!')
    def data(self):
        """
        :return: returns a xarray dataset include formatted USGS water well information
        """
        return self.ds
    def format(self, elevation=None, time='2022-03'):
        if self.elevation_type in ['NAVD88', 'NGVD29']:
            header = 'sl_lev_va'
        elif self.elevation_type.lower() == 'depth':
            header = 'lev_va'
        else:
            raise ValueError('{} not one of NAVD88, NGVD29, or depth'.format(self.elevation_type))
        if isinstance(time, str):
            self.df = core.process_well.water_head_format(self.ds, time=str(time), header=header, elevation=elevation)
        else:
            self.df = core.process_well.water_head_format(self.ds, time=str(time[0]), header=header, elevation=elevation)
            for i in time[1:]:
                tmp_df = core.process_well.water_head_format(self.ds, time=str(i), header=header, elevation=elevation)
                self.df = self.df.merge(tmp_df, how='left', on=['wellname','lat','long','datum','UTMX','UTMY','well_depth','altitude'])
        return self.df


#df.sl_lev_va=df.sl_lev_va.astype(float).div(3.2808)

'''
if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('-h')
    main(sys.argv[1:])




