#!/usr/bin/env python
import numpy as np
import pandas as pd
import xarray as xr
import pathlib
from ttemtoolbox import process_ttem, process_gamma, process_well
from ttemtoolbox import tools
from ttemtoolbox import __version__
from pathlib import Path
import argparse
import shutil
import sys
#import concurrent.


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
    parser.add_argument("-v", "--version", action='version', version='lastree {}'.format(__version__))
    subparser = parser.add_subparsers()
    subparser_ttem = subparser.add_parser('ttem')
    subparser_ttem.add_argument('ttem', metavar='PATH', help = 'Path to config file')
    subparser_ttem.add_argument('--layer_exclude', nargs='+', metavar='int(s)', type=int,
                                   help='Specify exclude layers when processing ttem data, \
                                   this can also be done in config file')
    subparser_ttem.add_argument('--line_exclude', nargs='+', metavar='int(s)', type=int,
                                   help='Specify exclude lines when processing ttem data, \
                                   this can also be done in config file')
    subparser_ttem.add_argument('--ID_exclude', nargs='+', metavar='int(s)', type=int,
                                help='Specify exclude ID when processing ttem data, \
                                   this can also be done in config file')
    subparser_lithology = subparser.add_parser('lithology')
    subparser_lithology.add_argument('lithology', metavar='PATH', help = 'Path to config file')
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


def main_process_ttem(config, 
                      layer_exclude: list = None, 
                      line_exclude: list = None, 
                      ID_exclude: list = None, 
                      resample: int = None):
    if layer_exclude is not None:
        layer_exclude_copy = layer_exclude
    else:
        print('found layer_exclude in config file')
        layer_exclude_copy = config['layer_exclude']
    if line_exclude is not None:
        line_exclude_copy = line_exclude
    else:
        print('found line_exclude in config file')
        line_exclude_copy = config['line_exclude']
    if ID_exclude is not None:
        ID_exclude_copy = ID_exclude
    else:
        print('found ID_exclude in config file')
        ID_exclude_copy = config['ID_exclude']
    if resample is not None:
        resample_copy = resample
    else:
        print('found resample in config file')
        resample_copy = config['resample']
    ttem = process_ttem.ProcessTTEM(
        fname = config['ttem_path'],
        doi_path=config['doi_path'],
        layer_exclude = layer_exclude_copy,
        line_exclude = line_exclude_copy,
        ID_exclude = ID_exclude_copy,
        resample = resample_copy
    )
    ttem_data = ttem.data()
    ttem_summary = ttem.summary()
    ttem_summary = ttem_summary.to_csv(config['delivered_folder'] + Path(config['ttem_path']).stem + '_summary.csv')
    ttem
    
    
    
def main(iargs=None):
    inps = vars(cmd_line_parse(iargs)) # parse to dict
    if inps.get('ttem'):
        print('Process ttem')
        config = tools.parse_config(inps['ttem'])
        ttem = main_process_ttem(config, )
    if inps.get('lithology'):
        print('process lithology')
    return print(inps)
        
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




