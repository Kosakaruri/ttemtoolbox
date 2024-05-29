#!/usr/bin/env python
# process_well.py
# Created: 2023-11-18
# Version 11.18.2023
# Author: Jiawei Li
import os
import pathlib
import re
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from itertools import compress
from pathlib import Path
from ttemtoolbox.defaults import constants
from ttemtoolbox import utils
from ttemtoolbox.utils import tools
from collections import namedtuple
class ProcessWell:
    """
    This class is use to process and format lithology well logs (from excel or csv) and water level data (from USGS).\
    All data were assume under metric unit (m). \n
    :param lithologyfname: one or a list of string, pathlib.PurePath object, pandas dataframe. The input files(s) \
            shall be either csv or excel file(s) that contains lithology and location data. sheet name and column name \
            needs to be clearly marked as Lithology, Location, Latitude, Longitude, Depth_top, Depth_bottom or anything\
            similiar, keyword(s) can be modified under tTEM_toolbox/defaults/constants.py.\
    """
    def __init__(self,
                 fname: str| pathlib.PurePath | list,
                 crs: str = 'epsg:4326',
                 unit: str = 'feet'):
        if isinstance(fname, str | pathlib.PurePath):
            self.fname = [fname]
            print('reading lithology from {}'.format(Path(fname).name))
        elif isinstance(fname, list):
            if len(fname) == 0:
                raise ValueError('Input file path is empty')
            else:
                self.fname = fname
                print('reading lithology from {}'.format([Path(f).name for f in fname]))
        if unit == 'feet': 
            self.unit = 'feet'
            self.unitconvert = 3.28084
        elif unit == 'meter':
            self.unit = 'meter'
            self.unitconvert = 1
        self._crs = crs
        self.data = self._format_well()
        self.crs = self.data.crs
        


    @staticmethod
    def _find_all_readable(path:pathlib.PurePath)->list:
        """
        This will receive a single path-like input and try to filter all readable file paths for well logs uses.
        :param path: path-like pathlib.PurePath object or string
        :return: list of pathlib.PurePath objects
        """
        readable_ext = constants.CSV_EXTENSION + constants.EXCEL_EXTENSION
        if not isinstance(path, (str, pathlib.PurePath)):
            raise TypeError('Input path must be a string or pathlib.PurePath object')

        if Path(path).is_dir():
            file_list = [f for f in Path(path).iterdir() if f.suffix in readable_ext]
            if len(file_list) == 0:
                raise ValueError('No {} file found in {}'.format(readable_ext, path))
            return file_list
        elif Path(path).is_file():
            if Path(path).suffix in readable_ext:
                file_list = [path]
            else:
                raise ValueError('Input file does not have extension of {}'.format(readable_ext))
            return file_list

    @staticmethod
    def _format_input(fname:str| pathlib.PurePath| list| pd.DataFrame) -> list:
        """
        This will format input file path(s) to a list of pandas dataframe (read from csv) and/or dict that includes all sheets in the excel\
         file, each sheet were pandas dataframe. If input is a pandas dataframe, it will return the input dataframe in a list.
        :param fname: one or a list of string, pathlib.PurePath object, pandas dataframe
        :return: a list of pandas dataframe and/or dict
        """
        if isinstance(fname, (str, pathlib.PurePath)):
            fname = [fname]
        elif isinstance(fname, list):
            pass
        else:
            raise TypeError('Input must be one or a list of string, pathlib.PurePath objects')
        export_list = []
        for path in fname:
            file_list = ProcessWell._find_all_readable(path)
            excels = [file for file in file_list if Path(file).suffix in constants.EXCEL_EXTENSION]
            csvs = [file for file in file_list if Path(file).suffix in constants.CSV_EXTENSION]
            read_excels = [pd.read_excel(file, sheet_name=None) for file in excels]
            read_csvs = [pd.read_csv(file) for file in csvs]
            combined = read_excels + read_csvs
            export_list.append(combined)
        result = [item for sublist in export_list for item in sublist]
        return result
    @staticmethod
    def _read_lithology(fname: str| pathlib.PurePath |list| pd.DataFrame, mtoft=1) -> pd.DataFrame:
        """
        Try to read lithology sheet from Excel file with tab name similar to 'Lithology', or csv file contains lithology data.
        :param fname: one or a list of string, pathlib.PurePath object, pandas dataframe
        :return:
        """
        result = ProcessWell._format_input(fname)
        lithology_list = []
        for single_file in result:
            if isinstance(single_file, dict):  # which means it is an Excel file
                match_sheet_name = tools.keyword_search(single_file, constants.LITHOLOGY_SHEET_NAMES)
                if len(match_sheet_name) == 0:
                    continue
                lithology_sheet = single_file[match_sheet_name[0]]
                lithology_list.append(lithology_sheet)
            if isinstance(single_file, pd.DataFrame):  # which means it is a csv file
                match_column_lithology = tools.keyword_search(single_file, constants.LITHOLOGY_COLUMN_NAMES_KEYWORD)
                if match_column_lithology > 0:
                    lithology_sheet = single_file
                    lithology_list.append(lithology_sheet)
        concat_list = []
        for sheet in lithology_list:
            match_column_lithology = tools.keyword_search(sheet, constants.LITHOLOGY_COLUMN_NAMES_KEYWORD)
            match_column_bore = tools.keyword_search(sheet, constants.LITHOLOGY_COLUMN_NAMES_BORE)
            match_column_depth_top = tools.keyword_search(sheet, constants.LITHOLOGY_COLUMN_NAMES_DEPTH_TOP)
            match_column_depth_bottom = tools.keyword_search(sheet, constants.LITHOLOGY_COLUMN_NAMES_DEPTH_BOTTOM)

            lithology = pd.DataFrame(sheet[match_column_lithology[0]])
            lithology.columns = ['Keyword']
            lithology['Bore'] = sheet[match_column_bore[0]]
            lithology['Depth_top'] = sheet[match_column_depth_top[0]]/mtoft
            lithology['Depth_top']= lithology['Depth_top'].round(2)
            lithology['Depth_bottom'] = sheet[match_column_depth_bottom[0]]/mtoft
            lithology['Depth_bottom'] = lithology['Depth_bottom'].round(2)
            lithology['Thickness'] = lithology['Depth_bottom'].subtract(lithology['Depth_top'])

            concat_list.append(lithology)
        result = pd.concat(concat_list)
        result = result[['Bore', 'Depth_top', 'Depth_bottom', 'Thickness', 'Keyword']]
        return result

    @staticmethod
    def _read_spatial(fname: str| pathlib.PurePath, mtoft=1) -> pd.DataFrame:
        """
        Similiar to _read_lithology, but read location sheet from Excel file with tab name similar to 'Location', \
        or csv file contains location data.
        :param fname: fname: one or a list of string, pathlib.PurePath object, pandas dataframe
        :return:
        """
        result = ProcessWell._format_input(fname)
        location_list = []
        for single_file in result:
            if isinstance(single_file, dict):
                match_sheet_name = utils.tools.keyword_search(single_file, constants.LOCATION_SHEET_NAMES)
                if len(match_sheet_name) == 0:
                    continue
                location_sheet = single_file[match_sheet_name[0]]
                location_list.append(location_sheet)
            if isinstance(single_file, pd.DataFrame):
                match_column_location = utils.tools.keyword_search(single_file, constants.LOCATION_COLUMN_NAMES_LON)
                if match_column_location > 0:
                    location_sheet = single_file
                    location_list.append(location_sheet)
        concat_list = []
        for sheet in location_list:
            match_column_lat = utils.tools.keyword_search(sheet, constants.LOCATION_COLUMN_NAMES_LAT)
            match_column_lon = utils.tools.keyword_search(sheet, constants.LOCATION_COLUMN_NAMES_LON)
            match_column_elevation = utils.tools.keyword_search(sheet, constants.LOCATION_COLUMN_NAMES_ELEVATION)
            location = pd.DataFrame(sheet[match_column_lat[0]])
            location.columns = ['Latitude']
            location['Longitude'] = sheet[match_column_lon[0]]
            location['Bore'] = sheet['Bore']
            location['Elevation'] = sheet[match_column_elevation[0]]/mtoft
            location['Elevation'] = location['Elevation'].round(2)
            concat_list.append(location)
        result = pd.concat(concat_list)
        return result


    @staticmethod
    def _fill(group, factor=100) -> pd.DataFrame:
        newgroup = group.loc[group.index.repeat(group.Thickness * factor)]
        mul_per_gr = newgroup.groupby('Elevation_top').cumcount()
        newgroup['Elevation_top'] = newgroup['Elevation_top'].subtract(mul_per_gr * 1 / factor)
        newgroup['Depth_top'] = newgroup['Depth_top'].add(mul_per_gr * 1 / factor)
        newgroup['Depth_bottom'] = newgroup['Depth_top'].add(1 / factor)
        newgroup['Elevation_bottom'] = newgroup['Elevation_top'].subtract(1 / factor)
        newgroup['Thickness'] = 1 / factor
        return newgroup

    @staticmethod
    def _lithology_location_connect(lithology: pd.DataFrame,
                                   location: pd.DataFrame) -> pd.DataFrame:
        """
        Connect lithology and location data by Borehole ID
        :param lithology: lithology dataframe
        :param location: location dataframe
        :return: combined dataframe
        """
        lithology_group = lithology.groupby('Bore')
        concatlist = []
        for name, group in lithology_group:
            group_location = location[location['Bore'] == name]
            if group_location.empty:
                continue
            group['Y'] = group_location['Latitude'].iloc[0]
            group['X'] = group_location['Longitude'].iloc[0]
            group['Z'] = group_location['Elevation'].iloc[0]
            group['Elevation_top'] = group['Z'].subtract(group['Depth_top'])
            group['Elevation_bottom'] = group['Z'].subtract(group['Depth_bottom'])
            concatlist.append(group)
        result = pd.concat(concatlist)
        return result


    @staticmethod
    def _assign_keyword_as_value(welllog_df) -> pd.DataFrame:
        conditionlist = [
            (welllog_df["Keyword"] == "fine grain"),
            (welllog_df["Keyword"] == "mix grain"),
            (welllog_df["Keyword"] == "coarse grain")
        ]
        choicelist = [1, 2, 3]
        welllog_df["Keyword_n"] = np.select(conditionlist, choicelist)
        return welllog_df


    def _format_well(self) -> gpd.GeoDataFrame:
        lithology = self._read_lithology(self.fname, self.unitconvert)
        location = self._read_spatial(self.fname, self.unitconvert)
        self.data = self._lithology_location_connect(lithology, location)
        self.data = ProcessWell._assign_keyword_as_value(self.data)
        self.data.reset_index(drop=True, inplace=True)
        gdf = gpd.GeoDataFrame(self.data, geometry=gpd.points_from_xy(self.data['X'], self.data['Y']), 
                               crs=self._crs)
        self.data = gdf
        return self.data

    def reproject(self, crs: str) -> gpd.GeoDataFrame:
        """
        Reproject the data to a given coordinate system.

        Parameters:
        - crs (str): The coordinate system to reproject the data to.

        Returns:
        - geopandas.GeoDataFrame: The reprojected data.
        """
        self.data = self.data.to_crs(crs)
        self._crs = crs
        self.data['X'] = self.data.geometry.x
        self.data['Y'] = self.data.geometry.y
        return self.data
    
    
    def resample(self, scale: int) -> gpd.GeoDataFrame:
        """
        Upscales the data by a given scale factor.

        Parameters:
        - scale (int): The scale factor to upscale the data by.

        Returns:
        - geopandas.GeoDataFrame: The upscaled data.
        """
        group = self.data.groupby('Bore')
        self.data = group.apply(lambda x:ProcessWell._fill(x, scale))
        self.data.reset_index(drop=True, inplace=True)
        print('resampling lithology to {} '.format(1/scale))
        return self.data
    
    def summary(self):
        groups = self.data.groupby('Bore')
        concat_list = []
        for bore, group in groups: 
            total_thickness = group['Thickness'].sum()
            keywordgroup = group.groupby('Keyword')
            keyword_summary = keywordgroup.agg({
                'Thickness': 'sum',
                'X': 'first',
                'Y': 'first',
                'Z': 'first'
            })
            keyword_summary[keyword_summary.index.name] = keyword_summary.index.values
            keyword_summary['ratio'] = keyword_summary['Thickness'] / total_thickness
            keyword_summary.reset_index(drop=True, inplace=True)
            keyword_summary['bore'] = bore
            keyword_summary['unit'] = 'meter'
            keyword_summary['total_thickness'] = total_thickness
            concat_list.append(keyword_summary)
        output = pd.concat(concat_list)
        return output
    
    def to_shp(self, output_filepath: str| pathlib.PurePath) -> None:
        """
        Save the data to a shapefile.

        Parameters:
        - path (str | pathlib.PurePath): The path to save the shapefile to.
        """
        summary = self.summary()
        gdf = gpd.GeoDataFrame(summary, geometry=gpd.points_from_xy(summary['X'], summary['Y']), 
                               crs=self._crs)
        if  Path(output_filepath).suffix.lower() == '.shp':
            gdf.to_file(output_filepath, driver='ESRI Shapefile')
            print('The output file saved to {}'.format(Path(output_filepath).resolve()))
        elif Path(output_filepath).suffix.lower() == '.gpkg':
            gdf.to_file(output_filepath, driver='GPKG', layer=Path(self.fname[0]).stem)
            print('The output file saved to {}'.format(Path(output_filepath).resolve()))
        elif Path(output_filepath).suffix.lower() == '.geojson':
            gdf.to_file(output_filepath, driver='GeoJSON')
            print('The output file saved to {}'.format(Path(output_filepath).resolve()))
        else: 
            raise ValueError("The output file format is not supported, please use .shp, .gpkg, or .geojson")

if __name__ == "__main__":
    print('This is a module, please import it to use it.')
    a = ProcessWell([r'C:\Users\jldz9\PycharmProjects\tTEM_toolbox\data\Well_log.xlsx'])

