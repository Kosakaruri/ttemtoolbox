#!/usr/bin/env python
# process_ttem.py
# Version 11.18.2023
# Created: 2023-11-17
# Author: Jiawei Li
import pathlib
from pathlib import Path
import re
import pandas as pd
import geopandas as gpd
import numpy as np
from ttemtoolbox.defaults.constants import XYZ_FILE_PATTERN, DOI_FILE_PATTERN
from ttemtoolbox.utils.tools import skip_metadata


class ProcessTTEM:
    """
    This function is used to format the tTEM data, and return a dataframe that contains filtered/processed tTEM data.\n
    if the input is a string or pathlib.PurePath object, the function will read the tTEM data from the file and \
    return a dataframe.\n
    if the input is a dataframe, the function will reuse the dataframe and return a dataframe.\n
    if the input is a list, the function will read all the tTEM data from the files in the list and return a \
    dataframe.\n
    Version 11.18.2023 \n
    :param fname: A string or pathlib.PurePath object that contains the path to the tTEM .xyz file exported from Aarhus Workbench
    :param doi_path: A string or pathlib.PurePath object that contains the path to the DOI file exported from Aarhus Workbench
    :param layer_exclude: A list that contains the layer number that you want to exclude from the tTEM data
    :param line_exclude: A list that contains the line number that you want to exclude from the tTEM data
    :param point_exclude: A list that contains the point number that you want to exclude from the tTEM data
    :param resample: A int value that indicates whether to fill the tTEM data with a factor, defaults is False
    :return: A pandas dataframe that contains the filtered/processed tTEM data
    """
    def __init__(self,
                 fname: pathlib.PurePath |str |list,
                 doi_path: pathlib.PurePath| str| list = None,
                 layer_exclude: list = None,
                 line_exclude: list = None,
                 ID_exclude: list = None,
                 resample: int = None,
                 unit: str = 'meter'):
        if not isinstance(fname, list):
            fname = [fname]
        if not isinstance(doi_path, list) and doi_path:
            doi_path = [doi_path]
        if unit == 'meter':
            self.unit = 'meter'
            self.unitconvert = 1
        elif unit == 'feet':
            self.unit = 'feet'
            self.unitconvert = 3.28084
        self.fname = fname
        self.doi_path = doi_path
        self.layer_exclude = layer_exclude
        self.line_exclude = line_exclude
        self.ID_exclude = ID_exclude
        self.resample = resample
        self.data = self._format_ttem()
        self.crs = self.data.crs

    @staticmethod
    def _read_ttem(fname: pathlib.PurePath| str, mtoft=1) -> pd.DataFrame| dict:
        """
        This function read tTEM data from .xyz file, and return a formatted dataframe that contains all the tTEM data. \n
        Version 11.18.2023 \n
        :param fname: A string or pathlib.PurePath object that contains the path to the tTEM .xyz file exported from Aarhus Workbench
        :return: A pandas dataframe that contains all the tTEM data without any filtering
        """
        data = skip_metadata(fname, XYZ_FILE_PATTERN)
        df = pd.DataFrame(data[1::], columns=data[0])
        df = df.astype({'ID': 'int64',
                                'Line_No': 'int64',
                                'Layer_No': 'int64',
                                'UTMX': 'float64',
                                'UTMY': 'float64',
                                'Elevation_Cell': 'float64',
                                'Resistivity': 'float64',
                                'Resistivity_STD': 'float64',
                                'Conductivity': 'float64',
                                'Depth_top': 'float64',
                                'Depth_bottom': 'float64',
                                'Thickness': 'float64',
                                'Thickness_STD': 'float64'
                                })
        df = df[~(df['Thickness_STD'] == float(9999))]
        df['Elevation_Cell'] = df['Elevation_Cell']/mtoft
        df['Depth_top'] = df['Depth_top']/mtoft
        df['Depth_bottom'] = df['Depth_bottom']/mtoft
        df['Thickness'] = df['Thickness']/mtoft
        return df
    
    @staticmethod
    def _find_crs(fname: pathlib.PurePath| str) -> str:
        """
        This function is used to find the CRS of the tTEM data, it will return the CRS of the tTEM data. \n
         \n
        :param fname: A string or pathlib.PurePath object that contains the path to the tTEM .xyz file exported from Aarhus Workbench
        :return: CRS of the tTEM data
        """
        with open(fname, "r") as file:
            lines = file.readlines()
        pattern = re.compile(r"epsg:(\d+)", re.IGNORECASE)
        for line in lines:
            match = pattern.search(line)
            if match:
                crs = match.group()
                crs = crs.upper()
                break
            else:
                crs = None
        return crs


    @staticmethod
    def _DOI(dataframe: pd.DataFrame,
             doi_path: pathlib.PurePath| str |list,
             mtoft=1) -> pd.DataFrame:
        """
        Remove all tTEM data under DOI elevation limit with provided DOI file from Aarhus Workbench \n
        Version 11.18.2023 \n
        :param dataframe: Datafram that constains tTEM data
        :param doi_path: path-like contains DOI file, or a list of path that contains multiple DOI files
        :return: Filtered tTEM data above DOI
        """
        
        doi_concatlist = []
        match_index = []
        for i in doi_path:
            print('Applying DOI {}.....'.format(Path(i).name))
            data = skip_metadata(i, DOI_FILE_PATTERN)
            tmp_doi_df = pd.DataFrame(data[1::], columns=data[0])
            doi_concatlist.append(tmp_doi_df)
        df_DOI = pd.concat(doi_concatlist)
        df_DOI = df_DOI.astype({'UTMX': 'float64',
                                'UTMY': 'float64',
                                'Value': 'float64'
                                })
        df_DOI['Value'] = df_DOI['Value']/mtoft
        df_group = dataframe.groupby(['UTMX', 'UTMY'])
        ttem_concatlist = []
        for name, group in df_group:
            try:
                elevation = df_DOI.loc[(df_DOI['UTMX'] == name[0]) & (df_DOI['UTMY'] == name[1])]['Value'].values[0]
                new_group = group[group['Elevation_Cell'] >= elevation]
                
                ttem_concatlist.append(new_group)
            except IndexError:
                continue
        df_out = pd.concat(ttem_concatlist)
        return df_out

    @staticmethod
    def _layer_exclude(dataframe: pd.DataFrame,
                       layer_exclude: list) -> pd.DataFrame:
        df_out = dataframe[~np.isin(dataframe["Layer_No"], layer_exclude)]
        print('Exclude layer {}'.format(layer_exclude))
        return df_out

    @staticmethod
    def _line_exclude(dataframe: pd.DataFrame,
                      line_exclude: list) -> pd.DataFrame:
        df_out = dataframe[~np.isin(dataframe["Line_No"], line_exclude)]
        print('Exclude line {}'.format(line_exclude))
        return df_out

    @staticmethod
    def _ID_exclude(dataframe: pd.DataFrame,
                       ID_exclude: list) -> pd.DataFrame:
        df_out = dataframe[~dataframe["ID"].isin(ID_exclude)]
        [print('Exclude point {}'.format(x)) for x in ID_exclude]
        return df_out

    @staticmethod
    def _to_linear(group: pd.DataFrame,
                   factor: int) -> pd.DataFrame:
        """
        The core algorithm of the resample method, it fills the tTEM from log to linear.\n
        Version 11.18.2023\n
        :param group: tTEM dataframe, typically a groups from pd.groupby method
        :param factor: how thin your thickness should be divided, e.g. 10 means 1/10 m thickness
        :return: linear thickness tTEM dataframe
        """

        newgroup = group.loc[group.index.repeat(group.Thickness * factor)]
        mul_per_gr = newgroup.groupby('Elevation_Cell').cumcount()
        newgroup['Elevation_Cell'] = newgroup['Elevation_Cell'].subtract(mul_per_gr * 1 / factor)
        newgroup['Depth_top'] = newgroup['Depth_top'].add(mul_per_gr * 1 / factor)
        newgroup['Depth_bottom'] = newgroup['Depth_top'].add(1 / factor)
        newgroup['Elevation_End'] = newgroup['Elevation_Cell'].subtract(1 / factor)
        newgroup['Thickness'] = 1 / factor
        return newgroup

    @staticmethod
    def _resample(dataframe: pd.DataFrame,
                  factor: int) -> pd.DataFrame:
        """
        This staticmethod is connected with format_ttem method, it converts the tTEM thickness from log to linear \
        layers with a consistant layer thickness .\n
        Version 11.18.2023\n
        :param dataframe: Dataframe that contains the tTEM data
        :param factor: how thin your thickness should be divided, e.g. 10 means 1/10 m thickness
        :return: resampled dataframe
        """
        concatlist = []
        groups = dataframe.groupby(['UTMX', 'UTMY'])
        for name, group in groups:
            newgroup = ProcessTTEM._to_linear(group, factor)
            concatlist.append(newgroup)
        result = pd.concat(concatlist)
        result.reset_index(drop=True, inplace=True)
        return result
    
    @staticmethod
    def _get_crs(fname: str | pathlib.PurePath) -> str:
        try:
            crs = ProcessTTEM._find_crs(fname[0])
        except:
            print('No CRS found in the file, set CRS to None, use set_crs method to assign a CRS.')
            crs = None
        return crs

    def _format_ttem(self):
        """
        This is the core method of the class that read file under varies input circumstances, and return a \
        formatted dataframe that contains filtered tTEM data. \n
        Version: 11.18.2023\n
        :return: A pandas dataframe that contains filtered tTEM data
        """
    # Read data under different input circumstances
        from pathlib import Path
        crs = self._get_crs(self.fname)
        tmp_df = pd.DataFrame()
        if len(self.fname) == 0:
            raise ValueError("The input is empty!")
        if isinstance(self.fname[0], (str, pathlib.PurePath)):
            concatlist = []
            for i in self.fname:
                tmp_df = self._read_ttem(i, self.unitconvert)
                concatlist.append(tmp_df)
                print("Reading data from file {}...".format(Path(i).name))
            tmp_df = pd.concat(concatlist)
        elif isinstance(self.fname[0], pd.DataFrame):
            print("Reading data from cache...")
            tmp_df = pd.concat(self.fname)
        if tmp_df.empty:
            raise ValueError("The input is empty!")
    # Create filter parameters
        if self.layer_exclude is not None:
            tmp_df = self._layer_exclude(tmp_df, self.layer_exclude)
        if self.line_exclude is not None:
            tmp_df = self._layer_exclude(tmp_df, self.line_exclude)
        if self.ID_exclude is not None:
            tmp_df = self._ID_exclude(tmp_df, self.ID_exclude)
        if self.doi_path is not None:
            tmp_df = self._DOI(tmp_df, self.doi_path)
        if self.resample is not None:
            tmp_df = self._resample(tmp_df, self.resample)
    # Sort the dataframe
        tmp_df = tmp_df.sort_values(by=['ID', 'Line_No','Layer_No'])
        tmp_df.reset_index(drop=True, inplace=True)
        tmp_df["Elevation_End"] = tmp_df["Elevation_Cell"].subtract(tmp_df["Thickness"])
        self.data = tmp_df.copy()
        self.data.rename(columns={'UTMX': 'X', 'UTMY': 'Y'},inplace=True)
        if crs is not None:
            self.data = gpd.GeoDataFrame(self.data, 
                                        geometry=gpd.points_from_xy(self.data['X'], self.data['Y']),
                                        crs=crs)
        else: 
            self.data = gpd.GeoDataFrame(self.data, 
                                        geometry=gpd.points_from_xy(self.data['X'], self.data['Y']))
        return self.data

    

    def summary(self) -> gpd.GeoDataFrame:
        """
        This function generate a summary of the tTEM file which can be plot in the GIS contains all key information \
        about the tTEM
        :return: pd.DataFrame containing the summary of the tTEM info
        """

        id_group = self.data.groupby('ID')
        agg_group = id_group.agg({'Depth_bottom': 'max',
                                  'Elevation_Cell': 'max',
                                  'Elevation_End': 'min',
                                  'Resistivity': ['min', 'max', 'mean'],
                                  'X': 'mean', 'Y': 'mean'})
        agg_group.columns = agg_group.columns.map('_'.join)
        agg_group.index.name = None
        agg_group['ID'] = agg_group.index
        self.summary = agg_group
        self.summary.reset_index(drop=True, inplace=True)
        self.summary.rename(columns={'X_mean': 'X', 'Y_mean': 'Y'}, inplace=True)
        return self.summary
    
    def set_crs(self, new_crs: str):
        """
        Assigns a new coordinate reference system (CRS) to the object.

        Parameters:
            new_crs (str): The new CRS to be assigned. It should be in the format 'EPSG:<code>',
                           where <code> is the EPSG code of the CRS.

        Returns:
            str: The newly assigned CRS.

        Raises:
            ValueError: If the input CRS is not in the correct format.

        Example:
            >>> obj = ProcessTTEM(fname)
            >>> obj.assign_crs('EPSG:4326')
            The CRS is assigned to EPSG:4326
            'EPSG:4326'
        """
        pattern = r'^EPSG:\d+$'
        if bool(re.match(pattern, new_crs)):
            self.data.set_crs(new_crs, inplace=True)
            print('The CRS is assigned to {}'.format(new_crs))
        else: 
            raise ValueError("The input CRS is not valid, please use EPSG format, e.g. EPSG:4326")
        return self.crs
    
    def reproject(self, new_crs: str):
        """
        Reprojects the data to a new coordinate reference system (CRS).

        Parameters:
            new_crs (str): The new CRS to reproject the data to.

        Returns:
            GeoDataFrame: The reprojected data as a GeoDataFrame.
        """
        
        self.data = self.data.to_crs(new_crs)
        self.crs = self.data.crs
        self.data['X'] = self.data.geometry.x
        self.data['Y'] = self.data.geometry.y
        return self.data
        
        
    def to_shp(self, output_filepath: str | pathlib.PurePath):
        """
        This method converts the tTEM data to a shapefile or other supported geospatial formats.\n
        :param output_filepath: The path to save the output shapefile or geospatial file.
        
        """
        ttem_gdf = gpd.GeoDataFrame(self.summary, 
                                    geometry=gpd.points_from_xy(self.summary['X'], self.summary['Y']),
                                    crs=self.crs)
        if  Path(output_filepath).suffix.lower() == '.shp':
            ttem_gdf.to_file(output_filepath, driver='ESRI Shapefile')
            print('The output file is saved to {}'.format(Path(output_filepath).resolve()))
        elif Path(output_filepath).suffix.lower() == '.gpkg':
            ttem_gdf.to_file(output_filepath, driver='GPKG', layer=Path(self.fname[0]).stem)
            print('The output file is saved to {}'.format(Path(output_filepath).resolve()))
        elif Path(output_filepath).suffix.lower() == '.geojson':
            ttem_gdf.to_file(output_filepath, driver='GeoJSON')
            print('The output file is saved to {}'.format(Path(output_filepath).resolve()))
        else: 
            raise ValueError("The output file format is not supported, please use .shp, .gpkg, or .geojson")
            


if __name__ == "__main__":
    print('This is a module, please import it to use it.')
    import ttemtoolbox
    from pathlib import Path
    import geopandas as gpd
    from pathlib import Path
    workdir = Path.cwd()
    ttem_lslake = workdir.parent.parent.joinpath(r'data\PD22_I03_MOD.xyz')
    ttem_lsl = ttemtoolbox.process_ttem.ProcessTTEM(ttem_lslake)
