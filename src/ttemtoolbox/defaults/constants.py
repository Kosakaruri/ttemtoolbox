#!/usr/bin/env python
# constants.py
# Version: 11.17.2023
# Created: 2023-11-17
# Author: Jiawei Li

XYZ_FILE_PATTERN = 'ID'
EPSG_FILE_PATTERN = 'EPSG'
DOI_FILE_PATTERN = 'UTMX'
CSV_EXTENSION = ('.csv',)
EXCEL_EXTENSION = ('.xlsx', '.xls', '.xlsm')
LITHOLOGY_SHEET_NAMES = ('lithology','litho')
LITHOLOGY_COLUMN_NAMES_KEYWORD = ('lithology','litho', 'keyword')
LITHOLOGY_COLUMN_NAMES_BORE = ('bore','borehole')
LITHOLOGY_COLUMN_NAMES_DEPTH_TOP = ( 'depth1', 'depth_1', 'depthtop','depth_top')
LITHOLOGY_COLUMN_NAMES_DEPTH_BOTTOM = ( 'depth2',  'depth_2', 'depthbottom', 'depth_bottom')
LOCATION_SHEET_NAMES = ('location', 'coordinates')
LOCATION_COLUMN_NAMES_UTMX = ('utmx', 'utm_x')
LOCATION_COLUMN_NAMES_UTMY = ('utmy', 'utm_y')
LOCATION_COLUMN_NAMES_LAT = ( 'lat',  'latitude', 'y')
LOCATION_COLUMN_NAMES_LON = ('lon',  'longitude', 'x')
LOCATION_COLUMN_NAMES_ELEVATION = ( 'elevation', 'elev',  'elevation_m',  'elev_m', 'elevation(m)', 'elev(m)','z')

