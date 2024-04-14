from glob import glob
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import * 
# import matplotlib.pyplot as plt
import requests
import urllib

files = glob('*.xyz')

# replace with bounds of study area
# xmin,xmax,ymin,ymax = 396849,448824,3906119,3974719

traces = list()
reduce = 1
layers = 18

geoprobe_locs = pd.read_csv('C:/Users/smithrg/OneDrive - Colostate/research/ttem/kansas/Voss_Workspace/Direct Push Data/rl01-locs.csv')
gdf = gpd.GeoDataFrame(geoprobe_locs,geometry=gpd.points_from_xy(geoprobe_locs['West'], geoprobe_locs['North']))
gdf = gdf.set_crs(epsg = '4326')
gdf_proj = gdf.to_crs(epsg = '32614')

for kk, file in enumerate(files):
    
    df = pd.read_csv(file,skiprows=24,delim_whitespace=True,header=None)
    df2 = pd.read_csv(file,skiprows=23,delim_whitespace=True)
    df.columns = df2.columns[1::]
    
    tops = [col for col in df.columns if ('TOP' in col and 'STD' not in col)]
    bottoms = [col for col in df.columns if ('BOT' in col and 'STD' not in col)]
    rho = [col for col in df.columns if ('RHO' in col and 'STD' not in col)]
    x,y = np.array(df['UTMX']), np.array(df['UTMY'])
    x,y = x[::reduce],y[::reduce]
    dist = np.sqrt(np.diff(x)**2+np.diff(y)**2)
    distfilt = np.where(dist>(np.median(dist)*3))
    
    thick = [col for col in df.columns if ('THK' in col and 'STD' not in col)]
    
    elevation = np.array(df['ELEVATION'])
    
    grid = np.zeros((layers,len(x)))
    xgrid = np.tile(x,(layers,1))
    ygrid = np.tile(y,(layers,1))
    zgrid = np.zeros(xgrid.shape)
    alpha = np.ones(grid.shape)
    
    print(file)
    print(np.median(dist))
    print(distfilt)
    
    if len(distfilt[0])>0:
    
        for tt, dfilt in enumerate(distfilt[0]):
    
            if tt==0:
                x2,y2 = x[0:(dfilt+1)],y[0:(dfilt+1)]
                xgrid = np.tile(x2,(layers,1))
                ygrid = np.tile(y2,(layers,1))
                zgrid = np.zeros(xgrid.shape)
                grid = np.zeros((layers,len(x2)))
                for zz, c in enumerate(tops):
                    if zz<layers:
                        z = elevation - (np.array((df[c] + df[bottoms[zz]])/2))
                        doi = np.array(df['DOI_CONSERVATIVE'])
                        filt = z<=doi
                        # alpha[zz,:] = 0+filt[::reduce]
                        log_rho = np.log10(df[rho[zz]])
                        # log_rho[filt==0] = 4
                        thick_val = np.array(df[thick[zz]])
                       
                        grid[zz,:] = log_rho[::reduce][0:(dfilt+1)]
                        zgrid[zz,:] = z[::reduce][0:(dfilt+1)]
                
            elif tt<len(distfilt):
                x2,y2 = x[(distfilt[0][tt-1]+1):(dfilt+1)],y[(distfilt[0][tt-1]+1):(dfilt+1)]
                xgrid = np.tile(x2,(layers,1))
                ygrid = np.tile(y2,(layers,1))
                zgrid = np.zeros(xgrid.shape)
                grid = np.zeros((layers,len(x2)))
                for zz, c in enumerate(tops):
                    if zz<layers:
                        z = elevation - (np.array((df[c] + df[bottoms[zz]])/2))
                        doi = np.array(df['DOI_CONSERVATIVE'])
                        filt = z<=doi
                        # alpha[zz,:] = 0+filt[::reduce]
                        log_rho = np.log10(df[rho[zz]])
                        # log_rho[filt==0] = 4
                        thick_val = np.array(df[thick[zz]])
                       
                        grid[zz,:] = log_rho[::reduce][(distfilt[0][tt-1]+1):(dfilt+1)]
                        zgrid[zz,:] = z[::reduce][(distfilt[0][tt-1]+1):(dfilt+1)]
                        
            else:
                x2,y2 = x[(dfilt+1)::],y[(dfilt+1)::]
                xgrid = np.tile(x2,(layers,1))
                ygrid = np.tile(y2,(layers,1))
                zgrid = np.zeros(xgrid.shape)
                grid = np.zeros((layers,len(x2)))
                for zz, c in enumerate(tops):
                    if zz<layers:
                        z = elevation - (np.array((df[c] + df[bottoms[zz]])/2))
                        doi = np.array(df['DOI_CONSERVATIVE'])
                        filt = z<=doi
                        # alpha[zz,:] = 0+filt[::reduce]
                        log_rho = np.log10(df[rho[zz]])
                        # log_rho[filt==0] = 4
                        thick_val = np.array(df[thick[zz]])
                       
                        grid[zz,:] = log_rho[::reduce][(dfilt+1)::]
                        zgrid[zz,:] = z[::reduce][(dfilt+1)::]
                        
        trace = go.Surface(x=xgrid, y=ygrid, z=zgrid,surfacecolor=grid,
                           showlegend = True if tt==0 else False,
                            legendgroup = file.split('_')[0],
                            name = file.split('_')[0],
                           # opacityscale = [[0,1],[0.8,1],[1,0]],
                            colorscale='Turbo',
                            colorbar={"title": 'log_10(resistivity, ohm m)'},
                           # colorscale=["blue",
                           #               "yellow", "red",
                           #               "purple","light gray"],
                           cmin=0.5,cmax=2.5)
        traces.append(trace)
    
        fig = go.Figure(data=[trace])
        fig.write_html('surfaces/'+file.split('_')[0]+'.html')
                        
    else:
        for zz, c in enumerate(tops):
            if zz<layers:
                z = elevation - (np.array((df[c] + df[bottoms[zz]])/2))
                doi = np.array(df['DOI_CONSERVATIVE'])
                filt = z<=doi
                # alpha[zz,:] = 0+filt[::reduce]
                log_rho = np.log10(df[rho[zz]])
                # log_rho[filt==0] = 4
                thick_val = np.array(df[thick[zz]])
               
                grid[zz,:] = log_rho[::reduce]
                zgrid[zz,:] = z[::reduce]
                
        trace = go.Surface(x=xgrid, y=ygrid, z=zgrid,surfacecolor=grid,
                           showlegend = True,
                            legendgroup = file.split('_')[0],
                            name = file.split('_')[0],
                           # opacityscale = [[0,1],[0.8,1],[1,0]],
                            colorscale='Turbo',
                            colorbar={"title": 'log_10(resistivity, ohm m)'},
                           # colorscale=["blue",
                           #               "yellow", "red",
                           #               "purple","light gray"],
                           cmin=0.5,cmax=2.5)
        traces.append(trace)
    
        fig = go.Figure(data=[trace])
        fig.write_html('surfaces/'+file.split('_')[0]+'.html')
        
def elevation_function(df, lat_column, lon_column):
    """Query service using lat, lon. add the elevation values as a new column."""
    url = r'https://nationalmap.gov/epqs/pqs.php?'
    elevations = []
    for lat, lon in zip(df[lat_column], df[lon_column]):

        # define rest query params
        params = {
            'output': 'json',
            'x': lon,
            'y': lat,
            'units': 'Meters'
        }

        # format query string and return query value
        result = requests.get((url + urllib.parse.urlencode(params)))
        elevations.append(result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])

    df['elev_meters'] = elevations
    return(df)

elev_df = elevation_function(gdf_proj,'North','West')   
well_traces = list()     
for kk in range(len(gdf_proj)):
    sheetname = gdf_proj.iloc[kk,0]
    # sheetname = 'RL01-'+sheet
    
    data = pd.read_excel('C:/Users/smithrg/OneDrive - Colostate/research/ttem/kansas/Voss_Workspace/Direct Push Data/RL01 EC Transect Data.xlsx', sheetname)
    data['rho (ohm m)'] = 1/(data['EC (mS/m)']/1000)
    
    x = gdf_proj.geometry.x[kk]*np.ones((len(data)))
    y = gdf_proj.geometry.y[kk]*np.ones((len(data)))
    z = np.array(elev_df['elev_meters'].iloc[[kk]])-data['Depth (m)']
    color = np.log10(data['rho (ohm m)'])
    color[np.isinf(color)]=2.5
    color[color>2.5]=2.5
    
    well_trace = go.Scatter3d(x=x,y=y,z=np.array(z),mode='markers',legendgroup = 'Direct Push',name=sheetname,
        marker=dict(
            size=2,
            color=np.array(color),                # set color to an array/list of desired values
            colorscale='Turbo', 
            cmin=0.5,
            cmax=2.5,# choose a colorscale,
            colorbar={"title": 'log_10(resistivity, ohm m)'}
        ))
    well_traces.append(well_trace)
    
fig2 = go.Figure(data=traces)
fig2.add_traces(well_traces)

fig2.update_layout(scene=dict(camera=dict(eye=dict(x=2, y=-0.2, z=0.5)), #the defaults values are 1.25, 1.25, 1.25
           aspectmode='manual', #this string can be 'data', 'cube', 'auto', 'manual'
           #a custom aspectratio is defined as follows:
           aspectratio=dict(x=1, y=3, z=.3)),
                  legend=dict(title=None, orientation='v', yanchor='top', y=1.1, xanchor='left', x=0.01),
                  hovermode = 'x')
    
fig2.write_html('surfaces.html')

fig3 = go.Figure(data=well_traces)

fig3.update_layout(scene=dict(camera=dict(eye=dict(x=2, y=-0.2, z=0.5)), #the defaults values are 1.25, 1.25, 1.25
           aspectmode='manual', #this string can be 'data', 'cube', 'auto', 'manual'
           #a custom aspectratio is defined as follows:
           aspectratio=dict(x=1, y=3, z=.3)),
                  legend=dict(title=None, orientation='v', yanchor='top', y=1.1, xanchor='left', x=0.01),
                  hovermode = 'x')
    
fig3.write_html('direct-push.html')

