import pandas as pd
from pathlib import Path
import ttemtoolbox as tt
import plotly.graph_objects as go
import plotly.express as px

workdir = Path.cwd().joinpath(r'data')
welllog = workdir.joinpath('Well_log.xlsx')
tTEM_file_list = list(workdir.glob('*MOD.xyz'))
DOI_file_list = list(workdir.glob('DOI*.xyz'))
shape_path = workdir.joinpath(r'shapefiles')
shapefiles = list(shape_path.glob('*.shp'))
tTEM_lsl = tt.core.process_ttem.ProcessTTEM(tTEM_file_list[2])
tTEM_north = tt.core.process_ttem.ProcessTTEM(tTEM_file_list[0], DOI_file_list)
tTEM_center = tt.core.process_ttem.ProcessTTEM(tTEM_file_list[1], DOI_file_list)
tTEM_north_data = tTEM_north.data()
tTEM_center_data = tTEM_center.data()
tTEM_lsl_data = tTEM_lsl.data()

def amountbuy(targetprice, singleprice):
    amount = targetprice/singleprice
    return amount

def price_discount(online_price, steam_price, total_price):
    steam_income = steam_price*0.85
    amount = amountbuy(total_price, steam_income)
    steam_total = steam_income*amount
    online_total = amount*online_price
    chinese_price = steam_total*7.24
    discount = online_total/chinese_price
    print('Buy{}, discount{}'.format(amount,discount))