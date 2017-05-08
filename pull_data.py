from grab_data import *
import pandas as pd


csv = pd.read_csv('./SF_location.csv')
geom = csv['Geom']


for g in geom:
    GetStreet(g, 90, "test")


