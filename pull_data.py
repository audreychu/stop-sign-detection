from grab_data import *
import pandas as pd
import sys 


def main():
    start = int(sys.argv[1])
    stop = int(sys.argv[2])
    csv = pd.read_csv('./SF_location.csv')
    geom = csv['Geom']
    for g in geom.iloc[start:stop]:
        GetStreet(g, 90, "test")

if __name__ == "__main__":
    main()
