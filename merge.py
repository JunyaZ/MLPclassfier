import os
import pandas as pd
import glob
import csv

df = pd.read_csv('C:\data\SNP1.csv')
# creates a list of all csv files
globbed_files = glob.glob("*.csv")
for csv in globbed_files:
    frame = pd.read_csv(csv)
    frame = frame.rename(columns={'0': "allele1", '1': "allele2"})
    result = pd.concat([df, frame], axis=1)
    del result['Unnamed: 0']
    result.to_csv(csv)
