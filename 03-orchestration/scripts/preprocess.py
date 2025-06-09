#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sys
import os

def preprocess(year = 2023, month = 3):

    df =pd.read_parquet(f'data/raw/nyc_taxi_{year}_{month:02d}.parquet')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    print(f"Preprocessed {len(df)} rows")
    
    os.makedirs('data/preprocessed', exist_ok=True)
    df.to_parquet(f'data/preprocessed/nyc_taxi_{year}_{month:02d}_preprocessed.parquet', engine='pyarrow', index=False)

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    preprocess(year, month)