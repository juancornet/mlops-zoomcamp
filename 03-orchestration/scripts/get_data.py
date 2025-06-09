#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sys
import os

def read_dataframe(year = 2023, month = 3):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    #df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    print(f"Read {len(df)} rows from {url}")

    os.makedirs('data', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    df.to_parquet(f'data/raw/nyc_taxi_{year}_{month:02d}.parquet', engine='pyarrow', index=False)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    read_dataframe(year, month)
    