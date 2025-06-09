#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Get dataset about nyc taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    df = read_dataframe(args.year, args.month)
    os.makedirs('data', exist_ok=True)
    df.to_parquet(f'data/nyc_taxi_{args.year}_{args.month:02d}.parquet', engine='pyarrow', index=False)