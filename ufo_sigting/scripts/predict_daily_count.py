import pandas as pd
import numpy as np

from datetime import datetime, date, timedelta
from pickle import load

from tensorflow.python.keras import models


def gen_input(dataset, lag_max):
  X = []
  for i in range(len(dataset) - lag_max + 1):
    a = i + lag_max
    X.append(dataset[i:a, 0])
  return np.array(X)

def get_7daysbefore_data(df, target_date):
  df['date'] = pd.to_datetime(df['date'])
  td = timedelta(days=7)

  return df[(df['date'] >= np.datetime64(target_date - td)) & (df['date'] < np.datetime64(target_date))]


def predict_daily_count(target_date):
  target_date = datetime.strptime(target_date, '%Y-%m-%d')
  target_date = date(target_date.year, target_date.month, target_date.day)

  database_path = '../datasets/scrubbed_only_us_dayly_count.csv'
  df = pd.read_csv(database_path)

  df_7days = get_7daysbefore_data(df=df, target_date=target_date)

  input_data_0 = df_7days.loc[:, '0'].values.astype('float32')
  input_data = np.reshape(input_data_0, (-1, 1))

  scaler_X = load(open('../scalers/scaler_X.pkl', "rb"))

  X = gen_input(input_data, 7)
  X_scaled = scaler_X.fit_transform(X)

  input = np.reshape(X_scaled, (X_scaled.shape[0],1,X_scaled.shape[1]))

  model_path = '../models/only_us_daily_count.h5'
  model = models.load_model(model_path)

  pred = model.predict(input)
  scaler_y = load(open('../scalers/scaler_y.pkl', "rb"))
  pred = scaler_y.inverse_transform(pred)

  return int(pred[0])
  

def write_result(target_date, result):
  database_path = '../datasets/scrubbed_only_us_dayly_count.csv'
  df = pd.read_csv(database_path, index_col=0)
  df.loc[target_date] = result
  df.to_csv(database_path)


def predict(target_date):
  result = predict_daily_count(target_date)
  write_result(target_date, result)
  return result


if __name__ == '__main__':
  target_date = '2014-05-15'
  result = predict(target_date)
  print(result)
