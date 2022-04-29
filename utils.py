import glob
import torch
import numpy as np
import pandas as pd

from datetime import datetime

import plotly
import plotly.graph_objects as go

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm

from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler

from pykrige.uk import UniversalKriging

from constants import SIZE_X
from constants import SIZE_Y
from constants import KM_LON
from constants import KM_LAT
from constants import SEED
from constants import MAX_LAT_AT
from constants import MIN_LAT_AT
from constants import MAX_LON_AT
from constants import MIN_LON_AT
from constants import MAX_LAT_CH
from constants import MIN_LAT_CH
from constants import MAX_LON_CH
from constants import MIN_LON_CH

from constants import EPOCHS
from constants import BATCH_SIZE

from constants import PATH_WEATHER_DATA_AT
from constants import PATH_COW_DATA
from constants import LOCATION_MAPPING_AT

N_L1 = 128
N_L2 = 32
N_L3 = 16
N_L4 = 8


def tic():
    import time
    global start_time_for_tictoc
    start_time_for_tictoc = time.time()


def toc():
    import time
    if 'start_time_for_tictoc' in globals():
        print(f'Elapsed time is {(time.time() - start_time_for_tictoc)} seconds.')
    else:
        print('toc: start time not set')


def relative_difference(predicted, map_value):
    array = 2 / (1 + np.exp(abs(predicted - map_value)))
    return array


def l2norm(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def l2norm_km(a, b, country, size_x=SIZE_X, size_y=SIZE_Y, km_lon=KM_LON, km_lat=KM_LAT):
    if country == 'A':
        max_lat = MAX_LAT_AT
        min_lat = MIN_LAT_AT
        max_lon = MAX_LON_AT
        min_lon = MIN_LON_AT
    elif country == 'S':
        max_lat = MAX_LAT_CH
        min_lat = MIN_LAT_CH
        max_lon = MAX_LON_CH
        min_lon = MIN_LON_CH

    a = np.array(a)
    b = np.array(b)
    scale_x = km_lon / size_x * (max_lon - min_lon)
    scale_y = km_lat / size_y * (max_lat - min_lat)

    return np.sqrt((a[0] * scale_x - b[0] * scale_x) ** 2 + (a[1] * scale_y - b[1] * scale_y) ** 2)


def print_metrics(df_val, split_key, reg, sensor_cols, target, kriged_maps, scaler, country):
    distance = []

    mse = []
    mae = []
    r2 = []

    for key in df_val[split_key].unique():
        tmp_df = df_val.loc[df_val[split_key] == key]

        w_prime = reg.predict(tmp_df[sensor_cols])
        x, y = bayes_filter(w_prime, kriged_maps[target[0]], 0, kriged_maps[target[0]].shape[0])
        real_x = int(tmp_df['real_x'].unique().item())
        real_y = int(tmp_df['real_y'].unique().item())

        distance.append(l2norm_km((real_x, real_y), (x, y), country))

        real_scaled = scaler.inverse_transform(tmp_df[target].to_numpy(), target)
        w_prime_scaled = scaler.inverse_transform(w_prime, target)

        mse.append(mean_squared_error(real_scaled, w_prime_scaled))
        mae.append(mean_absolute_error(real_scaled, w_prime_scaled))
        r2.append(r2_score(real_scaled, w_prime_scaled))

    print('R2 :', np.mean(r2))
    print('MAE:', np.mean(mae))
    print('MSE:', np.mean(mse))
    print('Average Error:', np.mean(np.array(distance)))
    print('Median Error:', np.median(np.array(distance)))
    print('')
    
    return distance


class Scaler:
    def __init__(self, **inputs):
        self.scalers = OrderedDict()
        for k, v in inputs.items():
            self.add(k, v)

    def add(self, key, array):
        self.scalers[key] = MinMaxScaler()
        self.scalers[key].fit(array.reshape(-1, 1))

    def transform(self, array, cols, inverse=False):
        array = array.copy()
        if not len(array.shape) > 1:
            array = array.reshape(-1, 1)

        step_size = len(cols)
        for i, k in enumerate(cols):
            for j in range(i, array.shape[1], step_size):
                if not inverse:
                    array[:, j] = self.scalers[k].transform(array[:, j].reshape(-1, 1)).squeeze()
                else:
                    array[:, j] = self.scalers[k].inverse_transform(array[:, j].reshape(-1, 1)).squeeze()
        return array

    def inverse_transform(self, array, cols):
        return self.transform(array, cols, True)


def bayes_filter(sensor_values, temperature_maps, start, duration, **kwargs):
    prior = np.zeros((temperature_maps.shape[1], temperature_maps.shape[2]))
    prior[:, :] = 1 / temperature_maps[0, :, :].size

    assert len(sensor_values) == duration, 'sensor_value and duration must be equal'

    posteriors = []
    for tick in range(duration):
        corr = relative_difference(sensor_values[tick], temperature_maps[tick + start, :, :])
        tmp = np.multiply(corr, prior)
        prior = tmp / np.sum(tmp)
        posteriors.append(prior)

    x, y = np.unravel_index(prior.argmax(), prior.shape)

    if 'posteriors' in kwargs:
        return x, y, posteriors
    else:
        return x, y


def set_seeds(seed=SEED):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)
    
    import torch
    torch.manual_seed(seed)


def join_AUT_data(path_weather, path_animals):
    df_weather = pd.DataFrame()
    for csv in glob.glob(path_weather):
        with open(csv, 'r') as f:
            lines = f.readlines()
            lat = float(lines[3].strip().split(';')[1])
            lon = float(lines[4].strip().split(';')[1])
            plz = csv.split('/')[-1].split('_')[0]

        tmp_df = pd.read_csv(csv, header=22, sep=';')
        tmp_df['postal_code'] = int(plz)
        tmp_df['lat'] = lat
        tmp_df['lon'] = lon
        tmp_df['datetime'] = pd.date_range(start='2019-01-01 01:00', end='2020-01-01', freq='1H')
        tmp_df = tmp_df.drop(columns=['# Date', 'UT time'])
        df_weather = df_weather.append(tmp_df)

    csv_files = sorted(glob.glob(path_animals))
    meta_fn = csv_files.pop(-1)
    meta_df = pd.read_csv(meta_fn)
    df = pd.DataFrame()

    for idx, csv in enumerate(csv_files):
        if csv == meta_fn:
            continue
        else:
            tmp_df = pd.read_csv(csv, delimiter=',')
            animal_id = csv.split('/')[-1].split('.')[0]
            tmp_df['animal_id'] = animal_id
            plz = meta_df['postal_code'].loc[meta_df['animal_id'] == animal_id].squeeze()
            tmp_df['postal_code'] = plz

            tmp_df['datetime'] = pd.to_datetime(tmp_df['datetime'], format='%Y-%m-%dT%H:%M:%S%z')
            tmp_df['datetime'] = tmp_df['datetime'].apply(lambda x: pd.to_datetime(x).tz_convert(None))
            tmp_df = tmp_df.set_index('datetime')

            tmp_weather = df_weather.loc[df_weather['postal_code'] == plz]
            tmp_weather = tmp_weather.drop(columns=['postal_code'])
            tmp_weather = tmp_weather.set_index('datetime')
            tmp_df = pd.merge(tmp_df, tmp_weather, how='outer', left_index=True, right_index=True)
            for c in df_weather.columns.drop('datetime'):
                if pd.api.types.is_numeric_dtype(tmp_df[c]):
                    tmp_df[c] = tmp_df[c].interpolate(method='pad')

            tmp_df = tmp_df.reset_index()
            tmp_df = tmp_df.dropna()
            tmp_df['YTD'] = tmp_df['datetime'].apply(lambda x: (datetime(x.year, x.month, x.day) - datetime(x.year, 1, 1)).days)
            tmp_df['YTD'] = tmp_df['YTD'].astype(int)
            df = df.append(tmp_df)

            print('Joining Data: {:3d}/{:3d}'.format(idx+1, len(csv_files)))

    return df


def preprocess_AUT_data(split_key, sensor_cols, target, aux_cols):
    df = join_AUT_data(PATH_WEATHER_DATA_AT, PATH_COW_DATA)
    df = df.groupby([split_key, pd.Grouper(key='datetime', freq='24h')]).mean()
    df = df.reset_index()

    df, scaler = scale_data(df, sensor_cols + target)

    # drop animals / farms who have missing values
    max_rows = df.groupby(split_key).count().max()[0]
    for unique in df[split_key].unique():
        if df.loc[df[split_key] == unique].shape[0] != max_rows:
            df = df.loc[~(df[split_key] == unique)]

    kriged_maps = {}
    for key in sensor_cols + target:
        if key == 'YTD':
            continue

        kriged_maps[key] = kriging(df, key, LOCATION_MAPPING_AT, MIN_LON_AT, MAX_LON_AT, MIN_LAT_AT, MAX_LAT_AT)
        grid_x = np.arange(MIN_LON_AT, MAX_LON_AT, (MAX_LON_AT - MIN_LON_AT) / SIZE_X)
        grid_y = np.arange(MIN_LAT_AT, MAX_LAT_AT, (MAX_LAT_AT - MIN_LAT_AT) / SIZE_Y)

        for unique in df[split_key].unique():
            lon = df.loc[df[split_key] == unique]['lon'].unique()[0]
            lat = df.loc[df[split_key] == unique]['lat'].unique()[0]
            real_x = np.abs(grid_x - lon).argmin()
            real_y = np.abs(grid_y - lat).argmin()
            df.loc[df[split_key] == unique, key] = kriged_maps[key][:, real_x, real_y]
            df.loc[df[split_key] == unique, 'real_x'] = real_x
            df.loc[df[split_key] == unique, 'real_y'] = real_y

    for key in sensor_cols + target:
        df[key] = scaler.inverse_transform(df[key].to_numpy(dtype=float), [key])

    df = df[aux_cols + sensor_cols + target + ['lat', 'lon', 'real_x', 'real_y']]
    df['YTD'] = df['YTD'].astype(int)
    return df, kriged_maps


def preprocess_CH_data(path):
    df = pd.DataFrame()
    csv_files = glob.glob(path)

    for file in csv_files:
        with open(file, 'r', encoding='ISO-8859-1') as f:
            station = f.readlines()[0].split('Station: ')[1].strip()

        tmp_df = pd.read_csv(file, encoding='ISO-8859-1', delimiter=';', header=1, low_memory=False, skiprows=3)
        tmp_df['datetime'] = pd.to_datetime(tmp_df['Datum/Zeit'].values, format='%d.%m.%Y')
        tmp_df['YTD'] = tmp_df['datetime'].apply(lambda x: (datetime(x.year, x.month, x.day) - datetime(x.year, 1, 1)).days)
        tmp_df['station'] = station
        tmp_df = tmp_df.interpolate()

        df = pd.concat([df, tmp_df], sort=False)

    for c in df.columns:
        df = df.rename({c: c.split(' ')[0]}, axis=1)

    return df


def scale_data(df, columns):
    scaler = Scaler()
    for k in set(columns):
        scaler.add(k, df[k].to_numpy())
        df['ORIGINAL_' + k] = df[k]
        df[k] = scaler.transform(df[k].to_numpy(dtype=float), [k])
    return df, scaler


def kriging(df, field, location_mapping, min_lon, max_lon, min_lat, max_lat, size_x=SIZE_X, size_y=SIZE_Y):
    mapping = location_mapping

    interpolated_map = []
    grid_x = np.arange(min_lon, max_lon, (max_lon - min_lon) / size_x)
    grid_y = np.arange(min_lat, max_lat, (max_lat - min_lat) / size_y)

    high = df[field].max()
    low = df[field].min()
    v_params = {'sill': high, 'range': high - low, 'nugget': low, 'slope': high - low}

    old_month = 0

    groups = df.groupby('datetime')
    for idx in groups.indices:
        if idx.month > old_month:
            old_month = idx.month
            print(f'Kriging {field}: {idx}')

        tmp_df = df.loc[df['datetime'] == idx]
        tmp_df = tmp_df.groupby('postal_code').mean()
        tmp_df = tmp_df.reset_index()

        points_x = [mapping['lon'].loc[mapping['zip'] == station].to_numpy().item() for station in tmp_df['postal_code'].to_list()]
        points_y = [mapping['lat'].loc[mapping['zip'] == station].to_numpy().item() for station in tmp_df['postal_code'].to_list()]
        real_values = tmp_df[field].to_numpy().squeeze()

        uk = UniversalKriging(points_x, points_y, real_values, variogram_model='linear', variogram_parameters=v_params)
        z, ss = uk.execute('grid', grid_x, grid_y)
        interpolated_map.append(z.transpose())

    interpolated_map = np.array(interpolated_map)
    return interpolated_map


def kriging_ch(df, key, size_x=SIZE_X, size_y=SIZE_Y):
    from constants import LOCATION_MAPPING_CH
    mapping = LOCATION_MAPPING_CH
    station_names = df['station'].unique()

    interpolated_map = []
    grid_x = np.arange(MIN_LON_CH, MAX_LON_CH, (MAX_LON_CH - MIN_LON_CH) / size_x)
    grid_y = np.arange(MIN_LAT_CH, MAX_LAT_CH, (MAX_LAT_CH - MIN_LAT_CH) / size_y)

    groups = df.groupby('datetime')

    for tick, idx in enumerate(groups.indices):
        tmp_df = df.loc[df['datetime'] == idx]
        tmp_df = tmp_df.sort_values('station')

        points_x = [mapping['lon'].loc[mapping['station'] == station].to_numpy().item() for station in station_names]
        points_y = [mapping['lat'].loc[mapping['station'] == station].to_numpy().item() for station in station_names]
        real_values = [tmp_df[key].loc[tmp_df['station'] == station].to_numpy().item() for station in station_names]

        uk = UniversalKriging(points_x, points_y, real_values, variogram_model='linear')
        z, ss = uk.execute('grid', grid_x, grid_y)
        interpolated_map.append(z.transpose())

    interpolated_map = np.array(interpolated_map)
    return interpolated_map


class Encoder(torch.nn.Module):
    def __init__(self, n_features_in):
        super(Encoder, self).__init__()

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(n_features_in, N_L1),
            torch.nn.BatchNorm1d(N_L1),
            torch.nn.ReLU()
        )
        self.encoder_1 = torch.nn.Sequential(
            torch.nn.Linear(N_L1, N_L2),
            torch.nn.BatchNorm1d(N_L2),
            torch.nn.ReLU()
        )
        self.encoder_2 = torch.nn.Sequential(
            torch.nn.Linear(N_L2, N_L3),
            torch.nn.BatchNorm1d(N_L3),
            torch.nn.ReLU()
        )
        self.encoder_3 = torch.nn.Sequential(
            torch.nn.Linear(N_L3, N_L4),
            torch.nn.BatchNorm1d(N_L4),
            torch.nn.ReLU()
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.encoder_1(out)
        out = self.encoder_2(out)
        out = self.encoder_3(out)
        return out


class LatentSpace(torch.nn.Module):
    def __init__(self):
        super(LatentSpace, self).__init__()

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(N_L4, N_L4),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(N_L4)
        )

    def forward(self, X):
        out = self.latent(X)
        return out


class Decoder(torch.nn.Module):
    def __init__(self, n_features_in):
        super(Decoder, self).__init__()

        self.decoder_1 = torch.nn.Sequential(
            torch.nn.Linear(N_L4, N_L3),
            torch.nn.BatchNorm1d(N_L3),
            torch.nn.ReLU()
        )

        self.decoder_2 = torch.nn.Sequential(
            torch.nn.Linear(N_L3, N_L2),
            torch.nn.BatchNorm1d(N_L2),
            torch.nn.ReLU()
        )

        self.decoder_3 = torch.nn.Sequential(
            torch.nn.Linear(N_L2, N_L1),
            torch.nn.BatchNorm1d(N_L1),
            torch.nn.ReLU()
        )

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(N_L1, n_features_in),
            torch.nn.ReLU()
        )

    def forward(self, X):
        out = self.decoder_1(X)
        out = self.decoder_2(out)
        out = self.decoder_3(out)
        out = self.output_layer(out)
        return out


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)


def train_teleport(train_A, train_B, val_A, val_B, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=0.001, beta=0.1):
    data_loader_A_train = torch.utils.data.DataLoader(Dataset(train_A), batch_size=batch_size, shuffle=True, drop_last=True)
    data_loader_A_val = torch.utils.data.DataLoader(Dataset(val_A), batch_size=batch_size, shuffle=True, drop_last=True)
    data_loader_B_train = torch.utils.data.DataLoader(Dataset(train_B), batch_size=batch_size, shuffle=True, drop_last=True)
    data_loader_B_val = torch.utils.data.DataLoader(Dataset(val_B), batch_size=batch_size, shuffle=True, drop_last=True)

    n_features = train_A.shape[1]
    enc_a = Encoder(n_features)
    dec_a = Decoder(n_features)
    lat = LatentSpace()
    enc_b = Encoder(n_features)
    dec_b = Decoder(n_features)

    loss_fn = torch.nn.MSELoss()
    params = list(enc_a.parameters()) + list(enc_b.parameters()) + list(lat.parameters()) + list(dec_a.parameters()) + list(dec_b.parameters())
    optimiser = torch.optim.Adam(params, lr=lr)

    losses_train_A = []
    losses_val_A = []
    losses_train_B = []
    losses_val_B = []

    for epoch in tqdm(range(epochs)):
        optimiser.zero_grad()

        loss_train_A = 0
        loss_train_B = 0

        for x in data_loader_A_train:
            x_prime = dec_a(lat(enc_a(x)))
            x_prime_2 = dec_b(lat(enc_a(x)))
            loss_train_A += loss_fn(x, x_prime) + (loss_fn(x, x_prime_2) * beta)

        for x in data_loader_B_train:
            x_prime = dec_b(lat(enc_b(x)))
            x_prime_2 = dec_a(lat(enc_b(x)))
            loss_train_B += loss_fn(x, x_prime) + (loss_fn(x, x_prime_2) * beta)

        loss_train = loss_train_A + loss_train_B
        loss_train.backward()
        optimiser.step()

        loss_val_A = 0
        loss_val_B = 0

        with torch.no_grad():
            for x in data_loader_A_val:
                x_prime_val = dec_a(lat(enc_a(x)))
                loss_val_A += loss_fn(x, x_prime_val)

            for x in data_loader_B_val:
                x_prime_val = dec_b(lat(enc_b(x)))
                loss_val_B += loss_fn(x, x_prime_val)

        losses_train_A.append(loss_train_A.item())
        losses_val_A.append(loss_val_A.item())
        losses_train_B.append(loss_train_B.item())
        losses_val_B.append(loss_val_B.item())

    models = {
        'enc_a': enc_a,
        'enc_b': enc_b,
        'lat': lat,
        'dec_a': dec_a,
        'dec_b': dec_b
    }
    return models


def test_models(split_key, df_val_west, df_val_east, sensor_cols, model_H, models_teleport, k_map, scaler, country):
    dec_a = models_teleport['dec_a']
    dec_b = models_teleport['dec_b']
    lat = models_teleport['lat']
    enc_a = models_teleport['enc_a']
    enc_b = models_teleport['enc_b']

    self_a = []
    self_b = []
    tele_a = []
    tele_b = []
    orig_a = []
    orig_b = []

    mae_tele_a = []
    mae_tele_b = []

    mae_ae = []

    df_pos = pd.DataFrame()

    for split in df_val_west[split_key].unique():
        tmp_df = df_val_west.loc[df_val_west[split_key] == split]
        S = torch.Tensor(tmp_df[sensor_cols].to_numpy())

        real_x = int(tmp_df['real_x'].iloc[0])
        real_y = int(tmp_df['real_y'].iloc[0])

        duration = len(tmp_df)

        with torch.no_grad():
            s_self_a = dec_a(lat(enc_a(S)))
            s_self_a = s_self_a.numpy()
            s_self_a[:, -1] = tmp_df['YTD']
            w_self_a = model_H.predict(s_self_a)
            self_x, self_y = bayes_filter(w_self_a, k_map, 0, duration)
            dist_self = l2norm_km((real_x, real_y), (self_x, self_y), country)
            self_a.append(dist_self)

            s_tele_a = dec_a(lat(enc_b(S)))
            s_tele_a = s_tele_a.numpy()
            s_tele_a[:, -1] = tmp_df['YTD']
            w_tele_a = model_H.predict(s_tele_a)
            tele_x, tele_y = bayes_filter(w_tele_a, k_map, 0, duration)
            dist_tele = l2norm_km((real_x, real_y), (tele_x, tele_y), country)
            tele_a.append(dist_tele)

            mae_ae.append(mean_absolute_error(tmp_df[sensor_cols[0]].to_numpy(), s_tele_a[:, 0] ))

        w_prime_o = model_H.predict(tmp_df[sensor_cols].to_numpy())
        x_o, y_o = bayes_filter(w_prime_o, k_map, 0, duration)
        dist_o = l2norm_km((real_x, real_y), (x_o, y_o), country)
        orig_a.append(dist_o)

        # RAE
        mae_tele_a.append(mean_absolute_error(tmp_df[sensor_cols].to_numpy()[:, 0], s_tele_a[:, 0]))

        res = dict(region='A', real_x=real_x, real_y=real_y, self_x=self_x, self_y=self_y, dist_self=dist_self,
                   tele_x=tele_x, tele_y=tele_y, dist_tele=dist_tele)
        df_pos = pd.concat([df_pos, pd.DataFrame(res, index=[0])])

    for split in df_val_east[split_key].unique():
        tmp_df = df_val_east.loc[df_val_east[split_key] == split]
        S = torch.Tensor(tmp_df[sensor_cols].to_numpy())

        real_x = int(tmp_df['real_x'].iloc[0])
        real_y = int(tmp_df['real_y'].iloc[0])

        with torch.no_grad():
            s_self_b = dec_b(lat(enc_b(S)))
            s_self_b = s_self_b.numpy()
            s_self_b[:, -1] = tmp_df['YTD']
            w_self_b = model_H.predict(s_self_b)
            self_x, self_y = bayes_filter(w_self_b, k_map, 0, duration)
            dist_self = l2norm_km((real_x, real_y), (self_x, self_y), country)
            self_b.append(dist_self)

            s_tele_b = dec_b(lat(enc_a(S)))
            s_tele_b = s_tele_b.numpy()
            s_tele_b[:, -1] = tmp_df['YTD']
            w_tele_b = model_H.predict(s_tele_b)
            tele_x, tele_y = bayes_filter(w_tele_b, k_map, 0, duration)
            dist_tele = l2norm_km((real_x, real_y), (tele_x, tele_y), country)
            tele_b.append(dist_tele)

            mae_ae.append(mean_absolute_error(tmp_df[sensor_cols[0]].to_numpy(), s_tele_b[:, 0]))

        w_prime_o = model_H.predict(tmp_df[sensor_cols].to_numpy())
        x_o, y_o = bayes_filter(w_prime_o, k_map, 0, duration)
        dist_o = l2norm_km((real_x, real_y), (x_o, y_o), country)
        orig_b.append(dist_o)

        # MAE
        mae_tele_b.append(mean_absolute_error(tmp_df[sensor_cols].to_numpy()[:, 0], s_tele_b[:, 0]))

        real_values_scaled = scaler.inverse_transform(tmp_df[sensor_cols].to_numpy(), sensor_cols)[:, 0]
        s_tele_a_scaled = scaler.inverse_transform(s_tele_a, sensor_cols)[:, 0]

        res = dict(region='B', real_x=real_x, real_y=real_y, self_x=self_x, self_y=self_y, dist_self=dist_self,
                   tele_x=tele_x, tele_y=tele_y, dist_tele=dist_tele)
        df_pos = pd.concat([df_pos, pd.DataFrame(res, index=[0])])

    print('Orig West', np.array(orig_a).mean())
    print('Self West', np.array(self_a).mean())
    print('Tele West', np.array(tele_a).mean())
    print('Orig East', np.array(orig_b).mean())
    print('Self East', np.array(self_b).mean())
    print('Tele East', np.array(tele_b).mean())

    return orig_a, tele_a, orig_b, tele_b, mae_tele_a, mae_tele_b, mae_ae
