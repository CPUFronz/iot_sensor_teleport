import os
import pandas as pd

SEED = 42
SIZE_X = 200
SIZE_Y = 100

PATH_COW_DATA = os.path.abspath('data/cows/*.csv')
PATH_WEATHER_DATA_AT = os.path.abspath('data/weather_from_postal_code/*.csv')

KM_LAT = 111.133
KM_LON = 78.84
MIN_LAT_CH = 46
MAX_LAT_CH = 48
MIN_LON_CH = 6
MAX_LON_CH = 10
MIN_LAT_AT = 46
MAX_LAT_AT = 49
MIN_LON_AT = 9
MAX_LON_AT = 17

BATCH_SIZE = 256
EPOCHS = 5000
NUM_ESTIMATORS = 5000

LOCATION_MAPPING_CH = pd.DataFrame(columns=['station', 'lat', 'lon', 'location'], data=[
    ['Magadino-Cadenazzo', 46.147522, 8.882279, 'Land, < 1000 m'],
    ['Davos-Seehornwald', 46.815624, 9.867038, 'Land, > 1000 m'],
    ['Rigi-Seebodenalp', 47.069156, 8.473862, 'Land, > 1000 m'],
    ['Chaumont', 47.026979, 6.958040, 'Land, > 1000 m'],
    ['Lausanne-César-Roux', 46.522948, 6.637254, 'Stadt, Verkehr'],
    ['Basel-Binningen', 47.537029, 7.565688, 'Vorstädtisch'],
    ['Lugano-Università', 46.011069, 8.958990, 'Stadt'],
    ['Payerne', 46.821769, 6.936284, 'Land, < 1000 m'],
    ['Beromünster', 47.206254, 8.190420, 'Land, < 1000 m'],
    ['Dübendorf-Empa', 47.402167, 8.612170, 'Vorstädtisch'],
    ['Tänikon', 47.480510, 8.907059, 'Land, < 1000 m'],
    ['Bern-Bollwerk', 46.950556, 7.440643, 'Stadt, Verkehr'],
    ['Zürich-Kaserne', 47.376905, 8.532089, 'Stadt'],
    ['Härkingen-A1', 47.309644, 7.833019, 'Land, Autobahn'],
    ['Sion-Aéroport-A9', 46.219452, 7.340655, 'Land, Autobahn'],
    ['Jungfraujoch', 46.545132, 7.971331, 'Hochgebirge']
])

LOCATION_MAPPING_AT = pd.DataFrame(columns=['zip', 'lat', 'lon', 'province'], data=[
    [1210, 48.283333, 16.412222, 'Vienna'],
    [2381, 48.15, 16.166667, 'Lower Austria'],
    [2831, 47.660278, 16.158333, 'Lower Austria'],
    [2852, 47.45, 16.2, 'Lower Austria'],
    [2860, 47.5, 16.283333, 'Lower Austria'],
    [3143, 48.158889, 15.687222, 'Lower Austria'],
    [3300, 48.123, 14.87213, 'Lower Austria'],
    [3340, 47.966667, 14.766667, 'Lower Austria'],
    [3610, 48.4, 15.466667, 'Lower Austria'],
    [3684, 48.266667, 15.033333, 'Lower Austria'],
    [3720, 48.55, 15.85, 'Lower Austria'],
    [3911, 48.516667, 15.066667, 'Lower Austria'],
    [3920, 48.566667, 14.95, 'Lower Austria'],
    [4091, 48.533333, 13.65, 'Upper Austria'],
    [4131, 48.445, 13.936111, 'Upper Austria'],
    [4133, 48.465833, 13.881944, 'Upper Austria'],
    [4191, 48.552778, 14.22, 'Upper Austria'],
    [4274, 48.394444, 14, 'Upper Austria'],
    [4283, 48.35, 14.666667, 'Upper Austria'],
    [4451, 48.021667, 14.408889, 'Upper Austria'],
    [4483, 48.15, 14.425556, 'Upper Austria'],
    [4582, 47.665278, 14.340833, 'Upper Austria'],
    [4591, 47.883611, 14.258889, 'Upper Austria'],
    [4680, 48.185556, 13.642222, 'Upper Austria'],
    [4753, 48.263611, 13.573611, 'Upper Austria'],
    [4754, 48.265, 13.523889, 'Upper Austria'],
    [4792, 48.483333, 13.566667, 'Upper Austria'],
    [4793, 48.481944, 13.611111, 'Upper Austria'],
    [4794, 48.433056, 13.65, 'Upper Austria'],
    [4863, 47.950278, 13.583611, 'Upper Austria'],
    [4921, 48.194167, 13.544444, 'Upper Austria'],
    [5211, 48, 13.216667, 'Upper Austria'],
    [5300, 47.85, 13.066667, 'Salzburg'],
    [5301, 47.866944, 13.124167, 'Salzburg'],
    [5311, 47.831667, 13.400556, 'Salzburg'],
    [6070, 47.266667, 11.433333, 'Tyrol'],
    [6092, 47.234167, 11.300833, 'Tyrol'],
    [6112, 47.283333, 11.583333, 'Tyrol'],
    [6162, 47.233333, 11.366667, 'Tyrol'],
    [7400, 47.283333, 16.2, 'Burgenland'],
    [7433, 47.366667, 16.233333, 'Burgenland'],
    [8045, 47.140833, 15.490556, 'Styria'],
    [8063, 47.122778, 15.599167, 'Styria'],
    [8076, 47.013333, 15.556389, 'Styria'],
    [8143, 46.946667, 15.376389, 'Styria'],
    [8160, 47.25, 15.594444, 'Styria'],
    [8162, 47.283333, 15.516667, 'Styria'],
    [8163, 47.285833, 15.478056, 'Styria'],
    [8225, 47.301944, 15.833889, 'Styria'],
    [8232, 47.340278, 15.990833, 'Styria'],
    [8333, 47, 15.933333, 'Styria'],
    [8442, 46.786111, 15.45, 'Styria'],
    [8521, 46.83, 15.384167, 'Styria'],
    [8580, 47.066667, 15.083333, 'Styria'],
    [8691, 47.6775, 15.644167, 'Styria'],
    [8700, 47.381667, 15.097222, 'Styria'],
    [8714, 47.3, 14.93, 'Styria'],
    [8715, 47.25, 14.883333, 'Styria'],
    [8723, 47.25, 14.85, 'Styria'],
    [8733, 47.271111, 14.860556, 'Styria'],
    [8741, 47.130833, 14.739444, 'Styria'],
    [8742, 47.068056, 14.695, 'Styria'],
    [8773, 47.38, 14.9, 'Styria'],
    [8793, 47.426111, 15.006667, 'Styria'],
    [8951, 47.533333, 14.083333, 'Styria'],
    [8952, 47.49, 14.098611, 'Styria'],
    [8962, 47.445556, 13.901111, 'Styria'],
    [8967, 47.4075, 13.766944, 'Styria'],
    [8983, 47.555, 13.929167, 'Styria'],
    [9161, 46.55, 14.3, 'Carinthia'],
    [9411, 46.835917, 14.784686, 'Carinthia'],
    [9470, 46.702134, 14.853646, 'Carinthia'],
    [9560, 46.709603, 14.089366, 'Carinthia'],
    [9900, 46.862523, 12.726034, 'Tyrol'],
    [9912, 46.782273, 12.550642, 'Tyrol'],
    [39025, 46.646976, 10.988418, 'South Tyrol']
])

