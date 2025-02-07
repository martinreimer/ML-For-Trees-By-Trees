import pandas as pd
import os
import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CUR_DIR, "..", "..", "data", "merged", "climavi_tree_openmeteo_weather", "final", "Baum Sparkassenweiher_+_openmeteo_merged_processed.csv")

# Define a function to map month to season
def get_season(month):
    if month in [12, 1, 2]:
        return 0 # Winter
    elif month in [3, 4, 5]:
        return 1 # Spring
    elif month in [6, 7, 8]:
        return 2 # Summer
    elif month in [9, 10, 11]:
        return 3 # Autumn


def load_df(path=DATASET_PATH):
    '''
    Load the data from the specified path.
    '''
    df = pd.read_csv(path, sep=',')
    # convert datetime to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    # set datetime as index
    df.set_index('datetime', inplace=True)

    feature_columns = ['-10|ENV__SOIL__VWC', '-30|ENV__SOIL__VWC', '-45|ENV__SOIL__VWC', 
                        '-10|ENV__SOIL__T', '-30|ENV__SOIL__T', '-45|ENV__SOIL__T', 
                        'TOP|ENV__ATMO__T',
                        'DOC_IRRIGATION_PREDICTION',
                        'temperature_2m', 
                        'relative_humidity_2m',
                        'precipitation',
                        'weather_code',
                        'direct_radiation',
                        'et0_fao_evapotranspiration'
                    ]
    df = df[feature_columns]
    return df

def load_daily_data(path=DATASET_PATH):
    """
    Load the data from the specified path.
    
    Parameters:

    Returns:
    pd.DataFrame: DataFrame containing the data
    """
    df = load_df(path)

    # Resample to daily frequency and apply the specified aggregation methods
    df_daily = pd.DataFrame()

    # ENV__SOIL__VWC

    # ENV__SOIL__VWC: max, min, close value
    df_daily['-10|ENV__SOIL__VWC|min'] = df['-10|ENV__SOIL__VWC'].resample('D').min()
    df_daily['-10|ENV__SOIL__VWC|close'] = df['-10|ENV__SOIL__VWC'].resample('D').last()
    df_daily['-10|ENV__SOIL__VWC|max'] = df['-10|ENV__SOIL__VWC'].resample('D').max()

    # ENV__SOIL__VWC: y-targets min
    df_daily['-10|ENV__SOIL__VWC|min|D+1'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(-1)
    df_daily['-10|ENV__SOIL__VWC|min|D+2'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(-2)
    df_daily['-10|ENV__SOIL__VWC|min|D+3'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(-3)
    df_daily['-10|ENV__SOIL__VWC|min|D+4'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(-4)
    df_daily['-10|ENV__SOIL__VWC|min|D+5'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(-5)
    df_daily['-10|ENV__SOIL__VWC|min|D+6'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(-6)
    df_daily['-10|ENV__SOIL__VWC|min|D+7'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(-7)

    # ENV__SOIL__VWC: y-targets last / close
    df_daily['-10|ENV__SOIL__VWC|close|D+1'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(-1)
    df_daily['-10|ENV__SOIL__VWC|close|D+2'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(-2)
    df_daily['-10|ENV__SOIL__VWC|close|D+3'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(-3)
    df_daily['-10|ENV__SOIL__VWC|close|D+4'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(-4)
    df_daily['-10|ENV__SOIL__VWC|close|D+5'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(-5)
    df_daily['-10|ENV__SOIL__VWC|close|D+6'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(-6)
    df_daily['-10|ENV__SOIL__VWC|close|D+7'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(-7)

    # ENV__SOIL__VWC: y-targets max
    df_daily['-10|ENV__SOIL__VWC|max|D+1'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(-1)
    df_daily['-10|ENV__SOIL__VWC|max|D+2'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(-2)
    df_daily['-10|ENV__SOIL__VWC|max|D+3'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(-3)
    df_daily['-10|ENV__SOIL__VWC|max|D+4'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(-4)
    df_daily['-10|ENV__SOIL__VWC|max|D+5'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(-5)
    df_daily['-10|ENV__SOIL__VWC|max|D+6'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(-6)
    df_daily['-10|ENV__SOIL__VWC|max|D+7'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(-7)

    # ENV__SOIL__VWC: #past soil - min
    df_daily['-10|ENV__SOIL__VWC|min|D-6'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(6)
    df_daily['-10|ENV__SOIL__VWC|min|D-5'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(5)
    df_daily['-10|ENV__SOIL__VWC|min|D-4'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(4)
    df_daily['-10|ENV__SOIL__VWC|min|D-3'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(3)
    df_daily['-10|ENV__SOIL__VWC|min|D-2'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(2)
    df_daily['-10|ENV__SOIL__VWC|min|D-1'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(1)

    # ENV__SOIL__VWC: #past soil - max
    df_daily['-10|ENV__SOIL__VWC|max|D-6'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(6)
    df_daily['-10|ENV__SOIL__VWC|max|D-5'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(5)
    df_daily['-10|ENV__SOIL__VWC|max|D-4'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(4)
    df_daily['-10|ENV__SOIL__VWC|max|D-3'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(3)
    df_daily['-10|ENV__SOIL__VWC|max|D-2'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(2)
    df_daily['-10|ENV__SOIL__VWC|max|D-1'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(1)

    # ENV__SOIL__VWC: #past soil - close
    df_daily['-10|ENV__SOIL__VWC|close|D-6'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(6)
    df_daily['-10|ENV__SOIL__VWC|close|D-5'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(5)
    df_daily['-10|ENV__SOIL__VWC|close|D-4'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(4)
    df_daily['-10|ENV__SOIL__VWC|close|D-3'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(3)
    df_daily['-10|ENV__SOIL__VWC|close|D-2'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(2)
    df_daily['-10|ENV__SOIL__VWC|close|D-1'] = df_daily['-10|ENV__SOIL__VWC|close'].shift(1)

    # difference today and past: min
    df_daily['-10|ENV__SOIL__VWC|min|Today_D-1_diff'] = df_daily['-10|ENV__SOIL__VWC|min'] - df_daily['-10|ENV__SOIL__VWC|min|D-1']
    df_daily['-10|ENV__SOIL__VWC|min|Today_D-2_diff'] = df_daily['-10|ENV__SOIL__VWC|min'] - df_daily['-10|ENV__SOIL__VWC|min|D-2']
    df_daily['-10|ENV__SOIL__VWC|min|Today_D-3_diff'] = df_daily['-10|ENV__SOIL__VWC|min'] - df_daily['-10|ENV__SOIL__VWC|min|D-3']
    df_daily['-10|ENV__SOIL__VWC|min|Today_D-4_diff'] = df_daily['-10|ENV__SOIL__VWC|min'] - df_daily['-10|ENV__SOIL__VWC|min|D-4']
    df_daily['-10|ENV__SOIL__VWC|min|Today_D-5_diff'] = df_daily['-10|ENV__SOIL__VWC|min'] - df_daily['-10|ENV__SOIL__VWC|min|D-5']
    df_daily['-10|ENV__SOIL__VWC|min|Today_D-6_diff'] = df_daily['-10|ENV__SOIL__VWC|min'] - df_daily['-10|ENV__SOIL__VWC|min|D-6']
    # difference today and past: max
    df_daily['-10|ENV__SOIL__VWC|max|Today_D-1_diff'] = df_daily['-10|ENV__SOIL__VWC|max'] - df_daily['-10|ENV__SOIL__VWC|max|D-1']
    df_daily['-10|ENV__SOIL__VWC|max|Today_D-2_diff'] = df_daily['-10|ENV__SOIL__VWC|max'] - df_daily['-10|ENV__SOIL__VWC|max|D-2']
    df_daily['-10|ENV__SOIL__VWC|max|Today_D-3_diff'] = df_daily['-10|ENV__SOIL__VWC|max'] - df_daily['-10|ENV__SOIL__VWC|max|D-3']
    df_daily['-10|ENV__SOIL__VWC|max|Today_D-4_diff'] = df_daily['-10|ENV__SOIL__VWC|max'] - df_daily['-10|ENV__SOIL__VWC|max|D-4']
    df_daily['-10|ENV__SOIL__VWC|max|Today_D-5_diff'] = df_daily['-10|ENV__SOIL__VWC|max'] - df_daily['-10|ENV__SOIL__VWC|max|D-5']
    df_daily['-10|ENV__SOIL__VWC|max|Today_D-6_diff'] = df_daily['-10|ENV__SOIL__VWC|max'] - df_daily['-10|ENV__SOIL__VWC|max|D-6']
    # difference today and past: close
    df_daily['-10|ENV__SOIL__VWC|close|Today_D-1_diff'] = df_daily['-10|ENV__SOIL__VWC|close'] - df_daily['-10|ENV__SOIL__VWC|close|D-1']
    df_daily['-10|ENV__SOIL__VWC|close|Today_D-2_diff'] = df_daily['-10|ENV__SOIL__VWC|close'] - df_daily['-10|ENV__SOIL__VWC|close|D-2']
    df_daily['-10|ENV__SOIL__VWC|close|Today_D-3_diff'] = df_daily['-10|ENV__SOIL__VWC|close'] - df_daily['-10|ENV__SOIL__VWC|close|D-3']
    df_daily['-10|ENV__SOIL__VWC|close|Today_D-4_diff'] = df_daily['-10|ENV__SOIL__VWC|close'] - df_daily['-10|ENV__SOIL__VWC|close|D-4']
    df_daily['-10|ENV__SOIL__VWC|close|Today_D-5_diff'] = df_daily['-10|ENV__SOIL__VWC|close'] - df_daily['-10|ENV__SOIL__VWC|close|D-5']
    df_daily['-10|ENV__SOIL__VWC|close|Today_D-6_diff'] = df_daily['-10|ENV__SOIL__VWC|close'] - df_daily['-10|ENV__SOIL__VWC|close|D-6']

    # difference past and day before: Moving difference: min
    df_daily['-10|ENV__SOIL__VWC|min|Moving_diff'] = df_daily['-10|ENV__SOIL__VWC|min'] - df_daily['-10|ENV__SOIL__VWC|min'].shift(1)
    df_daily['-10|ENV__SOIL__VWC|min|Moving_diff_-1'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(1) - df_daily['-10|ENV__SOIL__VWC|min'].shift(2)
    df_daily['-10|ENV__SOIL__VWC|min|Moving_diff_-2'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(2) - df_daily['-10|ENV__SOIL__VWC|min'].shift(3)
    df_daily['-10|ENV__SOIL__VWC|min|Moving_diff_-3'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(3) - df_daily['-10|ENV__SOIL__VWC|min'].shift(4)
    df_daily['-10|ENV__SOIL__VWC|min|Moving_diff_-4'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(4) - df_daily['-10|ENV__SOIL__VWC|min'].shift(5)
    df_daily['-10|ENV__SOIL__VWC|min|Moving_diff_-5'] = df_daily['-10|ENV__SOIL__VWC|min'].shift(5) - df_daily['-10|ENV__SOIL__VWC|min'].shift(6)

    # difference past and day before: Moving difference: max
    df_daily['-10|ENV__SOIL__VWC|max|Moving_diff'] = df_daily['-10|ENV__SOIL__VWC|max'] - df_daily['-10|ENV__SOIL__VWC|max'].shift(1)
    df_daily['-10|ENV__SOIL__VWC|max|Moving_diff_-1'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(1) - df_daily['-10|ENV__SOIL__VWC|max'].shift(2)
    df_daily['-10|ENV__SOIL__VWC|max|Moving_diff_-2'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(2) - df_daily['-10|ENV__SOIL__VWC|max'].shift(3)
    df_daily['-10|ENV__SOIL__VWC|max|Moving_diff_-3'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(3) - df_daily['-10|ENV__SOIL__VWC|max'].shift(4)
    df_daily['-10|ENV__SOIL__VWC|max|Moving_diff_-4'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(4) - df_daily['-10|ENV__SOIL__VWC|max'].shift(5)
    df_daily['-10|ENV__SOIL__VWC|max|Moving_diff_-5'] = df_daily['-10|ENV__SOIL__VWC|max'].shift(5) - df_daily['-10|ENV__SOIL__VWC|max'].shift(6)

    # ENV__SOIL__T
    # ENV__SOIL__T: mean
    df_daily['-10|ENV__SOIL__T|mean'] = df['-10|ENV__SOIL__T'].resample('D').mean()
    # ENV__SOIL__T: past
    df_daily['-10|ENV__SOIL__T|mean|D-1'] = df_daily['-10|ENV__SOIL__T|mean'].shift(1)
    df_daily['-10|ENV__SOIL__T|mean|D-2'] = df_daily['-10|ENV__SOIL__T|mean'].shift(2)
    df_daily['-10|ENV__SOIL__T|mean|D-3'] = df_daily['-10|ENV__SOIL__T|mean'].shift(3)
    df_daily['-10|ENV__SOIL__T|mean|D-4'] = df_daily['-10|ENV__SOIL__T|mean'].shift(4)
    df_daily['-10|ENV__SOIL__T|mean|D-5'] = df_daily['-10|ENV__SOIL__T|mean'].shift(5)
    df_daily['-10|ENV__SOIL__T|mean|D-6'] = df_daily['-10|ENV__SOIL__T|mean'].shift(6)
    # ENV__SOIL__T: Difference past to today
    df_daily['-10|ENV__SOIL__T|mean|Today_D-1_diff'] = df_daily['-10|ENV__SOIL__T|mean'] - df_daily['-10|ENV__SOIL__T|mean|D-1']
    df_daily['-10|ENV__SOIL__T|mean|Today_D-2_diff'] = df_daily['-10|ENV__SOIL__T|mean'] - df_daily['-10|ENV__SOIL__T|mean|D-2']
    df_daily['-10|ENV__SOIL__T|mean|Today_D-3_diff'] = df_daily['-10|ENV__SOIL__T|mean'] - df_daily['-10|ENV__SOIL__T|mean|D-3']
    df_daily['-10|ENV__SOIL__T|mean|Today_D-4_diff'] = df_daily['-10|ENV__SOIL__T|mean'] - df_daily['-10|ENV__SOIL__T|mean|D-4']
    df_daily['-10|ENV__SOIL__T|mean|Today_D-5_diff'] = df_daily['-10|ENV__SOIL__T|mean'] - df_daily['-10|ENV__SOIL__T|mean|D-5']
    df_daily['-10|ENV__SOIL__T|mean|Today_D-6_diff'] = df_daily['-10|ENV__SOIL__T|mean'] - df_daily['-10|ENV__SOIL__T|mean|D-6']
    # ENV__SOIL__T: Moving difference
    df_daily['-10|ENV__SOIL__T|mean|Moving_diff'] = df_daily['-10|ENV__SOIL__T|mean'] - df_daily['-10|ENV__SOIL__T|mean'].shift(1)
    df_daily['-10|ENV__SOIL__T|mean|Moving_diff_-1'] = df_daily['-10|ENV__SOIL__T|mean'].shift(1) - df_daily['-10|ENV__SOIL__T|mean'].shift(2)
    df_daily['-10|ENV__SOIL__T|mean|Moving_diff_-2'] = df_daily['-10|ENV__SOIL__T|mean'].shift(2) - df_daily['-10|ENV__SOIL__T|mean'].shift(3)
    df_daily['-10|ENV__SOIL__T|mean|Moving_diff_-3'] = df_daily['-10|ENV__SOIL__T|mean'].shift(3) - df_daily['-10|ENV__SOIL__T|mean'].shift(4)
    df_daily['-10|ENV__SOIL__T|mean|Moving_diff_-4'] = df_daily['-10|ENV__SOIL__T|mean'].shift(4) - df_daily['-10|ENV__SOIL__T|mean'].shift(5)
    df_daily['-10|ENV__SOIL__T|mean|Moving_diff_-5'] = df_daily['-10|ENV__SOIL__T|mean'].shift(5) - df_daily['-10|ENV__SOIL__T|mean'].shift(6)


    # direct_radiation
    df_daily['direct_radiation|sum'] = df['direct_radiation'].resample('D').sum()
    # direct_radiation: forecast
    df_daily['direct_radiation|sum|D+1'] = df_daily['direct_radiation|sum'].shift(-1)
    df_daily['direct_radiation|sum|D+2'] = df_daily['direct_radiation|sum'].shift(-2)
    df_daily['direct_radiation|sum|D+3'] = df_daily['direct_radiation|sum'].shift(-3)
    df_daily['direct_radiation|sum|D+4'] = df_daily['direct_radiation|sum'].shift(-4)
    df_daily['direct_radiation|sum|D+5'] = df_daily['direct_radiation|sum'].shift(-5)
    df_daily['direct_radiation|sum|D+6'] = df_daily['direct_radiation|sum'].shift(-6)
    df_daily['direct_radiation|sum|D+7'] = df_daily['direct_radiation|sum'].shift(-7)
    # direct_radiation: past
    df_daily['direct_radiation|sum|D-1'] = df_daily['direct_radiation|sum'].shift(1)
    df_daily['direct_radiation|sum|D-2'] = df_daily['direct_radiation|sum'].shift(2)
    df_daily['direct_radiation|sum|D-3'] = df_daily['direct_radiation|sum'].shift(3)
    df_daily['direct_radiation|sum|D-4'] = df_daily['direct_radiation|sum'].shift(4)
    df_daily['direct_radiation|sum|D-5'] = df_daily['direct_radiation|sum'].shift(5)
    df_daily['direct_radiation|sum|D-6'] = df_daily['direct_radiation|sum'].shift(6)
    # direct_radiation: past accumulated
    df_daily['direct_radiation|sum|7_D_acc'] = df_daily['direct_radiation|sum'].shift(1) + df_daily['direct_radiation|sum'].shift(2) + df_daily['direct_radiation|sum'].shift(3) + df_daily['direct_radiation|sum'].shift(4) + df_daily['direct_radiation|sum'].shift(5) + df_daily['direct_radiation|sum'].shift(6)


    # precipitation
    df_daily['precipitation|sum'] = df['precipitation'].resample('D').sum()
    # rain forecast
    df_daily['precipitation|sum|D+1'] = df_daily['precipitation|sum'].shift(-1)
    df_daily['precipitation|sum|D+2'] = df_daily['precipitation|sum'].shift(-2)
    df_daily['precipitation|sum|D+3'] = df_daily['precipitation|sum'].shift(-3)
    df_daily['precipitation|sum|D+4'] = df_daily['precipitation|sum'].shift(-4)
    df_daily['precipitation|sum|D+5'] = df_daily['precipitation|sum'].shift(-5)
    df_daily['precipitation|sum|D+6'] = df_daily['precipitation|sum'].shift(-6)
    df_daily['precipitation|sum|D+7'] = df_daily['precipitation|sum'].shift(-7)
    # past rain
    df_daily['precipitation|sum|D-1'] = df_daily['precipitation|sum'].shift(1)
    df_daily['precipitation|sum|D-2'] = df_daily['precipitation|sum'].shift(2)
    df_daily['precipitation|sum|D-3'] = df_daily['precipitation|sum'].shift(3)
    df_daily['precipitation|sum|D-4'] = df_daily['precipitation|sum'].shift(4)
    df_daily['precipitation|sum|D-5'] = df_daily['precipitation|sum'].shift(5)
    df_daily['precipitation|sum|D-6'] = df_daily['precipitation|sum'].shift(6)
    # past accumulated rain
    df_daily['precipitation|sum|7_D_acc'] = df_daily['precipitation|sum'].shift(1) + df_daily['precipitation|sum'].shift(2) + df_daily['precipitation|sum'].shift(3) + df_daily['precipitation|sum'].shift(4) + df_daily['precipitation|sum'].shift(5) + df_daily['precipitation|sum'].shift(6)
    # lots of rain binary column
    df_daily['precipitation|lots_of_rain'] = df_daily['precipitation|sum'].apply(lambda x: 1 if x >= 10 else 0)
    df_daily['precipitation|lots_of_rain|D-1'] = df_daily['precipitation|lots_of_rain'].shift(1)
    df_daily['precipitation|lots_of_rain|D-2'] = df_daily['precipitation|lots_of_rain'].shift(2)
    df_daily['precipitation|lots_of_rain|D-3'] = df_daily['precipitation|lots_of_rain'].shift(3)
    df_daily['precipitation|lots_of_rain|D-4'] = df_daily['precipitation|lots_of_rain'].shift(4)
    df_daily['precipitation|lots_of_rain|D-5'] = df_daily['precipitation|lots_of_rain'].shift(5)
    df_daily['precipitation|lots_of_rain|D-6'] = df_daily['precipitation|lots_of_rain'].shift(6)
    df_daily['precipitation|lots_of_rain|D-7'] = df_daily['precipitation|lots_of_rain'].shift(7)
    df_daily['precipitation|lots_of_rain|D+1'] = df_daily['precipitation|lots_of_rain'].shift(-1)
    df_daily['precipitation|lots_of_rain|D+2'] = df_daily['precipitation|lots_of_rain'].shift(-2)
    df_daily['precipitation|lots_of_rain|D+3'] = df_daily['precipitation|lots_of_rain'].shift(-3)
    df_daily['precipitation|lots_of_rain|D+4'] = df_daily['precipitation|lots_of_rain'].shift(-4)
    df_daily['precipitation|lots_of_rain|D+5'] = df_daily['precipitation|lots_of_rain'].shift(-5)
    df_daily['precipitation|lots_of_rain|D+6'] = df_daily['precipitation|lots_of_rain'].shift(-6)
    df_daily['precipitation|lots_of_rain|D+7'] = df_daily['precipitation|lots_of_rain'].shift(-7)
    # rain of today + tomorrow
    df_daily['precipitation|sum|D+D+1'] = df_daily['precipitation|sum'] + df_daily['precipitation|sum'].shift(-1)
    df_daily['precipitation|sum|D+1+D+2'] = df_daily['precipitation|sum'].shift(-1) + df_daily['precipitation|sum'].shift(-2)
    df_daily['precipitation|sum|D+2+D+3'] = df_daily['precipitation|sum'].shift(-2) + df_daily['precipitation|sum'].shift(-3)
    df_daily['precipitation|sum|D+3+D+4'] = df_daily['precipitation|sum'].shift(-3) + df_daily['precipitation|sum'].shift(-4)
    df_daily['precipitation|sum|D+4+D+5'] = df_daily['precipitation|sum'].shift(-4) + df_daily['precipitation|sum'].shift(-5)
    df_daily['precipitation|sum|D+5+D+6'] = df_daily['precipitation|sum'].shift(-5) + df_daily['precipitation|sum'].shift(-6)
    df_daily['precipitation|sum|D+6+D+7'] = df_daily['precipitation|sum'].shift(-6) + df_daily['precipitation|sum'].shift(-7)
    
    # DOC_IRRIGATION_PREDICTION
    df_daily['DOC_IRRIGATION_PREDICTION|max'] = df['DOC_IRRIGATION_PREDICTION'].resample('D').max()
    # DOC_IRRIGATION_PREDICTION: shift -1 irrigation prediction bcs its shifted
    #df_daily['DOC_IRRIGATION_PREDICTION|max'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(-1)
    # create column for irrigation prediction to mark rows which have irrigation prediction in the next 3 days or 5 days before
    df_daily['DOC_IRRIGATION_PREDICTION|max|D-1'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(1)
    df_daily['DOC_IRRIGATION_PREDICTION|max|D-2'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(2)
    df_daily['DOC_IRRIGATION_PREDICTION|max|D-3'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(3)
    df_daily['DOC_IRRIGATION_PREDICTION|max|D-4'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(4)
    df_daily['DOC_IRRIGATION_PREDICTION|max|D-5'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(5)
    df_daily['DOC_IRRIGATION_PREDICTION|max|D-6'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(6)

    df_daily['DOC_IRRIGATION_PREDICTION|max|D+1'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(-1)
    df_daily['DOC_IRRIGATION_PREDICTION|max|D+2'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(-2)
    df_daily['DOC_IRRIGATION_PREDICTION|max|D+3'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(-3)
    df_daily['DOC_IRRIGATION_PREDICTION|max|D+4'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(-4)
    df_daily['DOC_IRRIGATION_PREDICTION|max|D+5'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(-5)
    df_daily['DOC_IRRIGATION_PREDICTION|max|D+6'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(-6)
    df_daily['DOC_IRRIGATION_PREDICTION|max|D+7'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(-7)
    # create accumulated irrigation prediction of last 5 days
    df_daily['DOC_IRRIGATION_PREDICTION|max|7_D_acc'] = df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(1) + df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(2) + df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(3) + df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(4) + df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(5) + + df_daily['DOC_IRRIGATION_PREDICTION|max'].shift(6)


    # weather temperature
    df_daily['temperature_2m|mean'] = df['temperature_2m'].resample('D').mean()
    df_daily['temperature_2m|max'] = df['temperature_2m'].resample('D').max()
    # temperature_2m: forecast: mean
    df_daily['temperature_2m|mean|D+1'] = df_daily['temperature_2m|mean'].shift(-1)
    df_daily['temperature_2m|mean|D+2'] = df_daily['temperature_2m|mean'].shift(-2)
    df_daily['temperature_2m|mean|D+3'] = df_daily['temperature_2m|mean'].shift(-3)
    df_daily['temperature_2m|mean|D+4'] = df_daily['temperature_2m|mean'].shift(-4)
    df_daily['temperature_2m|mean|D+5'] = df_daily['temperature_2m|mean'].shift(-5)
    df_daily['temperature_2m|mean|D+6'] = df_daily['temperature_2m|mean'].shift(-6)
    df_daily['temperature_2m|mean|D+7'] = df_daily['temperature_2m|mean'].shift(-7)
    # temperature_2m: forecast: max
    df_daily['temperature_2m|max|D+1'] = df_daily['temperature_2m|max'].shift(-1)
    df_daily['temperature_2m|max|D+2'] = df_daily['temperature_2m|max'].shift(-2)
    df_daily['temperature_2m|max|D+3'] = df_daily['temperature_2m|max'].shift(-3)
    df_daily['temperature_2m|max|D+4'] = df_daily['temperature_2m|max'].shift(-4)
    df_daily['temperature_2m|max|D+5'] = df_daily['temperature_2m|max'].shift(-5)
    df_daily['temperature_2m|max|D+6'] = df_daily['temperature_2m|max'].shift(-6)
    df_daily['temperature_2m|max|D+7'] = df_daily['temperature_2m|max'].shift(-7)

    # temperature_2m: past: mean
    df_daily['temperature_2m|mean|D-1'] = df_daily['temperature_2m|mean'].shift(1)
    df_daily['temperature_2m|mean|D-2'] = df_daily['temperature_2m|mean'].shift(2)
    df_daily['temperature_2m|mean|D-3'] = df_daily['temperature_2m|mean'].shift(3)
    df_daily['temperature_2m|mean|D-4'] = df_daily['temperature_2m|mean'].shift(4)
    df_daily['temperature_2m|mean|D-5'] = df_daily['temperature_2m|mean'].shift(5)
    df_daily['temperature_2m|mean|D-6'] = df_daily['temperature_2m|mean'].shift(6)
    # temperature_2m: past: max
    df_daily['temperature_2m|max|D-1'] = df_daily['temperature_2m|max'].shift(1)
    df_daily['temperature_2m|max|D-2'] = df_daily['temperature_2m|max'].shift(2)
    df_daily['temperature_2m|max|D-3'] = df_daily['temperature_2m|max'].shift(3)
    df_daily['temperature_2m|max|D-4'] = df_daily['temperature_2m|max'].shift(4)
    df_daily['temperature_2m|max|D-5'] = df_daily['temperature_2m|max'].shift(5)
    df_daily['temperature_2m|max|D-6'] = df_daily['temperature_2m|max'].shift(6)

    # temperature_2m: moving difference: mean
    df_daily['temperature_2m|mean|Moving_diff'] = df_daily['temperature_2m|mean'] - df_daily['temperature_2m|mean'].shift(1)
    df_daily['temperature_2m|mean|Moving_diff_-1'] = df_daily['temperature_2m|mean'].shift(1) - df_daily['temperature_2m|mean'].shift(2)
    df_daily['temperature_2m|mean|Moving_diff_-2'] = df_daily['temperature_2m|mean'].shift(2) - df_daily['temperature_2m|mean'].shift(3)
    df_daily['temperature_2m|mean|Moving_diff_-3'] = df_daily['temperature_2m|mean'].shift(3) - df_daily['temperature_2m|mean'].shift(4)
    df_daily['temperature_2m|mean|Moving_diff_-4'] = df_daily['temperature_2m|mean'].shift(4) - df_daily['temperature_2m|mean'].shift(5)
    df_daily['temperature_2m|mean|Moving_diff_-5'] = df_daily['temperature_2m|mean'].shift(5) - df_daily['temperature_2m|mean'].shift(6)

    # temperature_2m: moving difference: max
    df_daily['temperature_2m|max|Moving_diff'] = df_daily['temperature_2m|max'] - df_daily['temperature_2m|max'].shift(1)
    df_daily['temperature_2m|max|Moving_diff_-1'] = df_daily['temperature_2m|max'].shift(1) - df_daily['temperature_2m|max'].shift(2)
    df_daily['temperature_2m|max|Moving_diff_-2'] = df_daily['temperature_2m|max'].shift(2) - df_daily['temperature_2m|max'].shift(3)
    df_daily['temperature_2m|max|Moving_diff_-3'] = df_daily['temperature_2m|max'].shift(3) - df_daily['temperature_2m|max'].shift(4)
    df_daily['temperature_2m|max|Moving_diff_-4'] = df_daily['temperature_2m|max'].shift(4) - df_daily['temperature_2m|max'].shift(5)
    df_daily['temperature_2m|max|Moving_diff_-5'] = df_daily['temperature_2m|max'].shift(5) - df_daily['temperature_2m|max'].shift(6)


    # relative_humidity_2m
    df_daily['relative_humidity_2m|mean'] = df['relative_humidity_2m'].resample('D').mean()
    # relative_humidity_2m: forecast
    df_daily['relative_humidity_2m|mean|D+1'] = df_daily['relative_humidity_2m|mean'].shift(-1)
    df_daily['relative_humidity_2m|mean|D+2'] = df_daily['relative_humidity_2m|mean'].shift(-2)
    df_daily['relative_humidity_2m|mean|D+3'] = df_daily['relative_humidity_2m|mean'].shift(-3)
    df_daily['relative_humidity_2m|mean|D+4'] = df_daily['relative_humidity_2m|mean'].shift(-4)
    df_daily['relative_humidity_2m|mean|D+5'] = df_daily['relative_humidity_2m|mean'].shift(-5)
    df_daily['relative_humidity_2m|mean|D+6'] = df_daily['relative_humidity_2m|mean'].shift(-6)
    df_daily['relative_humidity_2m|mean|D+7'] = df_daily['relative_humidity_2m|mean'].shift(-7)
    # relative_humidity_2m: past
    df_daily['relative_humidity_2m|mean|D-1'] = df_daily['relative_humidity_2m|mean'].shift(1)
    df_daily['relative_humidity_2m|mean|D-2'] = df_daily['relative_humidity_2m|mean'].shift(2)
    df_daily['relative_humidity_2m|mean|D-3'] = df_daily['relative_humidity_2m|mean'].shift(3)
    df_daily['relative_humidity_2m|mean|D-4'] = df_daily['relative_humidity_2m|mean'].shift(4)
    df_daily['relative_humidity_2m|mean|D-5'] = df_daily['relative_humidity_2m|mean'].shift(5)
    df_daily['relative_humidity_2m|mean|D-6'] = df_daily['relative_humidity_2m|mean'].shift(6)
    

    # create month as feature
    df_daily['Month'] = df_daily.index.month

    # create season as feature
    df_daily['Season'] = df_daily.index.month.map(get_season)

    # create irrigation season as feature: April - October is irrigation season
    df_daily['Irrigation_Season'] = 0
    df_daily.loc[(df_daily.index.month >= 4) & (df_daily.index.month <= 10), 'Irrigation_Season'] = 1

    return df_daily

# Function to create lagged features for hourly data
def create_hourly_lagged_features(df, column, max_days_lag):
    max_lag_hours = max_days_lag * 24
    for lag in range(1, max_lag_hours + 1):
        df[f'{column}|H-{lag}'] = df[column].shift(lag)
    return df

def load_hourly_data(path=DATASET_PATH):
    """
    Load the data from the specified path.
    
    Parameters:

    Returns:
    pd.DataFrame: DataFrame containing the data
    """
    df_daily = load_daily_data(path)

    df = load_df(path)
    # Aggregate hourly soil moisture to daily min, max, and close values
    df_hourly = df.copy()  # Copy your original hourly data to avoid modifying it directly

    # List of columns to create lags for
    columns_to_lag = ['-10|ENV__SOIL__VWC', '-10|ENV__SOIL__T', 'direct_radiation', 'precipitation', 'temperature_2m', 'relative_humidity_2m']

    max_days_lag = 3  # The last 3 days in hourly features

    # Create lagged hourly features for each specified column in df_hourly
    for col in columns_to_lag:
        df_hourly = create_hourly_lagged_features(df_hourly, col, max_days_lag)

    # Add a date column to `df_hourly` for merging with `df_daily`
    df_hourly['date'] = df_hourly.index.date  # Assuming `index` is a datetime index

    # Select only the last hourly record of each day with the relevant lagged features
    df_hourly_last = df_hourly.groupby('date').last().reset_index()  # Only keep the last row for each day

    # Merge the last hourly data of each day (including lagged features) with the daily DataFrame
    df_daily['date'] = df_daily.index.date if df_daily.index.dtype == 'datetime64[ns]' else df_daily['date']
    df_daily_combined = df_daily.merge(df_hourly_last, on='date', how='left')

    # set datetime as index
    df_daily_combined["datetime"] = pd.to_datetime(df_daily_combined["date"])
    df_daily_combined.set_index('datetime', inplace=True)
    return df_daily_combined

def load_feature_groups(num_days_ahead_pred, y_target_types, data_granularity='d'):
    '''
    num_days_ahead_pred: int, number of days ahead to predict
    y_target_types: list of strings, target types to predict
        - 'min', 'max', 'close'
    '''

    # hourly feature groups
    feature_groups_hourly = {
        "past_raw_hourly_vwc_values": ['-10|ENV__SOIL__VWC|H-1', '-10|ENV__SOIL__VWC|H-2', '-10|ENV__SOIL__VWC|H-3', '-10|ENV__SOIL__VWC|H-4', '-10|ENV__SOIL__VWC|H-5', '-10|ENV__SOIL__VWC|H-6', '-10|ENV__SOIL__VWC|H-7', '-10|ENV__SOIL__VWC|H-8', '-10|ENV__SOIL__VWC|H-9', '-10|ENV__SOIL__VWC|H-10', '-10|ENV__SOIL__VWC|H-11', '-10|ENV__SOIL__VWC|H-12', '-10|ENV__SOIL__VWC|H-13', '-10|ENV__SOIL__VWC|H-14', '-10|ENV__SOIL__VWC|H-15', '-10|ENV__SOIL__VWC|H-16', '-10|ENV__SOIL__VWC|H-17', '-10|ENV__SOIL__VWC|H-18', '-10|ENV__SOIL__VWC|H-19', '-10|ENV__SOIL__VWC|H-20', '-10|ENV__SOIL__VWC|H-21', '-10|ENV__SOIL__VWC|H-22', '-10|ENV__SOIL__VWC|H-23', '-10|ENV__SOIL__VWC|H-24', '-10|ENV__SOIL__VWC|H-25', '-10|ENV__SOIL__VWC|H-26', '-10|ENV__SOIL__VWC|H-27', '-10|ENV__SOIL__VWC|H-28', '-10|ENV__SOIL__VWC|H-29', '-10|ENV__SOIL__VWC|H-30', '-10|ENV__SOIL__VWC|H-31', '-10|ENV__SOIL__VWC|H-32'],
        "past_raw_hourly_tree_temp": ['-10|ENV__SOIL__T|H-1', '-10|ENV__SOIL__T|H-2', '-10|ENV__SOIL__T|H-3', '-10|ENV__SOIL__T|H-4', '-10|ENV__SOIL__T|H-5', '-10|ENV__SOIL__T|H-6', '-10|ENV__SOIL__T|H-7', '-10|ENV__SOIL__T|H-8', '-10|ENV__SOIL__T|H-9', '-10|ENV__SOIL__T|H-10', '-10|ENV__SOIL__T|H-11', '-10|ENV__SOIL__T|H-12', '-10|ENV__SOIL__T|H-13', '-10|ENV__SOIL__T|H-14', '-10|ENV__SOIL__T|H-15', '-10|ENV__SOIL__T|H-16', '-10|ENV__SOIL__T|H-17', '-10|ENV__SOIL__T|H-18', '-10|ENV__SOIL__T|H-19', '-10|ENV__SOIL__T|H-20', '-10|ENV__SOIL__T|H-21', '-10|ENV__SOIL__T|H-22', '-10|ENV__SOIL__T|H-23', '-10|ENV__SOIL__T|H-24', '-10|ENV__SOIL__T|H-25', '-10|ENV__SOIL__T|H-26', '-10|ENV__SOIL__T|H-27', '-10|ENV__SOIL__T|H-28', '-10|ENV__SOIL__T|H-29', '-10|ENV__SOIL__T|H-30', '-10|ENV__SOIL__T|H-31', '-10|ENV__SOIL__T|H-32'],
        "past_raw_hourly_direct_radiation": ['direct_radiation|H-1', 'direct_radiation|H-2', 'direct_radiation|H-3', 'direct_radiation|H-4', 'direct_radiation|H-5', 'direct_radiation|H-6', 'direct_radiation|H-7', 'direct_radiation|H-8', 'direct_radiation|H-9', 'direct_radiation|H-10', 'direct_radiation|H-11', 'direct_radiation|H-12', 'direct_radiation|H-13', 'direct_radiation|H-14', 'direct_radiation|H-15', 'direct_radiation|H-16', 'direct_radiation|H-17', 'direct_radiation|H-18', 'direct_radiation|H-19', 'direct_radiation|H-20', 'direct_radiation|H-21', 'direct_radiation|H-22', 'direct_radiation|H-23', 'direct_radiation|H-24', 'direct_radiation|H-25', 'direct_radiation|H-26', 'direct_radiation|H-27', 'direct_radiation|H-28', 'direct_radiation|H-29', 'direct_radiation|H-30', 'direct_radiation|H-31', 'direct_radiation|H-32'],
        "past_hourly_rain": ['precipitation|H-1', 'precipitation|H-2', 'precipitation|H-3', 'precipitation|H-4', 'precipitation|H-5', 'precipitation|H-6', 'precipitation|H-7', 'precipitation|H-8', 'precipitation|H-9', 'precipitation|H-10', 'precipitation|H-11', 'precipitation|H-12', 'precipitation|H-13', 'precipitation|H-14', 'precipitation|H-15', 'precipitation|H-16', 'precipitation|H-17', 'precipitation|H-18', 'precipitation|H-19', 'precipitation|H-20', 'precipitation|H-21', 'precipitation|H-22', 'precipitation|H-23', 'precipitation|H-24', 'precipitation|H-25', 'precipitation|H-26', 'precipitation|H-27', 'precipitation|H-28', 'precipitation|H-29', 'precipitation|H-30', 'precipitation|H-31', 'precipitation|H-32'],
        "past_hourly_humdity": ['relative_humidity_2m|H-1', 'relative_humidity_2m|H-2', 'relative_humidity_2m|H-3', 'relative_humidity_2m|H-4', 'relative_humidity_2m|H-5', 'relative_humidity_2m|H-6', 'relative_humidity_2m|H-7', 'relative_humidity_2m|H-8', 'relative_humidity_2m|H-9', 'relative_humidity_2m|H-10', 'relative_humidity_2m|H-11', 'relative_humidity_2m|H-12', 'relative_humidity_2m|H-13', 'relative_humidity_2m|H-14', 'relative_humidity_2m|H-15', 'relative_humidity_2m|H-16', 'relative_humidity_2m|H-17', 'relative_humidity_2m|H-18', 'relative_humidity_2m|H-19', 'relative_humidity_2m|H-20', 'relative_humidity_2m|H-21', 'relative_humidity_2m|H-22', 'relative_humidity_2m|H-23', 'relative_humidity_2m|H-24', 'relative_humidity_2m|H-25', 'relative_humidity_2m|H-26', 'relative_humidity_2m|H-27', 'relative_humidity_2m|H-28', 'relative_humidity_2m|H-29', 'relative_humidity_2m|H-30', 'relative_humidity_2m|H-31', 'relative_humidity_2m|H-32'],
    }

    # 1 day ahead prediction
    feature_groups_1_day_ahead = {
        "past_raw_vwc_min_values": ['-10|ENV__SOIL__VWC|min', '-10|ENV__SOIL__VWC|min|D-6', '-10|ENV__SOIL__VWC|min|D-5', '-10|ENV__SOIL__VWC|min|D-4', '-10|ENV__SOIL__VWC|min|D-3', '-10|ENV__SOIL__VWC|min|D-2', '-10|ENV__SOIL__VWC|min|D-1'],
        "past_raw_vwc_max_values" : ['-10|ENV__SOIL__VWC|max', '-10|ENV__SOIL__VWC|max|D-6', '-10|ENV__SOIL__VWC|max|D-5', '-10|ENV__SOIL__VWC|max|D-4', '-10|ENV__SOIL__VWC|max|D-3', '-10|ENV__SOIL__VWC|max|D-2', '-10|ENV__SOIL__VWC|max|D-1'],
        "past_raw_vwc_close_values": ['-10|ENV__SOIL__VWC|close', '-10|ENV__SOIL__VWC|close|D-6', '-10|ENV__SOIL__VWC|close|D-5', '-10|ENV__SOIL__VWC|close|D-4', '-10|ENV__SOIL__VWC|close|D-3', '-10|ENV__SOIL__VWC|close|D-2', '-10|ENV__SOIL__VWC|close|D-1'],
        "past_vwc_min_moving_diff": ['-10|ENV__SOIL__VWC|min|Moving_diff', '-10|ENV__SOIL__VWC|min|Moving_diff_-1', '-10|ENV__SOIL__VWC|min|Moving_diff_-2', '-10|ENV__SOIL__VWC|min|Moving_diff_-3', '-10|ENV__SOIL__VWC|min|Moving_diff_-4', '-10|ENV__SOIL__VWC|min|Moving_diff_-5'],
        "past_vwc_max_moving_diff": ['-10|ENV__SOIL__VWC|max|Moving_diff', '-10|ENV__SOIL__VWC|max|Moving_diff_-1', '-10|ENV__SOIL__VWC|max|Moving_diff_-2', '-10|ENV__SOIL__VWC|max|Moving_diff_-3', '-10|ENV__SOIL__VWC|max|Moving_diff_-4', '-10|ENV__SOIL__VWC|max|Moving_diff_-5'],
        "past_vwc_min_current_diff": ['-10|ENV__SOIL__VWC|min|Today_D-1_diff', '-10|ENV__SOIL__VWC|min|Today_D-2_diff', '-10|ENV__SOIL__VWC|min|Today_D-3_diff', '-10|ENV__SOIL__VWC|min|Today_D-4_diff', '-10|ENV__SOIL__VWC|min|Today_D-5_diff'],
        "past_vwc_max_current_diff": ['-10|ENV__SOIL__VWC|max|Today_D-1_diff', '-10|ENV__SOIL__VWC|max|Today_D-2_diff', '-10|ENV__SOIL__VWC|max|Today_D-3_diff', '-10|ENV__SOIL__VWC|max|Today_D-4_diff', '-10|ENV__SOIL__VWC|max|Today_D-5_diff'],
        "past_vwc_close_current_diff": ['-10|ENV__SOIL__VWC|close|Today_D-1_diff', '-10|ENV__SOIL__VWC|close|Today_D-2_diff', '-10|ENV__SOIL__VWC|close|Today_D-3_diff', '-10|ENV__SOIL__VWC|close|Today_D-4_diff', '-10|ENV__SOIL__VWC|close|Today_D-5_diff'],

        "past_rain_sum": ['precipitation|sum', 'precipitation|sum|D-1', 'precipitation|sum|D-2', 'precipitation|sum|D-3', 'precipitation|sum|D-4', 'precipitation|sum|D-5', 'precipitation|sum|D-6'],
        "past_acc_rain_sum": ['precipitation|sum|7_D_acc'],
        "lots_of_rain": ["precipitation|lots_of_rain", "precipitation|lots_of_rain|D-1", "precipitation|lots_of_rain|D-2", "precipitation|lots_of_rain|D-3", "precipitation|lots_of_rain|D-4", "precipitation|lots_of_rain|D-5", "precipitation|lots_of_rain|D-6", "precipitation|lots_of_rain|D+1"],
        "future_rain_sum": ['precipitation|sum|D+1', 'precipitation|sum|D+D+1'],
        
        "irrigations": ['DOC_IRRIGATION_PREDICTION|max', 'DOC_IRRIGATION_PREDICTION|max|D-1', 'DOC_IRRIGATION_PREDICTION|max|D-2', 'DOC_IRRIGATION_PREDICTION|max|D-3', 'DOC_IRRIGATION_PREDICTION|max|D-4', 'DOC_IRRIGATION_PREDICTION|max|D-5', 'DOC_IRRIGATION_PREDICTION|max|D-6'],
        "past_acc_irrigations": ['DOC_IRRIGATION_PREDICTION|max|7_D_acc'],
        "future_irrigations": ['DOC_IRRIGATION_PREDICTION|max', 'DOC_IRRIGATION_PREDICTION|max|D+1'],
        
        "seasonality": ['Month', 'Season', 'Irrigation_Season'],
        
        "past_tree_temp": ['-10|ENV__SOIL__T|mean', '-10|ENV__SOIL__T|mean|D-1', '-10|ENV__SOIL__T|mean|D-2', '-10|ENV__SOIL__T|mean|D-3', '-10|ENV__SOIL__T|mean|D-4', '-10|ENV__SOIL__T|mean|D-5', '-10|ENV__SOIL__T|mean|D-6'],
        "past_tree_temp_moving_diff": ['-10|ENV__SOIL__T|mean|Moving_diff', '-10|ENV__SOIL__T|mean|Moving_diff_-1', '-10|ENV__SOIL__T|mean|Moving_diff_-2', '-10|ENV__SOIL__T|mean|Moving_diff_-3', '-10|ENV__SOIL__T|mean|Moving_diff_-4', '-10|ENV__SOIL__T|mean|Moving_diff_-5'],
        "past_tree_temp_current_diff": ['-10|ENV__SOIL__T|mean|Today_D-1_diff', '-10|ENV__SOIL__T|mean|Today_D-2_diff', '-10|ENV__SOIL__T|mean|Today_D-3_diff', '-10|ENV__SOIL__T|mean|Today_D-4_diff', '-10|ENV__SOIL__T|mean|Today_D-5_diff'],
        
        "past_weather_temp_mean": ['temperature_2m|mean', 'temperature_2m|mean', 'temperature_2m|mean|D-1', 'temperature_2m|mean|D-2', 'temperature_2m|mean|D-3', 'temperature_2m|mean|D-4', 'temperature_2m|mean|D-5', 'temperature_2m|mean|D-6'],
        "past_weather_temp_max": ['temperature_2m|max', 'temperature_2m|max|D-1', 'temperature_2m|max|D-2', 'temperature_2m|max|D-3', 'temperature_2m|max|D-4', 'temperature_2m|max|D-5', 'temperature_2m|max|D-6'],
        "past_weather_temp_mean_moving_diff": ['temperature_2m|mean|Moving_diff', 'temperature_2m|mean|Moving_diff_-1', 'temperature_2m|mean|Moving_diff_-2', 'temperature_2m|mean|Moving_diff_-3', 'temperature_2m|mean|Moving_diff_-4', 'temperature_2m|mean|Moving_diff_-5'],
        "past_weather_temp_max_moving_diff": ['temperature_2m|max|Moving_diff', 'temperature_2m|max|Moving_diff_-1', 'temperature_2m|max|Moving_diff_-2', 'temperature_2m|max|Moving_diff_-3', 'temperature_2m|max|Moving_diff_-4', 'temperature_2m|max|Moving_diff_-5'],
        "future_weather_temp_mean": ['temperature_2m|mean', 'temperature_2m|mean|D+1'],
        "future_weather_temp_max": ['temperature_2m|max', 'temperature_2m|max|D+1'],

        "future_radiation": ['direct_radiation|sum', 'direct_radiation|sum|D+1'],
        "past_radiation": ['direct_radiation|sum', 'direct_radiation|sum|D-1', 'direct_radiation|sum|D-2', 'direct_radiation|sum|D-3', 'direct_radiation|sum|D-4', 'direct_radiation|sum|D-5', 'direct_radiation|sum|D-6'],
        "past_acc_radiation": ['direct_radiation|sum|7_D_acc'],
        
        "past_humdity": ['relative_humidity_2m|mean', 'relative_humidity_2m|mean|D-1', 'relative_humidity_2m|mean|D-2', 'relative_humidity_2m|mean|D-3', 'relative_humidity_2m|mean|D-4', 'relative_humidity_2m|mean|D-5', 'relative_humidity_2m|mean|D-6'],
        "future_humidity": ['relative_humidity_2m|mean', 'relative_humidity_2m|mean|D+1'],
    }
    if data_granularity == 'h':
        feature_groups_1_day_ahead.update(feature_groups_hourly)

    y_targets_1_day_ahead = []
    for y_target_type in y_target_types:
        y_targets_1_day_ahead.append(f'-10|ENV__SOIL__VWC|{y_target_type}|D+1')

    # 3 days ahead prediction
    feature_groups_3_days_ahead = {
        "past_raw_vwc_min_values": ['-10|ENV__SOIL__VWC|min', '-10|ENV__SOIL__VWC|min|D-6', '-10|ENV__SOIL__VWC|min|D-5', '-10|ENV__SOIL__VWC|min|D-4', '-10|ENV__SOIL__VWC|min|D-3', '-10|ENV__SOIL__VWC|min|D-2', '-10|ENV__SOIL__VWC|min|D-1'],
        "past_raw_vwc_max_values" : ['-10|ENV__SOIL__VWC|max', '-10|ENV__SOIL__VWC|max|D-6', '-10|ENV__SOIL__VWC|max|D-5', '-10|ENV__SOIL__VWC|max|D-4', '-10|ENV__SOIL__VWC|max|D-3', '-10|ENV__SOIL__VWC|max|D-2', '-10|ENV__SOIL__VWC|max|D-1'],
        "past_raw_vwc_close_values": ['-10|ENV__SOIL__VWC|close', '-10|ENV__SOIL__VWC|close|D-6', '-10|ENV__SOIL__VWC|close|D-5', '-10|ENV__SOIL__VWC|close|D-4', '-10|ENV__SOIL__VWC|close|D-3', '-10|ENV__SOIL__VWC|close|D-2', '-10|ENV__SOIL__VWC|close|D-1'],
        "past_vwc_min_moving_diff": ['-10|ENV__SOIL__VWC|min|Moving_diff', '-10|ENV__SOIL__VWC|min|Moving_diff_-1', '-10|ENV__SOIL__VWC|min|Moving_diff_-2', '-10|ENV__SOIL__VWC|min|Moving_diff_-3', '-10|ENV__SOIL__VWC|min|Moving_diff_-4', '-10|ENV__SOIL__VWC|min|Moving_diff_-5'],
        "past_vwc_max_moving_diff": ['-10|ENV__SOIL__VWC|max|Moving_diff', '-10|ENV__SOIL__VWC|max|Moving_diff_-1', '-10|ENV__SOIL__VWC|max|Moving_diff_-2', '-10|ENV__SOIL__VWC|max|Moving_diff_-3', '-10|ENV__SOIL__VWC|max|Moving_diff_-4', '-10|ENV__SOIL__VWC|max|Moving_diff_-5'],
        "past_vwc_min_current_diff": ['-10|ENV__SOIL__VWC|min|Today_D-1_diff', '-10|ENV__SOIL__VWC|min|Today_D-2_diff', '-10|ENV__SOIL__VWC|min|Today_D-3_diff', '-10|ENV__SOIL__VWC|min|Today_D-4_diff', '-10|ENV__SOIL__VWC|min|Today_D-5_diff'],
        "past_vwc_max_current_diff": ['-10|ENV__SOIL__VWC|max|Today_D-1_diff', '-10|ENV__SOIL__VWC|max|Today_D-2_diff', '-10|ENV__SOIL__VWC|max|Today_D-3_diff', '-10|ENV__SOIL__VWC|max|Today_D-4_diff', '-10|ENV__SOIL__VWC|max|Today_D-5_diff'],
        "past_vwc_close_current_diff": ['-10|ENV__SOIL__VWC|close|Today_D-1_diff', '-10|ENV__SOIL__VWC|close|Today_D-2_diff', '-10|ENV__SOIL__VWC|close|Today_D-3_diff', '-10|ENV__SOIL__VWC|close|Today_D-4_diff', '-10|ENV__SOIL__VWC|close|Today_D-5_diff'],

        "past_rain_sum": ['precipitation|sum', 'precipitation|sum|D-1', 'precipitation|sum|D-2', 'precipitation|sum|D-3', 'precipitation|sum|D-4', 'precipitation|sum|D-5', 'precipitation|sum|D-6'],
        "past_acc_rain_sum": ['precipitation|sum|7_D_acc'],
        "lots_of_rain": ["precipitation|lots_of_rain", "precipitation|lots_of_rain|D-1", "precipitation|lots_of_rain|D-2", "precipitation|lots_of_rain|D-3", "precipitation|lots_of_rain|D-4", "precipitation|lots_of_rain|D-5", "precipitation|lots_of_rain|D-6", "precipitation|lots_of_rain|D+1", "precipitation|lots_of_rain|D+2", "precipitation|lots_of_rain|D+3"],
        "future_rain_sum": ['precipitation|sum|D+1', 'precipitation|sum|D+2', 'precipitation|sum|D+3', 'precipitation|sum|D+D+1', 'precipitation|sum|D+1+D+2', 'precipitation|sum|D+2+D+3'],
        
        "irrigations": ['DOC_IRRIGATION_PREDICTION|max', 'DOC_IRRIGATION_PREDICTION|max|D-1', 'DOC_IRRIGATION_PREDICTION|max|D-2', 'DOC_IRRIGATION_PREDICTION|max|D-3', 'DOC_IRRIGATION_PREDICTION|max|D-4', 'DOC_IRRIGATION_PREDICTION|max|D-5', 'DOC_IRRIGATION_PREDICTION|max|D-6'],
        "future_irrigations": ['DOC_IRRIGATION_PREDICTION|max', 'DOC_IRRIGATION_PREDICTION|max|D+1', 'DOC_IRRIGATION_PREDICTION|max|D+2', 'DOC_IRRIGATION_PREDICTION|max|D+3'],
        "past_acc_irrigations": ['DOC_IRRIGATION_PREDICTION|max|7_D_acc'],
        
        "seasonality": ['Month', 'Season', 'Irrigation_Season'],

        "past_tree_temp": ['-10|ENV__SOIL__T|mean', '-10|ENV__SOIL__T|mean|D-1', '-10|ENV__SOIL__T|mean|D-2', '-10|ENV__SOIL__T|mean|D-3', '-10|ENV__SOIL__T|mean|D-4', '-10|ENV__SOIL__T|mean|D-5', '-10|ENV__SOIL__T|mean|D-6'],
        "past_tree_temp_moving_diff": ['-10|ENV__SOIL__T|mean|Moving_diff', '-10|ENV__SOIL__T|mean|Moving_diff_-1', '-10|ENV__SOIL__T|mean|Moving_diff_-2', '-10|ENV__SOIL__T|mean|Moving_diff_-3', '-10|ENV__SOIL__T|mean|Moving_diff_-4', '-10|ENV__SOIL__T|mean|Moving_diff_-5'],
        "past_tree_temp_current_diff": ['-10|ENV__SOIL__T|mean|Today_D-1_diff', '-10|ENV__SOIL__T|mean|Today_D-2_diff', '-10|ENV__SOIL__T|mean|Today_D-3_diff', '-10|ENV__SOIL__T|mean|Today_D-4_diff', '-10|ENV__SOIL__T|mean|Today_D-5_diff'],
        
        "past_weather_temp_mean": ['temperature_2m|mean', 'temperature_2m|mean', 'temperature_2m|mean|D-1', 'temperature_2m|mean|D-2', 'temperature_2m|mean|D-3', 'temperature_2m|mean|D-4', 'temperature_2m|mean|D-5', 'temperature_2m|mean|D-6'],
        "past_weather_temp_max": ['temperature_2m|max', 'temperature_2m|max|D-1', 'temperature_2m|max|D-2', 'temperature_2m|max|D-3', 'temperature_2m|max|D-4', 'temperature_2m|max|D-5', 'temperature_2m|max|D-6'],
        "past_weather_temp_mean_moving_diff": ['temperature_2m|mean|Moving_diff', 'temperature_2m|mean|Moving_diff_-1', 'temperature_2m|mean|Moving_diff_-2', 'temperature_2m|mean|Moving_diff_-3', 'temperature_2m|mean|Moving_diff_-4', 'temperature_2m|mean|Moving_diff_-5'],
        "past_weather_temp_max_moving_diff": ['temperature_2m|max|Moving_diff', 'temperature_2m|max|Moving_diff_-1', 'temperature_2m|max|Moving_diff_-2', 'temperature_2m|max|Moving_diff_-3', 'temperature_2m|max|Moving_diff_-4', 'temperature_2m|max|Moving_diff_-5'],
        "future_weather_temp_mean": ['temperature_2m|mean', 'temperature_2m|mean|D+1', 'temperature_2m|mean|D+2', 'temperature_2m|mean|D+3'],
        "future_weather_temp_max": ['temperature_2m|max', 'temperature_2m|max|D+1', 'temperature_2m|max|D+2', 'temperature_2m|max|D+3'],

        "future_radiation": ['direct_radiation|sum', 'direct_radiation|sum|D+1', 'direct_radiation|sum|D+2', 'direct_radiation|sum|D+3'],
        "past_radiation": ['direct_radiation|sum', 'direct_radiation|sum|D-1', 'direct_radiation|sum|D-2', 'direct_radiation|sum|D-3', 'direct_radiation|sum|D-4', 'direct_radiation|sum|D-5', 'direct_radiation|sum|D-6'],
        "past_acc_radiation": ['direct_radiation|sum|7_D_acc'],
        
        "past_humdity": ['relative_humidity_2m|mean', 'relative_humidity_2m|mean|D-1', 'relative_humidity_2m|mean|D-2', 'relative_humidity_2m|mean|D-3', 'relative_humidity_2m|mean|D-4', 'relative_humidity_2m|mean|D-5', 'relative_humidity_2m|mean|D-6'],
        "future_humidity": ['relative_humidity_2m|mean', 'relative_humidity_2m|mean|D+1', 'relative_humidity_2m|mean|D+2', 'relative_humidity_2m|mean|D+3'],
    }
    if data_granularity == 'h':
        feature_groups_3_days_ahead.update(feature_groups_hourly)


    y_targets_3_days_ahead = []#['-10|ENV__SOIL__VWC|min|D+1', '-10|ENV__SOIL__VWC|min|D+2', '-10|ENV__SOIL__VWC|min|D+3', '-10|ENV__SOIL__VWC|max|D+1', '-10|ENV__SOIL__VWC|max|D+2', '-10|ENV__SOIL__VWC|max|D+3', '-10|ENV__SOIL__VWC|close|D+1', '-10|ENV__SOIL__VWC|close|D+2', '-10|ENV__SOIL__VWC|close|D+3']
    for y_target_type in y_target_types:
        for i in range(1, num_days_ahead_pred+1):
            y_targets_3_days_ahead.append(f'-10|ENV__SOIL__VWC|{y_target_type}|D+{i}')

    # 5 days ahead prediction
    feature_groups_5_days_ahead = {
        "past_raw_vwc_min_values": ['-10|ENV__SOIL__VWC|min', '-10|ENV__SOIL__VWC|min|D-6', '-10|ENV__SOIL__VWC|min|D-5', '-10|ENV__SOIL__VWC|min|D-4', '-10|ENV__SOIL__VWC|min|D-3', '-10|ENV__SOIL__VWC|min|D-2', '-10|ENV__SOIL__VWC|min|D-1'],
        "past_raw_vwc_max_values" : ['-10|ENV__SOIL__VWC|max', '-10|ENV__SOIL__VWC|max|D-6', '-10|ENV__SOIL__VWC|max|D-5', '-10|ENV__SOIL__VWC|max|D-4', '-10|ENV__SOIL__VWC|max|D-3', '-10|ENV__SOIL__VWC|max|D-2', '-10|ENV__SOIL__VWC|max|D-1'],
        "past_raw_vwc_close_values": ['-10|ENV__SOIL__VWC|close', '-10|ENV__SOIL__VWC|close|D-6', '-10|ENV__SOIL__VWC|close|D-5', '-10|ENV__SOIL__VWC|close|D-4', '-10|ENV__SOIL__VWC|close|D-3', '-10|ENV__SOIL__VWC|close|D-2', '-10|ENV__SOIL__VWC|close|D-1'],
        "past_vwc_min_moving_diff": ['-10|ENV__SOIL__VWC|min|Moving_diff', '-10|ENV__SOIL__VWC|min|Moving_diff_-1', '-10|ENV__SOIL__VWC|min|Moving_diff_-2', '-10|ENV__SOIL__VWC|min|Moving_diff_-3', '-10|ENV__SOIL__VWC|min|Moving_diff_-4', '-10|ENV__SOIL__VWC|min|Moving_diff_-5'],
        "past_vwc_max_moving_diff": ['-10|ENV__SOIL__VWC|max|Moving_diff', '-10|ENV__SOIL__VWC|max|Moving_diff_-1', '-10|ENV__SOIL__VWC|max|Moving_diff_-2', '-10|ENV__SOIL__VWC|max|Moving_diff_-3', '-10|ENV__SOIL__VWC|max|Moving_diff_-4', '-10|ENV__SOIL__VWC|max|Moving_diff_-5'],

        "past_vwc_min_current_diff": ['-10|ENV__SOIL__VWC|min|Today_D-1_diff', '-10|ENV__SOIL__VWC|min|Today_D-2_diff', '-10|ENV__SOIL__VWC|min|Today_D-3_diff', '-10|ENV__SOIL__VWC|min|Today_D-4_diff', '-10|ENV__SOIL__VWC|min|Today_D-5_diff'],
        "past_vwc_max_current_diff": ['-10|ENV__SOIL__VWC|max|Today_D-1_diff', '-10|ENV__SOIL__VWC|max|Today_D-2_diff', '-10|ENV__SOIL__VWC|max|Today_D-3_diff', '-10|ENV__SOIL__VWC|max|Today_D-4_diff', '-10|ENV__SOIL__VWC|max|Today_D-5_diff'],
        "past_vwc_close_current_diff": ['-10|ENV__SOIL__VWC|close|Today_D-1_diff', '-10|ENV__SOIL__VWC|close|Today_D-2_diff', '-10|ENV__SOIL__VWC|close|Today_D-3_diff', '-10|ENV__SOIL__VWC|close|Today_D-4_diff', '-10|ENV__SOIL__VWC|close|Today_D-5_diff'],

        "past_rain_sum": ['precipitation|sum', 'precipitation|sum|D-1', 'precipitation|sum|D-2', 'precipitation|sum|D-3', 'precipitation|sum|D-4', 'precipitation|sum|D-5', 'precipitation|sum|D-6'],
        "past_acc_rain_sum": ['precipitation|sum|7_D_acc'],
        "lots_of_rain": ["precipitation|lots_of_rain", "precipitation|lots_of_rain|D-1", "precipitation|lots_of_rain|D-2", "precipitation|lots_of_rain|D-3", "precipitation|lots_of_rain|D-4", "precipitation|lots_of_rain|D-5", "precipitation|lots_of_rain|D-6", "precipitation|lots_of_rain|D+1", "precipitation|lots_of_rain|D+2", "precipitation|lots_of_rain|D+3", "precipitation|lots_of_rain|D+4", "precipitation|lots_of_rain|D+5"],
        "future_rain_sum": ['precipitation|sum|D+1', 'precipitation|sum|D+2', 'precipitation|sum|D+3', 'precipitation|sum|D+4', 'precipitation|sum|D+5',  'precipitation|sum|D+D+1', 'precipitation|sum|D+1+D+2', 'precipitation|sum|D+2+D+3', 'precipitation|sum|D+3+D+4', 'precipitation|sum|D+4+D+5'],
        
        "irrigations": ['DOC_IRRIGATION_PREDICTION|max', 'DOC_IRRIGATION_PREDICTION|max|D-1', 'DOC_IRRIGATION_PREDICTION|max|D-2', 'DOC_IRRIGATION_PREDICTION|max|D-3', 'DOC_IRRIGATION_PREDICTION|max|D-4', 'DOC_IRRIGATION_PREDICTION|max|D-5', 'DOC_IRRIGATION_PREDICTION|max|D-6'],
        "future_irrigations": ['DOC_IRRIGATION_PREDICTION|max', 'DOC_IRRIGATION_PREDICTION|max|D+1', 'DOC_IRRIGATION_PREDICTION|max|D+2', 'DOC_IRRIGATION_PREDICTION|max|D+3', 'DOC_IRRIGATION_PREDICTION|max|D+4', 'DOC_IRRIGATION_PREDICTION|max|D+5'],
        "past_acc_irrigations": ['DOC_IRRIGATION_PREDICTION|max|7_D_acc'],
        
        "seasonality": ['Month', 'Season', 'Irrigation_Season'],

        "past_tree_temp": ['-10|ENV__SOIL__T|mean', '-10|ENV__SOIL__T|mean|D-1', '-10|ENV__SOIL__T|mean|D-2', '-10|ENV__SOIL__T|mean|D-3', '-10|ENV__SOIL__T|mean|D-4', '-10|ENV__SOIL__T|mean|D-5', '-10|ENV__SOIL__T|mean|D-6'],
        "past_tree_temp_moving_diff": ['-10|ENV__SOIL__T|mean|Moving_diff', '-10|ENV__SOIL__T|mean|Moving_diff_-1', '-10|ENV__SOIL__T|mean|Moving_diff_-2', '-10|ENV__SOIL__T|mean|Moving_diff_-3', '-10|ENV__SOIL__T|mean|Moving_diff_-4', '-10|ENV__SOIL__T|mean|Moving_diff_-5'],
        "past_tree_temp_current_diff": ['-10|ENV__SOIL__T|mean|Today_D-1_diff', '-10|ENV__SOIL__T|mean|Today_D-2_diff', '-10|ENV__SOIL__T|mean|Today_D-3_diff', '-10|ENV__SOIL__T|mean|Today_D-4_diff', '-10|ENV__SOIL__T|mean|Today_D-5_diff'],
        
        "past_weather_temp_mean": ['temperature_2m|mean', 'temperature_2m|mean', 'temperature_2m|mean|D-1', 'temperature_2m|mean|D-2', 'temperature_2m|mean|D-3', 'temperature_2m|mean|D-4', 'temperature_2m|mean|D-5', 'temperature_2m|mean|D-6'],
        "past_weather_temp_max": ['temperature_2m|max', 'temperature_2m|max|D-1', 'temperature_2m|max|D-2', 'temperature_2m|max|D-3', 'temperature_2m|max|D-4', 'temperature_2m|max|D-5', 'temperature_2m|max|D-6'],
        "past_weather_temp_mean_moving_diff": ['temperature_2m|mean|Moving_diff', 'temperature_2m|mean|Moving_diff_-1', 'temperature_2m|mean|Moving_diff_-2', 'temperature_2m|mean|Moving_diff_-3', 'temperature_2m|mean|Moving_diff_-4', 'temperature_2m|mean|Moving_diff_-5'],
        "past_weather_temp_max_moving_diff": ['temperature_2m|max|Moving_diff', 'temperature_2m|max|Moving_diff_-1', 'temperature_2m|max|Moving_diff_-2', 'temperature_2m|max|Moving_diff_-3', 'temperature_2m|max|Moving_diff_-4', 'temperature_2m|max|Moving_diff_-5'],
        "future_weather_temp_mean": ['temperature_2m|mean', 'temperature_2m|mean|D+1', 'temperature_2m|mean|D+2', 'temperature_2m|mean|D+3', 'temperature_2m|mean|D+4', 'temperature_2m|mean|D+5'],
        "future_weather_temp_max": ['temperature_2m|max', 'temperature_2m|max|D+1', 'temperature_2m|max|D+2', 'temperature_2m|max|D+3', 'temperature_2m|max|D+4', 'temperature_2m|max|D+5'],

        "future_radiation": ['direct_radiation|sum', 'direct_radiation|sum|D+1', 'direct_radiation|sum|D+2', 'direct_radiation|sum|D+3', 'direct_radiation|sum|D+4', 'direct_radiation|sum|D+5'],
        "past_radiation": ['direct_radiation|sum', 'direct_radiation|sum|D-1', 'direct_radiation|sum|D-2', 'direct_radiation|sum|D-3', 'direct_radiation|sum|D-4', 'direct_radiation|sum|D-5', 'direct_radiation|sum|D-6'],
        "past_acc_radiation": ['direct_radiation|sum|7_D_acc'],
        
        "past_humdity": ['relative_humidity_2m|mean', 'relative_humidity_2m|mean|D-1', 'relative_humidity_2m|mean|D-2', 'relative_humidity_2m|mean|D-3', 'relative_humidity_2m|mean|D-4', 'relative_humidity_2m|mean|D-5', 'relative_humidity_2m|mean|D-6'],
        "future_humidity": ['relative_humidity_2m|mean', 'relative_humidity_2m|mean|D+1', 'relative_humidity_2m|mean|D+2', 'relative_humidity_2m|mean|D+3', 'relative_humidity_2m|mean|D+4', 'relative_humidity_2m|mean|D+5'],
    }
    if data_granularity == 'h':
        feature_groups_5_days_ahead.update(feature_groups_hourly)


    y_targets_5_days_ahead = []#['-10|ENV__SOIL__VWC|min|D+1', '-10|ENV__SOIL__VWC|min|D+2', '-10|ENV__SOIL__VWC|min|D+3', '-10|ENV__SOIL__VWC|min|D+4', '-10|ENV__SOIL__VWC|min|D+5', '-10|ENV__SOIL__VWC|max|D+1', '-10|ENV__SOIL__VWC|max|D+2', '-10|ENV__SOIL__VWC|max|D+3', '-10|ENV__SOIL__VWC|max|D+4', '-10|ENV__SOIL__VWC|max|D+5', '-10|ENV__SOIL__VWC|close|D+1', '-10|ENV__SOIL__VWC|close|D+2', '-10|ENV__SOIL__VWC|close|D+3', '-10|ENV__SOIL__VWC|close|D+4', '-10|ENV__SOIL__VWC|close|D+5']
    for y_target_type in y_target_types:
        for i in range(1, num_days_ahead_pred+1):
            y_targets_5_days_ahead.append(f'-10|ENV__SOIL__VWC|{y_target_type}|D+{i}')

    # 7 days ahead prediction
    feature_groups_7_days_ahead = {
        "past_raw_vwc_min_values": ['-10|ENV__SOIL__VWC|min', '-10|ENV__SOIL__VWC|min|D-6', '-10|ENV__SOIL__VWC|min|D-5', '-10|ENV__SOIL__VWC|min|D-4', '-10|ENV__SOIL__VWC|min|D-3', '-10|ENV__SOIL__VWC|min|D-2', '-10|ENV__SOIL__VWC|min|D-1'],
        "past_raw_vwc_max_values" : ['-10|ENV__SOIL__VWC|max', '-10|ENV__SOIL__VWC|max|D-6', '-10|ENV__SOIL__VWC|max|D-5', '-10|ENV__SOIL__VWC|max|D-4', '-10|ENV__SOIL__VWC|max|D-3', '-10|ENV__SOIL__VWC|max|D-2', '-10|ENV__SOIL__VWC|max|D-1'],
        "past_raw_vwc_close_values": ['-10|ENV__SOIL__VWC|close', '-10|ENV__SOIL__VWC|close|D-6', '-10|ENV__SOIL__VWC|close|D-5', '-10|ENV__SOIL__VWC|close|D-4', '-10|ENV__SOIL__VWC|close|D-3', '-10|ENV__SOIL__VWC|close|D-2', '-10|ENV__SOIL__VWC|close|D-1'],
        "past_vwc_min_moving_diff": ['-10|ENV__SOIL__VWC|min|Moving_diff', '-10|ENV__SOIL__VWC|min|Moving_diff_-1', '-10|ENV__SOIL__VWC|min|Moving_diff_-2', '-10|ENV__SOIL__VWC|min|Moving_diff_-3', '-10|ENV__SOIL__VWC|min|Moving_diff_-4', '-10|ENV__SOIL__VWC|min|Moving_diff_-5'],
        "past_vwc_max_moving_diff": ['-10|ENV__SOIL__VWC|max|Moving_diff', '-10|ENV__SOIL__VWC|max|Moving_diff_-1', '-10|ENV__SOIL__VWC|max|Moving_diff_-2', '-10|ENV__SOIL__VWC|max|Moving_diff_-3', '-10|ENV__SOIL__VWC|max|Moving_diff_-4', '-10|ENV__SOIL__VWC|max|Moving_diff_-5'],
        "past_vwc_min_current_diff": ['-10|ENV__SOIL__VWC|min|Today_D-1_diff', '-10|ENV__SOIL__VWC|min|Today_D-2_diff', '-10|ENV__SOIL__VWC|min|Today_D-3_diff', '-10|ENV__SOIL__VWC|min|Today_D-4_diff', '-10|ENV__SOIL__VWC|min|Today_D-5_diff'],
        "past_vwc_max_current_diff": ['-10|ENV__SOIL__VWC|max|Today_D-1_diff', '-10|ENV__SOIL__VWC|max|Today_D-2_diff', '-10|ENV__SOIL__VWC|max|Today_D-3_diff', '-10|ENV__SOIL__VWC|max|Today_D-4_diff', '-10|ENV__SOIL__VWC|max|Today_D-5_diff'],
        "past_vwc_close_current_diff": ['-10|ENV__SOIL__VWC|close|Today_D-1_diff', '-10|ENV__SOIL__VWC|close|Today_D-2_diff', '-10|ENV__SOIL__VWC|close|Today_D-3_diff', '-10|ENV__SOIL__VWC|close|Today_D-4_diff', '-10|ENV__SOIL__VWC|close|Today_D-5_diff'],

        "past_rain_sum": ['precipitation|sum', 'precipitation|sum|D-1', 'precipitation|sum|D-2', 'precipitation|sum|D-3', 'precipitation|sum|D-4', 'precipitation|sum|D-5', 'precipitation|sum|D-6'],
        "past_acc_rain_sum": ['precipitation|sum|7_D_acc'],
        "lots_of_rain": ["precipitation|lots_of_rain", "precipitation|lots_of_rain|D-1", "precipitation|lots_of_rain|D-2", "precipitation|lots_of_rain|D-3", "precipitation|lots_of_rain|D-4", "precipitation|lots_of_rain|D-5", "precipitation|lots_of_rain|D-6", "precipitation|lots_of_rain|D+1", "precipitation|lots_of_rain|D+2", "precipitation|lots_of_rain|D+3", "precipitation|lots_of_rain|D+4", "precipitation|lots_of_rain|D+5", "precipitation|lots_of_rain|D+6", "precipitation|lots_of_rain|D+7"],
        "future_rain_sum": ['precipitation|sum|D+1', 'precipitation|sum|D+2', 'precipitation|sum|D+3', 'precipitation|sum|D+4', 'precipitation|sum|D+5', 'precipitation|sum|D+6', 'precipitation|sum|D+7', 'precipitation|sum|D+D+1', 'precipitation|sum|D+1+D+2', 'precipitation|sum|D+2+D+3', 'precipitation|sum|D+3+D+4', 'precipitation|sum|D+4+D+5', 'precipitation|sum|D+5+D+6', 'precipitation|sum|D+6+D+7'],
        
        "irrigations": ['DOC_IRRIGATION_PREDICTION|max', 'DOC_IRRIGATION_PREDICTION|max|D-1', 'DOC_IRRIGATION_PREDICTION|max|D-2', 'DOC_IRRIGATION_PREDICTION|max|D-3', 'DOC_IRRIGATION_PREDICTION|max|D-4', 'DOC_IRRIGATION_PREDICTION|max|D-5', 'DOC_IRRIGATION_PREDICTION|max|D-6'],
        "future_irrigations": ['DOC_IRRIGATION_PREDICTION|max', 'DOC_IRRIGATION_PREDICTION|max|D+1', 'DOC_IRRIGATION_PREDICTION|max|D+2', 'DOC_IRRIGATION_PREDICTION|max|D+3', 'DOC_IRRIGATION_PREDICTION|max|D+4', 'DOC_IRRIGATION_PREDICTION|max|D+5', 'DOC_IRRIGATION_PREDICTION|max|D+6', 'DOC_IRRIGATION_PREDICTION|max|D+7'],
        "past_acc_irrigations": ['DOC_IRRIGATION_PREDICTION|max|7_D_acc'],
        
        "seasonality": ['Month', 'Season', 'Irrigation_Season'],

        "past_tree_temp": ['-10|ENV__SOIL__T|mean', '-10|ENV__SOIL__T|mean|D-1', '-10|ENV__SOIL__T|mean|D-2', '-10|ENV__SOIL__T|mean|D-3', '-10|ENV__SOIL__T|mean|D-4', '-10|ENV__SOIL__T|mean|D-5', '-10|ENV__SOIL__T|mean|D-6'],
        "past_tree_temp_moving_diff": ['-10|ENV__SOIL__T|mean|Moving_diff', '-10|ENV__SOIL__T|mean|Moving_diff_-1', '-10|ENV__SOIL__T|mean|Moving_diff_-2', '-10|ENV__SOIL__T|mean|Moving_diff_-3', '-10|ENV__SOIL__T|mean|Moving_diff_-4', '-10|ENV__SOIL__T|mean|Moving_diff_-5'],
        "past_tree_temp_current_diff": ['-10|ENV__SOIL__T|mean|Today_D-1_diff', '-10|ENV__SOIL__T|mean|Today_D-2_diff', '-10|ENV__SOIL__T|mean|Today_D-3_diff', '-10|ENV__SOIL__T|mean|Today_D-4_diff', '-10|ENV__SOIL__T|mean|Today_D-5_diff'],
        
        "past_weather_temp_mean": ['temperature_2m|mean', 'temperature_2m|mean', 'temperature_2m|mean|D-1', 'temperature_2m|mean|D-2', 'temperature_2m|mean|D-3', 'temperature_2m|mean|D-4', 'temperature_2m|mean|D-5', 'temperature_2m|mean|D-6'],
        "past_weather_temp_max": ['temperature_2m|max', 'temperature_2m|max|D-1', 'temperature_2m|max|D-2', 'temperature_2m|max|D-3', 'temperature_2m|max|D-4', 'temperature_2m|max|D-5', 'temperature_2m|max|D-6'],
        "past_weather_temp_mean_moving_diff": ['temperature_2m|mean|Moving_diff', 'temperature_2m|mean|Moving_diff_-1', 'temperature_2m|mean|Moving_diff_-2', 'temperature_2m|mean|Moving_diff_-3', 'temperature_2m|mean|Moving_diff_-4', 'temperature_2m|mean|Moving_diff_-5'],
        "past_weather_temp_max_moving_diff": ['temperature_2m|max|Moving_diff', 'temperature_2m|max|Moving_diff_-1', 'temperature_2m|max|Moving_diff_-2', 'temperature_2m|max|Moving_diff_-3', 'temperature_2m|max|Moving_diff_-4', 'temperature_2m|max|Moving_diff_-5'],
        "future_weather_temp_mean": ['temperature_2m|mean', 'temperature_2m|mean|D+1', 'temperature_2m|mean|D+2', 'temperature_2m|mean|D+3', 'temperature_2m|mean|D+4', 'temperature_2m|mean|D+5', 'temperature_2m|mean|D+6', 'temperature_2m|mean|D+7'],
        "future_weather_temp_max": ['temperature_2m|max', 'temperature_2m|max|D+1', 'temperature_2m|max|D+2', 'temperature_2m|max|D+3', 'temperature_2m|max|D+4', 'temperature_2m|max|D+5', 'temperature_2m|max|D+6', 'temperature_2m|max|D+7'],
        
        "future_radiation": ['direct_radiation|sum', 'direct_radiation|sum|D+1', 'direct_radiation|sum|D+2', 'direct_radiation|sum|D+3', 'direct_radiation|sum|D+4', 'direct_radiation|sum|D+5', 'direct_radiation|sum|D+6', 'direct_radiation|sum|D+7'],
        "past_radiation": ['direct_radiation|sum', 'direct_radiation|sum|D-1', 'direct_radiation|sum|D-2', 'direct_radiation|sum|D-3', 'direct_radiation|sum|D-4', 'direct_radiation|sum|D-5', 'direct_radiation|sum|D-6'],
        "past_acc_radiation": ['direct_radiation|sum|7_D_acc'],
        
        "past_humdity": ['relative_humidity_2m|mean', 'relative_humidity_2m|mean|D-1', 'relative_humidity_2m|mean|D-2', 'relative_humidity_2m|mean|D-3', 'relative_humidity_2m|mean|D-4', 'relative_humidity_2m|mean|D-5', 'relative_humidity_2m|mean|D-6'],
        "future_humidity": ['relative_humidity_2m|mean', 'relative_humidity_2m|mean|D+1', 'relative_humidity_2m|mean|D+2', 'relative_humidity_2m|mean|D+3', 'relative_humidity_2m|mean|D+4', 'relative_humidity_2m|mean|D+5', 'relative_humidity_2m|mean|D+6', 'relative_humidity_2m|mean|D+7'],
    }
    if data_granularity == 'h':
        feature_groups_7_days_ahead.update(feature_groups_hourly)


    y_targets_7_days_ahead = []#['-10|ENV__SOIL__VWC|min|D+1', '-10|ENV__SOIL__VWC|min|D+2', '-10|ENV__SOIL__VWC|min|D+3', '-10|ENV__SOIL__VWC|min|D+4', '-10|ENV__SOIL__VWC|min|D+5', '-10|ENV__SOIL__VWC|min|D+6', '-10|ENV__SOIL__VWC|min|D+7', '-10|ENV__SOIL__VWC|max|D+1', '-10|ENV__SOIL__VWC|max|D+2', '-10|ENV__SOIL__VWC|max|D+3', '-10|ENV__SOIL__VWC|max|D+4', '-10|ENV__SOIL__VWC|max|D+5', '-10|ENV__SOIL__VWC|max|D+6', '-10|ENV__SOIL__VWC|max|D+7', '-10|ENV__SOIL__VWC|close|D+1', '-10|ENV__SOIL__VWC|close|D+2', '-10|ENV__SOIL__VWC|close|D+3', '-10|ENV__SOIL__VWC|close|D+4', '-10|ENV__SOIL__VWC|close|D+5', '-10|ENV__SOIL__VWC|close|D+6', '-10|ENV__SOIL__VWC|close|D+7']
    for y_target_type in y_target_types:
        for i in range(1, num_days_ahead_pred+1):
            y_targets_7_days_ahead.append(f'-10|ENV__SOIL__VWC|{y_target_type}|D+{i}')

    if num_days_ahead_pred == 1:
        return feature_groups_1_day_ahead, y_targets_1_day_ahead
    elif num_days_ahead_pred == 3:
        return feature_groups_3_days_ahead, y_targets_3_days_ahead
    elif num_days_ahead_pred == 5:
        return feature_groups_5_days_ahead, y_targets_5_days_ahead
    elif num_days_ahead_pred == 7:
        return feature_groups_7_days_ahead, y_targets_7_days_ahead
    else:
        raise ValueError("Invalid number of days ahead prediction")