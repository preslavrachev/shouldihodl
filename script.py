import pandas as pd
import time
from poloniex import Poloniex

def fetch_data():
    end = int(time.time())
    start = end - 100000 * 1800
    start, end

    polo = Poloniex()
    df = pd.DataFrame(polo.returnChartData("USDT_BTC", period=1800, start=start, end=end))
    return df


def extend_stats(df):
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df['wa_moving_50'] = df['weightedAverage'].rolling(50).mean()
    df['wa_moving_200'] = df['weightedAverage'].rolling(200).mean()
    return df


def label_data(df):
    """
    Data passed will be split into two basic groups, depending on whether the
    50-data-point-ma is larger or smaller than the 200-data-point-ma of a future
    date. In the default case, this will be a week (7d) into the future.
    """
    pass


def main():
    df = fetch_data()
    df = extend_stats(df)

    print (df)

if __name__ == '__main__':
    main()
