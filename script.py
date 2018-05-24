import pandas as pd
import time
from poloniex import Poloniex

# Data frame column names
CN_HIGH = 'high'
CN_LOW = 'low'
CN_LABEL = 'label'
CN_NEXT_PCT_CHANGE = 'next_4hr_pct_change'
CN_FUTURE_PRICE_WA = 'next_week_wa'
CN_WEIGHTED_AVERAGE = 'weightedAverage'


def fetch_data():
    end = int(time.time())
    start = end - 100000 * 1800
    start, end

    polo = Poloniex()
    df = pd.DataFrame(
        polo.returnChartData("USDT_BTC", period=1800, start=start, end=end))

    columns_to_convert_to_float = [CN_HIGH, CN_LOW, CN_WEIGHTED_AVERAGE]
    df[columns_to_convert_to_float] = df[columns_to_convert_to_float].astype(
        'float32')
    return df


def extend_stats(df):
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df['wa_moving_50'] = df['weightedAverage'].rolling(50).mean()
    df['wa_moving_200'] = df['weightedAverage'].rolling(200).mean()
    return df


def label_data(df):
    """
    Data passed will be split into two basic groups, depending on whether the
    price at a given future time is higher or lower than the current weighted
    average.
    """
    df[CN_FUTURE_PRICE_WA] = df['weightedAverage'].shift(-1 * 2 * 24 * 7)
    df[CN_NEXT_PCT_CHANGE] = (
        df[CN_FUTURE_PRICE_WA] / df['weightedAverage']) - 1
    df.loc[df[CN_NEXT_PCT_CHANGE] >= 0, CN_LABEL] = 1
    df.loc[df[CN_NEXT_PCT_CHANGE] <= 0, CN_LABEL] = -1

    return df


def main():
    df = fetch_data()
    df = extend_stats(df)
    df = label_data(df)

    print(df)


if __name__ == '__main__':
    main()
