import time

import pandas as pd
from poloniex import Poloniex
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Data frame column names
CN_HIGH = 'high'
CN_LOW = 'low'
CN_HILO = 'hilo'
CN_HILO_7D = 'hilo7d'
CN_LABEL = 'label'
CN_NEXT_PCT_CHANGE = 'next_4hr_pct_change'
CN_FUTURE_PRICE_WA = 'next_week_wa'
CN_WEIGHTED_AVERAGE = 'weightedAverage'
CN_WA_AVG_6400 = 'wa_avg_6400'

DURATION_24HR = 60 * 60 * 24
DURATION_7D = DURATION_24HR * 7

PERIOD = 1800
PERIODS_IN_7D = int(DURATION_7D / PERIOD)


def fetch_data():
    end = int(time.time())
    start = end - 100000 * 1800
    start, end

    polo = Poloniex()
    df = pd.DataFrame(
        polo.returnChartData("USDT_BTC", period=PERIOD, start=start, end=end))

    columns_to_convert_to_float = [CN_HIGH, CN_LOW, CN_WEIGHTED_AVERAGE]
    df[columns_to_convert_to_float] = df[columns_to_convert_to_float].astype(
        'float32')
    return df


def extend_stats(df):
    df[CN_HILO] = df[CN_LOW] / df[CN_HIGH]
    periods_in_14d = PERIODS_IN_7D * 2
    df[CN_HILO_7D] = df[CN_LOW].rolling(
        PERIODS_IN_7D).sum() / df[CN_HIGH].rolling(PERIODS_IN_7D).sum()

    df['date'] = pd.to_datetime(df['date'], unit='s')
    weighted_averages_ = df[CN_WEIGHTED_AVERAGE]
    df['wa_moving_50'] = weighted_averages_.rolling(50).mean()
    df['wa_moving_200'] = weighted_averages_.rolling(200).mean()
    df[CN_WA_AVG_6400] = weighted_averages_ / \
        weighted_averages_.rolling(6400).max()
    return df


def label_data(df):
    """
    Data passed will be split into two basic groups, depending on whether the
    price at a given future time is higher or lower than the current weighted
    average.
    """
    df[CN_FUTURE_PRICE_WA] = df['weightedAverage'].shift(-1 * PERIODS_IN_7D)
    df[CN_NEXT_PCT_CHANGE] = (
        df[CN_FUTURE_PRICE_WA] / df['weightedAverage']) - 1
    df.loc[df[CN_NEXT_PCT_CHANGE] >= 0, CN_LABEL] = 1
    df.loc[df[CN_NEXT_PCT_CHANGE] <= 0, CN_LABEL] = -1

    return df


def extract_inputs_from_data_frame(df):
    input_features = [
        # ph.CN_STOCHASTIC_14D_K,
        CN_WA_AVG_6400,
        # ph.CN_QV400_MEAN_REVERSAL,
        CN_HILO_7D
    ]

    return df[input_features].fillna(df[input_features].mean())


def classify_and_predict(df):
    df_clean = df.dropna()
    inputs = extract_inputs_from_data_frame(df_clean)
    labels = df_clean[CN_LABEL]
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(
        inputs, labels)

    tree_classifier = DecisionTreeClassifier(min_samples_leaf=100)
    # tree_classifier = RandomForestClassifier()
    tree_classifier.fit(inputs_train, labels_train)

    df['cluster'] = tree_classifier.predict(extract_inputs_from_data_frame(df))

    return {
        'predictions3': df['cluster'].rolling(3).mean().iloc[-1],
        'last_probability': tree_classifier.predict_proba(
            extract_inputs_from_data_frame(df.iloc[-1]).values.reshape(1, -1))[0],
        'test_accuracy': tree_classifier.score(inputs_test, labels_test, sample_weight=labels_test.abs())
    }


def main():
    df = fetch_data()
    df = extend_stats(df)
    df = label_data(df)

    feat = extract_inputs_from_data_frame(df)

    print(classify_and_predict(df))


if __name__ == '__main__':
    main()
