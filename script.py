import json
import time

import pandas as pd
from poloniex import Poloniex
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Data frame column names
CN_HIGH = 'high'
CN_LOW = 'low'
CN_HILO = 'hilo'
CN_HILO_2D = 'hilo2d'
CN_HILO_7D = 'hilo7d'
CN_HILO_14D = 'hilo14d'
CN_LABEL = 'label'
CN_NEXT_PCT_CHANGE = 'next_4hr_pct_change'
CN_FUTURE_PRICE_WA = 'next_week_wa'
CN_WEIGHTED_AVERAGE = 'weightedAverage'
CN_WA_AVG_1D = 'wa_avg_1d'
CN_WA_AVG_3200 = 'wa_avg_3200'
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

    df[CN_HILO_2D] = df[CN_LOW].rolling(
        48 * 2).mean() / df[CN_HIGH].rolling(48 * 2).mean()

    df[CN_HILO_7D] = df[CN_LOW].rolling(
    periods_in_14d = PERIODS_IN_7D * 2        
    df[CN_HILO_14D] = df[CN_LOW].rolling(
        periods_in_14d).mean() / df[CN_HIGH].rolling(periods_in_14d).mean()

    df['date'] = pd.to_datetime(df['date'], unit='s')
    weighted_averages_ = df[CN_WEIGHTED_AVERAGE]
    df[CN_WA_AVG_1D] = weighted_averages_.rolling(2 * 24).mean()
    df['wa_moving_200'] = weighted_averages_.rolling(200).mean()
    df[CN_WA_AVG_6400] = df[CN_WA_AVG_1D] / \
        weighted_averages_.rolling(6400).max()
    df[CN_WA_AVG_3200] = df[CN_WA_AVG_1D] / \
        weighted_averages_.rolling(3200).max()
    return df


def label_data(df):
    """
    Data passed will be split into two basic groups, depending on whether the
    price at a given future time is higher or lower than the current weighted
    average.
    """
    df[CN_FUTURE_PRICE_WA] = df[CN_WA_AVG_6400].shift(-1 * PERIODS_IN_7D)
    df[CN_NEXT_PCT_CHANGE] = (
        df[CN_FUTURE_PRICE_WA] / df[CN_WA_AVG_6400]) - 1
    df.loc[df[CN_NEXT_PCT_CHANGE] >= 0, CN_LABEL] = 1
    df.loc[df[CN_NEXT_PCT_CHANGE] <= 0, CN_LABEL] = -1

    return df


def extract_inputs_from_data_frame(df):
    input_features = [
        #CN_STOCHASTIC_14D_K,
        CN_WA_AVG_3200,
        CN_WA_AVG_6400,
        #CN_QV400_MEAN_REVERSAL,
        CN_HILO_2D,
        CN_HILO_14D,
        #CN_HILO_30D
    ]

    return df[input_features].fillna(df[input_features].mean())


def classify_and_predict(df):
    df_clean = df.dropna()
    inputs = extract_inputs_from_data_frame(df_clean)
    labels = df_clean[CN_LABEL]
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(
        inputs, labels)

    #tree_classifier = DecisionTreeClassifier(max_depth=2)
    tree_classifier = RandomForestClassifier(max_depth=2, n_estimators=50)
    tree_classifier.fit(inputs_train, labels_train)

    df['cluster'] = tree_classifier.predict(extract_inputs_from_data_frame(df))

    tn, fp, fn, tp = confusion_matrix(
        labels_test, tree_classifier.predict(inputs_test)).ravel()
    test_accuracy = cross_val_score(tree_classifier, X=inputs, y=labels, cv=10).mean()
    predicted_future_probability = tree_classifier.predict_proba(
        extract_inputs_from_data_frame(df.iloc[-3:])).mean(axis=0)

    print(predicted_future_probability)  
    # weighted_predicted_future_probability = predicted_future_probability * test_accuracy

    buy_decision = True if predicted_future_probability[1] > 0.5 else False

    return {
        # 'predictions3': df['cluster'].rolling(3).mean().iloc[-1],
        #'last_probability': list(predicted_future_probability),
        'test_accuracy': test_accuracy,
        # 'true_negatives': tn,
        # 'true_positives': tp,
        # 'false_negatives': fn,
        # 'false_positives': fp,
        'timestamp': int(time.time()),
        'buy_decision': buy_decision
    }


def main():
    df = fetch_data()
    df = extend_stats(df)
    df = label_data(df)

    predicition_decision = classify_and_predict(df)
    print(predicition_decision)

    with open('decision.json', 'w') as fp:
        json.dump(predicition_decision, fp)


if __name__ == '__main__':
    main()
