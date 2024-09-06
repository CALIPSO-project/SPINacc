# =============================================================================================
# MLacc - Machine-Learning-based acceleration of spin-up
#
# Copyright Laboratoire des Sciences du Climat et de l'Environnement (LSCE)
#           Unite mixte CEA-CNRS-UVSQ
#
# Code manager:
# Daniel Goll, LSCE, <email>
#
# This software is developed by Yan Sun, Yilong Wang and Daniel Goll.......
#
# This software is governed by the XXX license
# XXXX <License content>
#
# =============================================================================================

from Tools import *
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor


def training_BAT(X, Y, logfile, loocv, alg):
    """
    Train a machine learning model using Balanced Augmentation Technique (BAT).

    Args:
        X (pandas.DataFrame): Input features.
        Y (pandas.DataFrame): Target variables.
        logfile (file): File object for logging.
        loocv (bool): Whether to perform leave-one-out cross-validation.
        alg (str): ML algorithm to use ("mlp" or "gbm").

    Returns:
        tuple:
            - model: Trained machine learning model.
            - predY (numpy.ndarray): Predicted Y values.
    """
    print("Data shapes: ", X.shape, Y.shape)

    # run the KMeans algorithm to find the cluster centers, and resample the data
    mod = KMeans(n_clusters=3)
    lab = mod.fit_predict(Y)
    count = Counter(lab)

    if min(count.values()) <= 5:
        mod = KMeans(n_clusters=2)
        lab = mod.fit_predict(Y)
        count = Counter(lab)

    check.display("Counter(lab):" + str(count), logfile)
        
    over_samples = SMOTE()
    over_samples_X, over_samples_y = over_samples.fit_resample(
        pd.concat([X, Y], axis=1), lab
    )
    check.display("Counter(over_samples_y):" + str(Counter(over_samples_y)), logfile)
    X = over_samples_X[X.columns]
    Y = over_samples_X[Y.columns]
    print("Data shapes after resampling: ", X.shape, Y.shape)

    if alg == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(32, 32),
            max_iter=100,
            learning_rate="invscaling",
            learning_rate_init=0.1,
            random_state=1000,
            verbose=True,
        )
    elif alg == "bt":
        model = MultiOutputRegressor(
            BaggingRegressor(
                DecisionTreeRegressor(random_state=1000),
                max_samples=0.8,
                n_estimators=300,
                random_state=1000
            )
        )
    elif alg == "rf":
        model = MultiOutputRegressor(
            RandomForestRegressor(
                max_samples=0.8,
                n_estimators=300,
                random_state=1000
            )
        )
    elif alg == "gbm":
        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                max_samples=0.8,
                random_state=1000
                verbose=3,
            )
        )
    else:
        raise ValueError("invalid ML algorithm name")

    model.fit(X, Y)
    predY = model.predict(X)

    return model, predY
