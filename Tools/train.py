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


def training_BAT(XY_train, logfile, loocv=False, alg="gbm", upsample=False):
    """
    Train a machine learning model using Balanced Augmentation Technique (BAT).

    Args:
        XY_train (pandas.DataFrame): Training dataset.
        logfile (file): File object for logging.
        loocv (bool): Whether to perform leave-one-out cross-validation.
        alg (str): ML algorithm to use (options: "nn", "bt", "rf", "gbm").
        upsample (bool): Whether to upsample the dataset with SMOTE.

    Returns:
        tuple:
            - model: Trained machine learning model.
            - predY (numpy.ndarray): Predicted Y values.
    """
    Xtrain = XY_train.drop(columns="Y")
    Ytrain = XY_train["Y"]
    # labels=np.zeros(shape=(len(Ytrain),1))

    # if length of unique target is one
    if len(np.unique(Ytrain)) == 1:
        # return a set of default values and an empty TreeEnsemble
        TreeEns = []
        predY = Ytrain
        loocv_R2 = np.nan
        loocv_reMSE = np.nan
        loocv_slope = np.nan
        loocv_dNRMSE = np.nan
        loocv_sNRMSE = np.nan
        loocv_iNRMSE = np.nan
        loocv_f_SB = np.nan
        loocv_f_SDSD = np.nan
        loocv_f_LSC = np.nan
        return (
            TreeEns,
            predY,
            loocv_R2,
            loocv_reMSE,
            loocv_slope,
            loocv_dNRMSE,
            loocv_sNRMSE,
            loocv_iNRMSE,
            loocv_f_SB,
            loocv_f_SDSD,
            loocv_f_LSC,
        )

    # If the length of unique target variable is not 1,
    # run the KMeans algorithm to find the cluster centers, and resample the data
    if upsample:
        try:
            mod = KMeans(n_clusters=3)
            lab = mod.fit_predict(np.reshape(Ytrain, (-1, 1)))
            count = Counter(lab)
            check.display("Counter(lab):" + str(count), logfile)
            over_samples = SMOTE()
            over_samples_X, over_samples_y = over_samples.fit_resample(XY_train, lab)
            check.display(
                "Counter(over_samples_y):" + str(Counter(over_samples_y)), logfile
            )
            Xtrain = over_samples_X.iloc[:, 1:]
            Ytrain = over_samples_X.iloc[:, 0]
        except:
            mod = KMeans(n_clusters=2)
            lab = mod.fit_predict(np.reshape(Ytrain, (-1, 1)))
            count = Counter(lab)
            check.display("Counter(lab):" + str(Counter(lab)), logfile)
            # resample requires minimum number of a cluster >=6, if not, then repeat current samples
            for label, number in count.items():
                if number < 6:
                    XY_train = pd.concat(
                        (XY_train,)
                        + (XY_train[lab == label],) * int(np.ceil(6 / number) - 1)
                    )
                    lab = np.hstack(
                        (
                            lab,
                            np.repeat(lab[lab == label], int(np.ceil(6 / number) - 1)),
                        )
                    )
            check.display("Counter(lab):" + str(Counter(lab)), logfile)
            over_samples = SMOTE()
            over_samples_X, over_samples_y = over_samples.fit_resample(XY_train, lab)
            check.display(
                "Counter(over_samples_y):" + str(Counter(over_samples_y)), logfile
            )
            Xtrain = over_samples_X.iloc[:, 1:]
            Ytrain = over_samples_X.iloc[:, 0]

    SW = np.ones(shape=(len(Ytrain),))
    sq1 = np.percentile(Ytrain, 2)
    sq2 = np.percentile(Ytrain, 10)
    sq5 = np.percentile(Ytrain, 75)
    sq6 = np.percentile(Ytrain, 95)
    sq7 = np.percentile(Ytrain, 99)
    sq8 = np.percentile(Ytrain, 99.5)
    sq9 = np.percentile(Ytrain, 99.9)
    SW[Ytrain < sq2] = 2
    SW[Ytrain < sq1] = 4
    SW[Ytrain > sq5] = 1.5
    SW[Ytrain > sq6] = 3
    SW[Ytrain > sq7] = 8
    SW[Ytrain > sq8] = 9
    SW[Ytrain > sq9] = 10

    if alg == "nn":
        model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            max_iter=100,
            learning_rate="invscaling",
            learning_rate_init=0.5,
            random_state=1000,
            verbose=True,
        )
    elif alg == "bt":
        model = BaggingRegressor(
            DecisionTreeRegressor(random_state=1000),
            max_samples=0.8,
            n_estimators=300,
            random_state=1000,
        )
    elif alg == "rf":
        model = RandomForestRegressor(
            max_samples=0.8,
            n_estimators=300,
            random_state=1000,
        )
    elif alg == "gbm":
        model = XGBRegressor(
            n_estimators=300,
            max_samples=0.8,
            random_state=1000,
            verbose=3,
        )
    else:
        raise ValueError("invalid ML algorithm name")

    model.fit(Xtrain, Ytrain, sample_weight=SW)

    # predict
    predY = model.predict(Xtrain)

    # leave one out cross validations
    loo = LeaveOneOut()
    ytests = []
    ypreds = []
    if loocv == 1:
        XM = Xtrain.values
        YM = Ytrain.values
        # check.display('weidu'+str(np.shape(Xtrain)),logfile)
        for train_idx, test_idx in loo.split(XM):
            # check.display('train_idx='+str(train_idx),logfile)
            # check.display('type'+str(type(XM)),logfile)
            X_train, X_test = XM[train_idx, :], XM[test_idx, :]
            y_train, y_test = YM[train_idx], YM[test_idx]
            SW_train = SW[train_idx]
            model.fit(X_train, y_train, sample_weight=SW_train)
            y_pred = model.predict(X_test)
            ytests += list(y_test)
            ypreds += list(y_pred)
        ytests = np.array(ytests)
        ypreds = np.array(ypreds)
        loocv_R2 = r2_score(ytests, ypreds)
        loocv_MSE = mean_squared_error(ytests, ypreds)
        loocv_RMSE = np.sqrt(mean_squared_error(ypreds, ytests))
        loocv_dNRMSE = loocv_RMSE / (np.max(ytests) - np.min(ytests))
        loocv_sNRMSE = loocv_RMSE / np.std(ytests)
        loocv_iNRMSE = loocv_RMSE / (
            np.quantile(ytests, 0.75) - np.quantile(ytests, 0.75)
        )
        loocv_f_SB = (np.mean(ypreds - ytests)) ** 2 / loocv_MSE
        loocv_f_SDSD = (np.std(ytests) - np.std(ypreds)) ** 2 / loocv_MSE
        loocv_f_LSC = 1 - loocv_f_SB - loocv_f_SDSD
        loocv_reMSE = (
            (1 / len(ypreds))
            * np.sum((ypreds - ytests) ** 2)
            / np.sum((ytests - np.mean(ytests)) ** 2)
        )
        loocv_slope, intercept, r_value, p_value, std_err = stats.linregress(
            ytests, ypreds
        )
    return (
        model,
        predY,
        loocv_R2,
        loocv_reMSE,
        loocv_slope,
        loocv_dNRMSE,
        loocv_sNRMSE,
        loocv_iNRMSE,
        loocv_f_SB,
        loocv_f_SDSD,
        loocv_f_LSC,
    )
