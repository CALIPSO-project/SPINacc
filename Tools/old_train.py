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


##@param[in]   XY_train               latitudes of selected pixels
##@param[in]   logfile                logfile
##@retval      TreeEns                Tree ensemble
##@retval      predY                  predicted Y
def training_BAT(XY_train, logfile):
    XX = XY_train.iloc[:, 1:].values
    YY = XY_train.iloc[:, 0].values
    # labels=np.zeros(shape=(len(YY),1))
    mod = KMeans(n_clusters=3)
    lab = mod.fit_predict(np.reshape(YY, (-1, 1)))
    count = Counter(lab)
    check.display("Counter(lab):" + str(count), logfile)
    # only one value
    if len(np.unique(YY)) == 1:
        TreeEns = []
        predY = YY
        return TreeEns, predY

    try:
        over_samples = SMOTE()
        over_samples_X, over_samples_y = over_samples.fit_resample(XY_train, lab)
        check.display(
            "Counter(over_samples_y):" + str(Counter(over_samples_y)), logfile
        )
        Xtrain = over_samples_X.iloc[:, 1:]
        Ytrain = over_samples_X.iloc[:, 0]
    #  else:
    except:
        mod = KMeans(n_clusters=2)
        lab = mod.fit_predict(np.reshape(YY, (-1, 1)))
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
                    (lab, np.repeat(lab[lab == label], int(np.ceil(6 / number) - 1)))
                )
                print(len(lab), number, int(np.ceil(6 / number)))
        check.display("Counter(lab):" + str(Counter(lab)), logfile)
        over_samples = SMOTE()
        over_samples_X, over_samples_y = over_samples.fit_resample(XY_train, lab)
        check.display(
            "Counter(over_samples_y):" + str(Counter(over_samples_y)), logfile
        )
        Xtrain = over_samples_X.iloc[:, 1:]
        Ytrain = over_samples_X.iloc[:, 0]

    SW = np.ones(shape=(len(Ytrain),))
    sq1 = np.percentile(YY, 2)
    sq2 = np.percentile(YY, 10)
    sq5 = np.percentile(YY, 75)
    sq6 = np.percentile(YY, 95)
    sq7 = np.percentile(YY, 99)
    sq8 = np.percentile(YY, 99.5)
    sq9 = np.percentile(YY, 99.9)
    SW[Ytrain < sq2] = 2
    SW[Ytrain < sq1] = 4
    SW[Ytrain > sq5] = 1.5
    SW[Ytrain > sq6] = 3
    SW[Ytrain > sq7] = 8
    SW[Ytrain > sq8] = 9
    SW[Ytrain > sq9] = 10

    # Bagging ensemble
    tree = DecisionTreeRegressor(random_state=1000)
    #                               max_depth=14, min_samples_split=5)

    bag = BaggingRegressor(
        base_estimator=tree, max_samples=0.8, n_estimators=300, random_state=1000
    )
    TreeEns = bag.fit(Xtrain, Ytrain, sample_weight=SW)
    # sample_weight=SW
    predY = bag.predict(XX)
    # leave one out cross validations
    return TreeEns, predY
