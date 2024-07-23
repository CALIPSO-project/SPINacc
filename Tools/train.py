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

# import skorch
# from torch import nn, optim
from sklearn.neural_network import MLPRegressor


# class MLPRegressor(nn.Sequential):
#     def __init__(self, ndim_in, ndim_hid, ndim_out):
#         super(MLPRegressor, self).__init__(
#             nn.Linear(ndim_in, ndim_hid),
#             nn.ReLU(),
#             nn.Linear(ndim_hid, ndim_hid),
#             nn.ReLU(),
#             nn.Linear(ndim_hid, ndim_out),
#         )


##@param[in]   XY_train               latitudes of selected pixels
##@param[in]   logfile                logfile
##@param[in]   loocv                  do leave-one-out-cross-validation(1) or not (0)
##@retval      TreeEns                Tree ensemble
##@retval      predY                  predicted Y
def training_BAT(X, Y, logfile, loocv):
    print("Data shapes: ", X.shape, Y.shape)

    # run the KMeans algorithm to find the cluster centers, and resample the data
    mod = KMeans(n_clusters=3)
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

    # model = skorch.NeuralNetRegressor(
    #     module=MLPRegressor,
    #     max_epochs=1000,
    #     lr=0.01,
    #     criterion=nn.MSELoss,
    #     optimizer=optim.Adam,
    #     # device="cuda",
    # )
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        max_iter=100,
        learning_rate="invscaling",
        learning_rate_init=0.1,
        verbose=True,
    )

    model.fit(X, Y)
    predY = model.predict(X)

    return model, predY
