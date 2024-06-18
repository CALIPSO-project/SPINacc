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
    # if length of unique target is one
    # if len(np.unique(Y)) == 1:
    #     # return a set of default values and an empty TreeEnsemble
    #     TreeEns = []
    #     predY = Y
    #     return (
    #         TreeEns,
    #         predY,
    #         # np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    #     )

    # If the length of unique target variable is not 1,
    # run the KMeans algorithm to find the cluster centers, and resample the data
    # try:
    #     mod = KMeans(n_clusters=3)
    #     lab = mod.fit_predict(np.reshape(YY, (-1, 1)))
    #     count = Counter(lab)
    #     check.display("Counter(lab):" + str(count), logfile)
    #     over_samples = SMOTE()
    #     over_samples_X, over_samples_y = over_samples.fit_resample(XY_train, lab)
    #     check.display(
    #         "Counter(over_samples_y):" + str(Counter(over_samples_y)), logfile
    #     )
    #     Xtrain = over_samples_X.iloc[:, 1:]
    #     Ytrain = over_samples_X.iloc[:, 0]
    # #  else:
    # except:
    #     mod = KMeans(n_clusters=2)
    #     lab = mod.fit_predict(np.reshape(YY, (-1, 1)))
    #     count = Counter(lab)
    #     check.display("Counter(lab):" + str(Counter(lab)), logfile)
    #     # resample requires minimum number of a cluster >=6, if not, then repeat current samples
    #     for label, number in count.items():
    #         if number < 6:
    #             XY_train = pd.concat(
    #                 (XY_train,)
    #                 + (XY_train[lab == label],) * int(np.ceil(6 / number) - 1)
    #             )
    #             lab = np.hstack(
    #                 (lab, np.repeat(lab[lab == label], int(np.ceil(6 / number) - 1)))
    #             )
    #     #        print(len(lab),number,int(np.ceil(6/number)))
    #     check.display("Counter(lab):" + str(Counter(lab)), logfile)
    #     over_samples = SMOTE()
    #     over_samples_X, over_samples_y = over_samples.fit_resample(XY_train, lab)
    #     check.display(
    #         "Counter(over_samples_y):" + str(Counter(over_samples_y)), logfile
    #     )
    #     Xtrain = over_samples_X.iloc[:, 1:]
    #     Ytrain = over_samples_X.iloc[:, 0]

    # model = skorch.NeuralNetRegressor(
    #     module=MLPRegressor,
    #     max_epochs=1000,
    #     lr=0.01,
    #     criterion=nn.MSELoss,
    #     optimizer=optim.Adam,
    #     # device="cuda",
    # )
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, 
                         learning_rate='invscaling', learning_rate_init=0.1, verbose=True)
       
    model.fit(X, Y)
    predY = model.predict(X)
    
    return model, predY
