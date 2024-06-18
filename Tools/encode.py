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


##@param[in]   comXY                 X and Y variables
##@param[in]   colum
##@param[in]   Nm
##@retval      combine_XY_encode
def en_code(comXY, colum, Nm):
    combine_XY_encode = []
    jum = comXY.iloc[:, colum]
    append_array = DataFrame(np.zeros(shape=(len(jum), len(Nm))))
    append_array.index = comXY.index
    for ii in range(0, len(jum)):
        append_array.loc[comXY.index[ii], int(jum[comXY.index[ii]]) - 1] = 1
    append_array = (append_array.astype(int)).astype(bool)
    breakpoint()
    comXY = comXY.drop(["soil_orders"], axis=1)
    combine_XY_encode = pd.concat([comXY, append_array], axis=1)
    return combine_XY_encode
