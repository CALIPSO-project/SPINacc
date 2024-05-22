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


class pack:
    """
    Variables that are packaged
    """

    def __init__(self):
        self.__doc__ = "packaged data"

    def __getitem__(self, key):
        return self.__dict__[key]


class auxiliary:
    """
    Class to store auxiliary variables
    """

    def __init__(self):
        self.__doc__ = "auxiliary data"
