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

import os
import sys


##@param[in] sss string to be written
##@param[in] logfile verbose file
def verbose(sss, logfile):
    if logfile:
        logfile.write(sss + "\n")


##@param[in] sss string to be written
##@param[in] logfile verbose file
def display(sss, logfile):
    if logfile:
        logfile.write(sss + "\n")
    print(sss)


##@param[in] dirn directory name
##@param[in] logfile verbose file
def check_dir(dirn, logfile):
    if not os.path.isdir(dirn):
        display("Cannot find directory " + dirn, logfile)
        sys.exit()


##@param[in] file file name
##@param[in] logfile verbose file
def check_file(file, logfile):
    if not os.path.isfile(file):
        display("Cannot find file " + file, logfile)
        sys.exit()
