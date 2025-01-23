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


def verbose(sss, logfile):
    """
    Write a string to a file.

    Args:
        sss (str): String to be written.
        logfile (file): File object for logging.

    Returns:
        None
    """
    if logfile:
        logfile.write(sss + "\n")


def display(sss, logfile):
    """
    Print a string to the console and write it to a file.

    Args:
        sss (str): String to be printed.
        logfile (file): File object for logging

    Returns:
        None

    """
    if logfile:
        logfile.write(sss + "\n")
    print(sss)


def check_dir(dirn, logfile):
    """
    Check if a directory exists.

    Args:
        dirn (str): Directory name.
        logfile (file): File object for logging.

    Raises:
        SystemExit: If the directory does not exist.
    """
    if not os.path.isdir(dirn):
        display("Cannot find directory " + dirn, logfile)
        sys.exit()


def check_file(file, logfile):
    """
    Check if a file exists.

    Args:
        file (str): File name.
        logfile (file): File object for logging.

    Raises:
        SystemExit: If the file does not exist.
    """
    if not os.path.isfile(file):
        display("Cannot find file " + file, logfile)
        sys.exit()
