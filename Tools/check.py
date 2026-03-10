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


def check_config_consistency(config, varlist):
    """
    Check for incompatible combinations of options between config and varlist.

    This function should be called at program start, before any tasks are
    executed, to catch invalid configurations early.

    Args:
        config (object): The configuration object (e.g., a loaded config module or
            any object exposing configuration attributes).
        varlist (dict): The variable list dictionary loaded from varlist.json.

    Raises:
        ValueError: If an incompatible combination of options is detected.
    """
    errors = []

    # Check 1: unstructured format requires leave_one_out_cv=True
    resp_format = varlist.get("resp", {}).get("format", None)
    leave_one_out_cv = getattr(config, "leave_one_out_cv", False)
    if resp_format == "unstructured" and not leave_one_out_cv:
        errors.append(
            "varlist.json 'resp.format' is 'unstructured' but config.py "
            "'leave_one_out_cv' is False. "
            "The unstructured format requires leave_one_out_cv=True."
        )

    # Check 2: sel_most_PFT_sites=True requires old_cluster=False
    sel_most_PFT_sites = getattr(config, "sel_most_PFT_sites", False)
    old_cluster = getattr(config, "old_cluster", True)
    if sel_most_PFT_sites and old_cluster:
        errors.append(
            "config.py 'sel_most_PFT_sites' is True but 'old_cluster' is also True. "
            "sel_most_PFT_sites requires old_cluster=False."
        )

    if errors:
        raise ValueError(
            "Incompatible configuration options detected:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
