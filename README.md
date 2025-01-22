# SPINacc
A spinup acceleration tool for land surface model (LSM) family of ORCHIDEE.

Concept: The proposed machine-learning (ML)-enabled spin-up acceleration procedure (MLA) predicts the steady-state of any land pixel of the full model domain after training on a representative subset of pixels. As the computational efficiency of the current generation of LSMs scales linearly with the number of pixels and years simulated, MLA reduces the computation time quasi-linearly with the number of pixels predicted by ML.

Documentation of aims, concepts, workflows are described in [Sun et al (2022)](https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.16623).

![202208_ML_manuscript_figures_v1 0 pptx (2)](https://user-images.githubusercontent.com/79981678/209093236-1601237a-7959-42b6-b6f1-306be1bc0b44.png)

## Contents
The SPINacc package includes:
* `main.py` - the main python module that steers the execution of SPINacc.
* `DEF_*/`  - folders with configuration files for each of the supported ORCHIDEE versions.
    * `config.py` - settings to configure the machine learning performance.
    * `varlist.json` - directories for
    * `varlist-explained.md`
* `Tools/*` - relevant modules for SPINacc
* `AuxilaryTools/SteadyState_checker.py` - tool to assess the state of equilibration in ORCHIDEE simulations
* `tests/` - Reproducibility and regression tests
* `ORCHIDEE_cecill.txt` - ORCHIDEE's license file
* `job` - the job file for a bash environment
* `job_tcsh` - the job file for a tcsh environment

## Usage
### Running SPINacc
Here are the steps to launch SPINacc end-to-end, including the optional tests.

#### Installation

1. Navigate to the location in which you wish to install and clone the repo as so:
    ```
    git clone git@github.com:CALIPSO-project/SPINacc.git
    ```
2. Create a virtual environment and activate:
    ```
    python3 -m venv ./venv3
    source ./venv3/bin/activate
    ```
3. Build all relevant dependencies:
    ```
    cd SPINacc
    pip install -r requirements.txt
    ```
4. Add `Tools` to PYTHONPATH
    ```
    export PYTHONPATH=$(pwd)/Tools:$PYTHONPATH
    ```

#### Get data from Zenodo

This step is relevant to users not on the OBELIX supercomputer. See below if you are currently using OBELIX.

The `setup-data.sh` script has been provided to pull data from the associated ZENODO repository  [https://doi.org/10.5281/zenodo.10514124] and set paths to the forcing data and climate data in `DEF_Trunk/varlist.json`. The ZENODO repository does not include climate data files (variable name `twodeg`, without this, initialisation will fail and SPINacc will be unable to proceed).

It includes:
* `ORCHIDEE_forcing_data` - Explained in varlist-explained.md
* `reference` data - necessary to run the reproducibility checks (see link).

To ensure the script works without error, set the `MYTWODEG` and `MYFORCING` paths appropriately. The `MYFORCING` path points to where you want the forcing data to be extracted to. The default location is `ORCHIDEE_forcing_data` in the project root.

The script runs a `sed` command to replace all occurences of `/home/surface5/vbastri/` with the downloaded and extracted `ORCHIDEE_forcing_data` in `/your/path/to/forcing/vlad_files/vlad_files/` in `DEF_Trunk/varlist.json`. This can be done manually if desired.


#### Running SPINacc

These instructions are designed to get up and running with SPINacc quickly and run the accompanying tests. See section for a more detailed overview of how to adjust ML performance.

1. In `DEF_Trunk/config.py` modify the `results_dir` variable to point to a different path if desired. To run SPINacc from end-to-end, ensure that the steps are set as follows:
    ```
    tasks = [
        1,
        2,
        4,
        5,
    ]
    # 1 = test clustering
    # 2 = clustering
    # 3 = compress forcing
    # 4 = ML
    # 5 = evaluation / visualisation
    ```
    If running from scratch, ensure that `start_from_scratch` is set to `True` in `config.py`. The `start_from_sratch` step creates a `packdata.nc` file and only needs to be done once for a given version of ORCHIDEE. It is also possible to run just a single task, if desired.

2. Then run:
    ```
    python main.py DEF_Trunk/
    ```
    By default, `main.py` will look for the `DEF_Trunk` directory. SPINacc supports passing other configuration / job directories as arguments to `main.py`.

    Results are located in your **EXE_DIR** under `MLacc_results.csv`. Visualisations of R2, Slope and dNRMSE are located for each component in `Eval_all_biomassCpool.png`, `Eval_all_litterCpool.png` and `Eval_all_somCpool.png`.

    For other versions of ORCHIDEE, for instance, CNP2, outputs will be structured similarly.


<!-- * Choose the task you want to launch. In **DEF_TRUNK/MLacc.def**: in __config[3]__ section put **1** (for __task 1__), in __config[5]__ section put your path to your EXE_DIR and in __config[7]__ put 0 for task 1 at least (for the following tasks you can use previous results). -->
<!-- * In **tests/config.py** you have to modify: __test_path=/your/path/to/SPINacc/EXE_DIR/__ -->
<!-- * Also in **tests/config.py** you have to modify: __reference_path='/home/surface10/mrasolon/files_for_zenodo/reference/EXE_DIR/'__ to __reference_path='/your/path/to/reference/'__ -->
<!-- * For following tasks (**2, 3, 4** and **5**) you just need to modify the **config[3]** and **config[7]** sections in **DEF_TRUNK/MLacc.def** -->
<!-- * The results of reproducibility tests are stored in **EXE_DIR/tests_results.txt** -->

## Obtaining 'best' performance




## Run on Obelix Supercomputer

If you are already using the obelix supercomputer is likely that SPINacc will work without much adjustment to the `varlist.json` file.

* In __job__ : __setenv dirpython '/your/path/to/SPINacc/'__ and __setenv dirdef 'DEF_Trunk/'__
* Then launch your first job using  **qsub -q short job**, for task 1
* For tasks 3 and 4, it is better to use **qsub -q medium job**

## Overview of the individual tasks:

The different tasks are (the number of tasks does not correspond to sequence - YET):
* Task 1 [optional]: Evaluates the impact of varying the number of K-means clusters on model performance, setting a default of 4 clusters and producing a ‘dist_all.png’ graph.
![dist_all](https://user-images.githubusercontent.com/79981678/197764400-deaac192-a26b-4f38-8eb1-6a0b50da65c9.png)

* Task 2 performs the clustering using a K mean algorithm and saves the information on the location of the selected pixels (files starting with 'ID'). The location of the selected pixel (red) for a given PFT and all pixel with a cover fraction exceeding 'cluster_thres' [defined in varlist.json] (grey) are plotted in the figures 'ClustRes_PFT**.png'. Example of PFT2 is shown here:
![ClustRes_PFT2_trimed](https://user-images.githubusercontent.com/79981678/197765127-05ef8271-79a0-4775-803c-a1759c413376.png)

* Task 3: Creates compressed forcing files for ORCHIDEE, containing data for selected pixels only, aligned on a global pseudo-grid for efficient pixel-level simulations, with file specifications listed in varlist.json.

* Task 4 performs the ML training on results from ORCHIDEE simulation using the compressed forcing (production mode: resp-format=compressed) or global forcing (debug mode: resp-format=global), extrapolation to a global grid and writing the state variables into global restart files for ORCHIDEE. In debug mode Task 4 also performs the evaluation of ML training outputs vs real model outputs.

* Task 5 [optional]: Visualizes ML performance from Task 3, offering two evaluation modes: global pixel evaluation and leave-one-cross-validation (LOOCV) for training sites, generating plots for various state variables at the PFT level, including comparisons of ML predictions with conventional spinup data.
![Eval_all_loocv_biomassCpool_trim](https://user-images.githubusercontent.com/79981678/197768665-c868f95b-d7f4-4a2f-a942-d37c9e509596.png)

#### Set up baseline reproducibility checks

It is possible to run a set of baseline checks that compare the code to reference output.
These tests are important to ensure that regressions have not been unexpectedly introduced during development. Currently, these checks only work for the Trunk version of ORCHIDEE.

1. From Zenodo, Download `Reproducibility_tests_reference.zip`, unzip and store it in a directory `/your/path/to/reference/`

2. In `DEF_Trunk/config.py` set the `reference_dir` variable as above.

3. To execute the reproducibility checks at runtime ensure that True values are set in all relevant steps in `DEF_Trunk/config.py`.

4. Alternatively, steps can be executed following the completion of a successful run by doing the following:

    ```
    pytest --trunk=DEF_Trunk/ -v --capture=sys
    ```
    Above we demonstrate that it is possible to point to different output directories with the `--trunk` flag.

    To run a single test do:

    ```
    pytest --trunk=DEF_Trunk -v --capture=sys ./tests/test_task4.py
    ```

5. The configuration `config.py` in branch `main` should be configured correctly. But if not, ensure that
    the following assignments have been made:

    ```
    alg = 'bt'
    start_from_scratch = True
    take_unique = False
    kmeans_clusters = 4
    max_kmeans_clusters = 9
    random_seed = 1000
    ```

6. The checks are as follows:

    - `test_init.py`: Computes recursive compare of `packdata.nc` to reference `packdata.npy` and `auxil.npy`.
    - `test_task1.py`: Checks `dist_all.npy` to the reference.
    - `test_task2.py`: Checks `IDloc.npy`, `IDSel.npy` and `IDx.npy` to the reference.
    - `test_task3.py`: Currently not checked.
    - `test_task4.py`: Reads in old *.txt formatted files in reference and compares to new `MLacc_results.csv` across all components. Tolerance is 1e-2.
    - `test_task4_2.py`: Compares `SBG_FGSPIN.340Y.ORC22v8034_22501231_stomate_rest.nc` file to reference.
