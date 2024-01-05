# SPINacc
A spinup acceleration tool for land surface model (LSM) family of ORCHIDEE.

Concept: The proposed machine-learning (ML)-enabled spin-up acceleration procedure (MLA) predicts the steady-state of any land pixel of the full model domain after training on a representative subset of pixels. As the computational efficiency of the current generation of LSMs scales linearly with the number of pixels and years simulated, MLA reduces the computation time quasi-linearly with the number of pixels predicted by ML. 

Documentation of aims, concepts, workflows are described in Sun et al.202 [open-source]: https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.16623

![202208_ML_manuscript_figures_v1 0 pptx (2)](https://user-images.githubusercontent.com/79981678/209093236-1601237a-7959-42b6-b6f1-306be1bc0b44.png)

## CONTENT
The SPINacc package includes:
* job - the job file for a bash environment
* job_tcsh - the job file for a tcsh environment
* main.py - the main python module
* Tools/* - folder with the other python modules
* DEF_*/  - folders containting the configuration files for each of the supported ORCHIDEE versions
* AuxilaryTools/SteadyState_checker.py - tool to assess the state of equilibration in ORCHIDEE simulations
* tests/ - the reproducibility code in Python
* requirements.txt - listing necessary dependencies to use SPINacc
* ORCHIDEE_cecill.txt - the same license used by ORCHIDEE
* docs/ - more detailed documentation about ORCHIDEE simulations
 
## INFORMATION FOR USERS:
### HOW TO RUN THE CODE:


Here are the steps to launch the different tasks of this repository (and the reproducibility tests associated):

* Download the code: **git clone git@github.com:dsgoll123/SPINacc.git**
* __cd SPINacc__
* If you want to stay on the main code skip this point, otherwise do : __git checkout your_branch__
* Create an execution directory: __mkdir EXE_DIR__
* Choose the test you want to launch. In **DEF_TRUNK/MLacc.def**: in config[3] section put **1** (for __test 1__), in config[7] section put 0,  in config[5] section put your path to your EXE_DIR
* In __job__ : __setenv dirpython '/your/path/to/SPINacc/'__ and __setenv dirdef 'DEF_Trunk/'__
* Download the reference produced files files for reprodcibility on ZENODO (here: link) to __'/your/path/to/reference/'__
* In **tests/config.py** you have to modify: __test_path=/your/path/to/SPINacc/EXE_DIR/__
* Also in **tests/config.py** you have to modify: __reference_path='/home/surface10/mrasolon/files_for_zenodo/reference/EXE_DIR/'__ to __reference_path='/your/path/to/reference/'__
* Then launch via **qsub -q short job**, for test 1
* For next tests (**2, 3, 4** and **5**) you just need to modify the **config[3]** and **config[7]** sections in **DEF_TRUNK/MLacc.def** 
* For tasks 3 and 4, it is better to use **qsub -q medium job**
* Launching tasks in chain (e.g. "1, 2" or "3, 4, 5") will be a possibility soon 
* The results of the tasks are located in your **EXE_DIR**
* The results of reproducibility tests are stored in **EXE_DIR/tests_results.txt**


### OVERVIEW OF THE INDIVIDUAL TASKS OF THE TOOL:
(The detail of each tasks of the tool is provided in docs/documentation.txt)

The different tasks are (the number of tasks does not correspond to sequence - YET):
* Task 1 [optional]: Evaluates the impact of varying the number of K-means clusters on model performance, setting a default of 4 clusters and producing a ‘dist_all.png’ graph.
![dist_all](https://user-images.githubusercontent.com/79981678/197764400-deaac192-a26b-4f38-8eb1-6a0b50da65c9.png)

* Task 2 performs the clustering using a K mean algorithm and saves the information on the location of the selected pixels (files starting with 'ID'). The location of the selected pixel (red) for a given PFT and all pixel with a cover fraction exceeding 'cluster_thres' [defined in varlist.json] (grey) are plotted in the figures 'ClustRes_PFT**.png'. Example of PFT2 is shown here:
![ClustRes_PFT2_trimed](https://user-images.githubusercontent.com/79981678/197765127-05ef8271-79a0-4775-803c-a1759c413376.png)

* Task 3: Creates compressed forcing files for ORCHIDEE, containing data for selected pixels only, aligned on a global pseudo-grid for efficient pixel-level simulations, with file specifications listed in varlist.json.

* Task 4 performs the ML training on results from ORCHIDEE simulation using the compressed forcing (production mode: resp-format=compressed) or global forcing (debug mode: resp-format=global), extrapolation to a global grid and writing the state variables into global restart files for ORCHIDEE. In debug mode Task 4 also performs the evaluation of ML training outputs vs real model outputs.

* Task 5 [optional]: Visualizes ML performance from Task 3, offering two evaluation modes: global pixel evaluation and leave-one-cross-validation (LOOCV) for training sites, generating plots for various state variables at the PFT level, including comparisons of ML predictions with conventional spinup data.
![Eval_all_loocv_biomassCpool_trim](https://user-images.githubusercontent.com/79981678/197768665-c868f95b-d7f4-4a2f-a942-d37c9e509596.png)


### REPRODUCIBILITY TESTS : 
The possibility to choose to run (or not) reproducibility tests is coming soon.





