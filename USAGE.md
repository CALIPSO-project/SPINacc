## SPINacc
A spinup acceleration tool for land surface model (LSM) family of ORCHIDEE.

Concept: The proposed machine-learning (ML)-enabled spin-up acceleration procedure (MLA) predicts the steady-state of any land pixel of the full model domain after training on a representative subset of pixels. As the computational efficiency of the current generation of LSMs scales linearly with the number of pixels and years simulated, MLA reduces the computation time quasi-linearly with the number of pixels predicted by ML. 

Documentation of aims, concepts, workflows are described in Sun et al.202 [open-source]: https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.16623

![202208_ML_manuscript_figures_v1 0 pptx (2)](https://user-images.githubusercontent.com/79981678/209093236-1601237a-7959-42b6-b6f1-306be1bc0b44.png)


## INFORMATION FOR USERS:
### MORE INFOS ON SOME OF THE TASKS OF THE TOOL:

* For task 2: specify in the job file where the code of the tool is located using the variable dirpython (L4), and sfolders containting the configuration files for your version of ORCHIDEE using the variable dirdef. The supported model versions are CNP2 = CNP v1.3 (CNP with MIMICS soil), Trunk = Trunk 2.2, and CNP = CNP v1.2 (CNP with CENTURY soil), MICT = MICT [incomplete!] ). 

* For task 3: adjust the files in configuration folder: i.e. the type of task to be performed (which are described below) as well as the specifications of your ORCHIDEE simulation. The tool can run a single task or a sequence of tasks.
	* MLacc.def defines the task(s) to do and the execution directory; 
	* varlist.json defines the specification of the input data (ORCHIDEE training data, ORCHIDEE forcing data): e.g. resolution, state variables to predict, etc.
* For the task 4: execute the tool by (for obelix): qsub -q long job   / qsub -q long job_tcsh (dependig on your environmnet)
* For the task 5: the output of the tool is stored in the folder specified in MLacc.def under 'config[5] : execution directory'. The progress of the tool is writen in the file specifed in MLacc.def under 'config[1] : logfile'

### DETAILED INDIVIDUAL TASKS OF THE TOOL: 

The different tasks are (the number of tasks does not correspond to sequence - YET):
* Task 1 [optional]: Provides information on the expected gain in model performance by incrasing the number of (k-mean) clusters.  The optimal number of clusters (Ks) is a tradeoff between computation demand and ML performance; and Ks can vary according to your model and the simulation setup. The default number is 4 and is set via  'config[9] : number of K for final Kmean algorithm' in MLacc.def. This task produces the figure ‘dist_all.png’ which shows the sum of distance for different numbers of clusters, i.e. using different Ks. The default maximum number of Ks being tested is 9, you can set higher values if needed using config[11] in MLacc.def.
![dist_all](https://user-images.githubusercontent.com/79981678/197764400-deaac192-a26b-4f38-8eb1-6a0b50da65c9.png)

* Task 2 performs the clustering using a K mean algorithm and saves the information on the location of the selected pixels (files starting with 'ID'). The location of the selected pixel (red) for a given PFT and all pixel with a cover fraction exceeding 'cluster_thres' [defined in varlist.json] (grey) are plotted in the figures 'ClustRes_PFT*.png'. Example of PFT2 is shown here:
![ClustRes_PFT2_trimed](https://user-images.githubusercontent.com/79981678/197765127-05ef8271-79a0-4775-803c-a1759c413376.png)

* Task 3 generates compressed forcing files for ORCHIDEE which only contain information for the selected pixels. The data is aligned uniformly across the globe and stored on a new global pseudo-grid which ensures high computational efficiency for the pixel level simulations with ORCHIDEE. The forcing files which need to be processed must be listed in varlist.json under "sourcepath" for climate and "restart" for others (e.g. nutrient inputs).These files are compatible with ORCHIDEE and can be directly used in ORCHIDDEE simulations (e.g. using your COMP/X.cards in the libIGCM simulation configuration folder).

* Task 4 performs the ML training on results from ORCHIDEE simulation using the compressed forcing (production mode: resp-format=compressed) or global forcing (debug mode: resp-format=global), extrapolation to a global grid and writing the state variables into global restart files for ORCHIDEE. In debug mode Task 4 also performs the evaluation of ML training outputs vs real model outputs.

* Task 5 [optional] visualizes the performance of the ML in task 3. Two kinds of evaluations
are available: (1) the evaluation for global pixels (config[15]=0)  (developer mode; not described in the following) ; 
(2) the leave-one-cross-validation (LOOCV) for training sites (config[15]=1) which is the default case which evaluates the
performance of the ML training. The computational time required for the LOOCV is large compared to other tasks. Thus, it is envisioned to be performed in parallel to the main task. Plots fo are producte for all state variables at the leve of PFTs ('Eval_all_*'). Depending on the model version this includes separates files for biomass, soil, litter, som, and microbes. In case of the developer mode [0]: the plots show the correlation between ML predicted state variables on PFT and the ones from the conventional spinup. Shown are: coefficient of determination (R2), relative bias (rs), normalized root mean squared error by the difference between maximum and minimum (NRMSE). 
![Eval_all_loocv_biomassCpool_trim](https://user-images.githubusercontent.com/79981678/197768665-c868f95b-d7f4-4a2f-a942-d37c9e509596.png)


### HOW TO SPECIFY THE INPUT DATA / SIMULATIONS:

using the varlist.json files in fiolder DEF_:

You need to modify where the data for ML training is located using sourcepath & sourcefile. The data might be provided in different files, thus the data is separated into groups of variables:

* -climate: climate forcing data used during spinup (mind to specify the same time period as in the ORCHIDEE pixel level runs and in the full spatial domain simulation)
* -pred: other predictor variables used for the ML taken from a short (duration is model version specific; min. 1 year) transient ORCHIDEE simulation over the whole spatial domain (from scratch; not restarting). (This is done as the resolution of boundary conditions other than climate can differ from the resolution of ORCHIDEE, and need to be remapped first to ORCHIDEE resolution. This is automatically done by ORCHIDEE).
	* var1: NPP and LAI from last year of initial short simulation. (These two variables are not boundary conditions for ORCHIDEE but state variables. Information on these variables from the transient spinup phase have been found to improve ML performance for ORCHIDEE-CNP.)
	* var2 - var4: soil properties and/or nutrient-related variables. 
	* For var2 - var4, if they are missing in *_rest.nc (e.g. N and P deposition), please use the variables in *_stomate_history.nc.	
* -PFTmask: max PFT cover fractions, which are used to mask grids with extreme low PFT fractions. This information is usually found in the *_stomate_history.nc (VEGET_COV_MAX). In case that cover fractions are kept constant in time (i.e. DYNVEG=no), you can use any year of the initial short simulation for the whole domain. In case you have dynamic cover fractions: not supported (yet).

* -resp: response variables, i.e. the equlibrium pools from the traditional spin-up done over part of the spatial domain (production mode) or globally (debug mode). The field 'format' defines the format of the source restart file provided: "compressed" (production mode) or "global" (debug mode). In production mode the 'sourcefile' should specify the path to the stomate restart file obtained with compressed forcing (spinup over selected pixels), the 'targetfile' should specify the path to the stomate restart file from any (even short) global run (it will be used to fill the fields with trained/predicted data). In debug mode 'targetfile' can be dropped (then 'sourcefile' is used for both training and filling with predicted data).
* -restart: This lists the files which are being processed into a compressed format. Mind, all files from which ORCHIDEE reads in information when restarting must be listed here and passed to your libIGCM configuration for the site-runs. Besides the ORCHIDEE restart files (e.g. from the pre-run) this can include files in the sechiba.card. We recommend to include all files in sechiba.card as the extra computational demand is very minor and you do not have to inquire from which of the files information is actually read during a restart.

You can find the detailed information for each variable in the Trunk and CNP examples: DEF_Trunk/varlist.json, DEF_CNP/varlist.json 
You can create your varlist.json according to your case. 

## INFORMATION ON THE ORCHIDEE RUNS:
* PRE-RUN simulation (needed for task 1): This is conventional global ORCHIDEE spinup simulation. You can use your usual configuration. If you have an analytic spinup available, the duration should equal the lenght till after the first spin up step; otherwise we use 300 years (not optimized lenght). This run needs to report the following variables (mind the names might be different in your ORC version):
VEGET_COV_MAX, NPP, LAI and (for CNP only) soil_orders,soil_shield, NHX_DEPOSITION,NOY_DEPOSITION, and P_DEPOSITION  in stomate history. All time invariant boundary conditions variables of your ORCHIDEE version should be written in the restart files (e.g. clay_frac,silt_frac,bulk,soil_ph for CNP). 

* SITE-LEVEL simulation (needed for task 4): This is a ORCHIDEE spinup simulation which makes use of the compressed forcing produced by the ML tool. You can use a PRE-RUN type simulation configuration as template and replace all input files with the compressed ones. These include climate forcing files, driver/stomate/sechiba restart files, and nutrient input files (for ORCHIDEE-CNP). These files are in the execution directory of the tool after the task 3.
* RE-RUN Simulation: This is a conventional global ORCHIDEE spinup simulation restarting from the stomate restart file ( you find in the ML tool execution dir after task 4) from the ML tool, and sechiba and driver restart files from the PRE-RUN. This runs aims at reducing biases in the ML predicted steady state and it can be analyzed to diagnose to what extent a steady state has been achieved or not.

## INFORMATION FOR CODE DEVELOPERS:

### SIMULATIONS NEEDED TO TEST THE TOOL:
At current DEV state: You need the output and restart files from a conventional spinup simulation with ORCHIDEE. At this DEV stage, we need the whole domain as it facilitates the validation of the tool to have the 'true' equilibrium to benchmark the ML based predictions. 

Specifically, we need information from early during the spinup (transient phase, e.g. for ORCHIDEE trunk we use the year before the first analytical spinup up step, for ORCHIDEE-CNP which has not analytical spinup we use year 300) and from a year when the pools have approached a steady-state (e.g. 99% of pixels have absolute drifts in total C of less than 1 g per m2 per year).

The tool extracts from the transient phase:
* all boundary conditions (i.e. ORCHIDEE forcings) on the (final) spatial resolution of ORCHIDEE. 
* information on NPP and LAI during early spinup (transient phase) which we found vastly improves ML predictions.

From the stable state, we extract 
* all state variables to train the machine learning allgorithm

MIND: The exact simulation lenghts needed to reach steady-state depends on the model version, and steady-state criteria.





