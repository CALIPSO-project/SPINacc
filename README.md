# SPINacc
A spinup acceleration tool for land surface model (LSM) family of ORCHIDEE.

Concept: The proposed machine-learning (ML)-enabled spin-up acceleration procedure (MLA) predicts the steady-state of any land pixel of the full model domain after training on a representative subset of pixels. As the computational efficiency of the current generation of LSMs scales linearly with the number of pixels and years simulated, MLA reduces the computation time quasi-linearly with the number of pixels predicted by ML. 

Documentation of aims, concepts, workflows are described in Sun et al. in prep.

## CONTENT
The SPINacc package includes:
* job
* main.py
* Tools/*
* DEF_*/

 
## INFORMATION FOR USERS:
### HOW TO RUN THE CODE:

* First: copy the code from github to your own machine ( code is tested for LSCE's obelix ).
* Second: specify in the file 'job' or 'job_tcsh' (depending on your environment) where the code is located using the variable dirpython (L4), and specify the folder with the configuration for your version of ORCHIDEE using the variable dirdef. The supported model versions are CNP2 = CNP v1.3 (CNP with MIMICS soil), Trunk = Trunk 2.x, and CNP = CNP v1.2 (CNP with CENTURY soil), MICT = MICT). 
* Third: adjust the files in configuration folder: i.e. the type of task to be performed as well as the specifis of your simulation. The tool can run specific tasks individually.
	* MLacc.def defines the tasks to do and the execution directory; 
	* varlist.json defines the specification of the input data: e.g. resolution, state variables to predict, etc.
* Forth: execute the tool by: qsub -q long job   / qsub -q long job_tcsh (dependig on your environmnet)
* Fifth: the output of the tool is stored in the folder specified in MLacc.def under 'config[5] : execution directory'

### GUIDELINES FOR SETTING UP THE CONFIGURATION (MLacc.def):
The different tasks are:
* Task 1 [optional]: Provides information on the expected gain in model performance by incrasing the number of clusters. The optimal number of clusters (Ks) can vary according to your model and the simulation setup. The default number is 4 and is set via  'config[9] : number of K for final Kmean algorithm' in MLacc.def. The optimal number of clusters is a tradeoff between computation demand and ML performance.  This task produces the figure ‘dist_all.png’ which shows the sum of distance for different numbers of clusters, i.e. using different Ks. The default maximum number of Ks being tested is 9, you can set higher values if needed using config[11] in MLacc.def.
* Task 2 performs the clustering using a K mean algorithm and saves the information on the location of the selected pixels.
* Task 5 generates compressed forcing files for ORCHIDEE simulations which only contain information for the selected pixels. The data is aligned uniformly across the globe and stored on a new global pseudo-grid. This new input files ensure high computational efficiency for the pixel level simulations with ORCHIDEE. 
* Task 3 performs the ML training and write the state variables into restart files for global simulations with ORCHIDEE.
* Task 4 [optional] visualizes the performance of the ML in task 3. Two kinds of evaluations
are available: (1) the evaluation for global pixels (config[15]=0)  (developer mode) ; 
(2) the leave-one-cross-validation (LOOCV) for training sites (config[15]=1) which is the default case which evaluates the
performance of the ML training. It is very time consuming.
 

### HOW TO SPECIFY THE INPUT DATA / SIMULATIONS:

using the varlist.json files in fiolder DEF_:

You need to modify where the data for ML training is located using sourcepath & sourcefile. The data might be provided in different files, thus the data is separated into groups of variables:

* -climate: climate forcing data used during spinup (mind to specify the same time period as in the ORCHIDEE pixel level runs and in the full spatial domain simulation)
* -pred: other predictor variables used for the ML taken from a short (duration is model version specific; min. 1 year) transient ORCHIDEE simulation over the whole spatial domain (from scratch; not restarting). (This is done as the resolution of boundary conditions other than climate can differ from the resolution of ORCHIDEE, and need to be remapped first to ORCHIDEE resolution. This is automatically done by ORCHIDEE).
	* var1: NPP and LAI from last year of initial short simulation. (These two variables are not boundary conditions for ORCHIDEE but state variables. Information on these variables from the transient spinup phase have been found to improve ML performance for ORCHIDEE-CNP.)
	* var2 - var4: soil properties and/or nutrient-related variables. 
	* For var2 - var4, if they are missing in *_rest.nc (e.g. N and P deposition), please use the variables in *_stomate_history.nc.	
* -PFTmask: max PFT cover fractions, which are used to mask grids with extreme low PFT fractions. This information is usually found in the *_stomate_history.nc (VEGET_COV_MAX). In case that cover fractions are kept constant in time (i.e. DYNVEG=no), you can use any year of the initial short simulation for the whole domain. In case you have dynamic cover fractions: not supported (yet).

* -resp: response variables, i.e. the equlibrium pools from the traditional spin-up. Ultimately, it should specify the conventional spinup for part of the spatial domain. At the moment we use simulation for the whole spatial domain. They are usually in the *_stomate_history.nc

You can find the detailed information for each variable in the Trunk and CNP examples: DEF_Trunk/varlist.json, DEF_CNP/varlist.json 
You can create your varlist.json according to your case. 


## INFORMATION FOR CODE DEVELOPERS:

### SIMULATIONS NEEDED TO TEST THE TOOL
You need the output and restart files from a conventional spinup simulation with ORCHIDEE. At this DEV stage, we need the whole domain as it facilitates the validation of the tool to have the 'true' equilibrium. FINAL: it should be trained on the pixel-level simulation results.

Specifically, we need information from early during the spinup (transient phase) and when the pools have approached a steady-state.
From the transient phase, we
* extract all boundary conditions (i.e. ORCHIDEE forcings) on the spatial resolution of ORCHIDEE. 
* information on NPP and LAI during early spinup (transient phase) vastly improves ML predictions.
From the stable state, we 
* extract all state variables to train the machine learning allgorithm

The exact simulation lenghts needed to reach steady-state depends on the model version, and steady-state criteria.

### HOW TO UPDATE THE CODE ON GITHUB: you need to do multiple steps: 
* First, "git add" to add all the files that you changed. It is recommended to only add files you have changed and make sure you updated with any changes updated to github since you downloaded your copies.
* Second, "git commit" to commit them to your local copy (a difference between svn and git and is that git has a local copy that you commit to). 
* Third, "git push" to push them to the master repository (here). 
This might help: https://git-scm.com/docs/gittutorial

### Other USEFUL COMMANDS: 
* "git diff" will show you all the changes you have made. 
* "git pull" will update all your local files with what exists in the master code (here). 
* "git status" will show you what files have been modified.





