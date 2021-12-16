# SPINacc
A spinup acceleration procedure for land surface models which is model independent.

Documentation of aims, concepts, workflows are found in file: !MISSING!

For more information about git, see these pages:
https://rogerdudler.github.io/git-guide/
https://www.freecodecamp.org/


## CONTENT
The SPINacc package includes:
* job
* main.py
* Tools/*
* DEF_*/

 
## INFORMATION FOR USERS:
### HOW TO RUN THE CODE:

* First: copy the code to your own directory on obelix (this can be done from the command line with "git clone https://github.com/dsgoll123/SPINacc").
* Second: specify in the file 'job' or 'job_tcsh' (depending on your environment) where the code is located using dirpython (L4), and specify the folder with the configuration for your version of ORCHIDEE using dirdef (for now we have: CNP= CNP v1.2 (CENTURY soil), CNP2 = CNP v1.3 (MIMICS soil), Trunk = Trunk 2.0, MICT = MICT). 
* Third: adjust the files in configuration folder: i.e. the type of task to be performed as well as the specifis of your simulation:
	* MLacc.def defines the tasks to do and the execution directory; tasks are 1= test number of k clusters 2=clustering, 3= ML training,
evaluation and extrapolation, 4 visualizations of ML performance (see below for more information)
	* varlist.json defines the specification of the input data: e.g. resolution, state variables to predict, etc.
* Forth: execute the tool by: qsub -q long job   / qsub -q long job_tcsh (dependig on your environmnet)

### GUIDELINES FOR SETTING UP THE CONFIGURATION (MLacc.def):

* Task 1 [optional]: The optimal number of clusters (Ks) can vary according to your
model and the simulation setup. The number of clusters which are a tradeoff
between computation demand and ML performance. Task 1 helps you to decide
on the number of clusters. Task 1 produces the figure ‘dist_all.png’ which shows
the sum of distance for different numbers of clusters, i.e. using different Ks. The default maximum number of Ks being tested is 9, you can set higher values if
needed using config[11] in MLacc.def.
* Task 2 prepares the data for the ML training in task 3 which is saved in the
‘dirdef’ folder (config[7]=1).
* Task 3 performs the ML training based on the data prepared by task 2 and write the state variables into a template of a restart file of ORCHIDEE.
* Task 4 visualizes the performance of the ML in task 3, Two kinds of evaluations
are designed: (1) the evaluation for global pixels (config[15]=0) which is designed
for the use by developers; (2) the leave-one-cross-validation (LOOCV) for
training sites (config[15]=1) which is the default case which evaluates the
performance of the ML training, It is time consuming,

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
You need the output and restart files from a conventional spinup simulation with ORCHIDEE over the whole(!) spatial domain. At this stage, we need the whole domain as it facilitates the validation of the tool to have the 'true' equilibrium.

We need information from early during the spinup (transient phase) to 
* extract all boundary conditions (i.e. ORCHIDEE forcings) on the spatial resolution of ORCHIDEE. 
* information on NPP and LAI during early spinup (transient phase) vastly improves ML predictions.

We need information from end of the spinup  (steady state)
* to train and test the machine learning tool

The exact simulation lenghts depend on the model version.

### HOW TO UPDATE THE CODE ON GITHUB: you need to do multiple steps: 
* First, "git add" to add all the files that you changed. It is recommended to only add files you have changed and make sure you updated with any changes updated to github since you downloaded your copies.
* Second, "git commit" to commit them to your local copy (a difference between svn and git and is that git has a local copy that you commit to). 
* Third, "git push" to push them to the master repository (here). 
This might help: https://git-scm.com/docs/gittutorial

### Other USEFUL COMMANDS: 
* "git diff" will show you all the changes you have made. 
* "git pull" will update all your local files with what exists in the master code (here). 
* "git status" will show you what files have been modified.





