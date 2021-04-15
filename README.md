# SPINacc
A spinup acceleration procedure for land surface models which is model independent.

Documentation of aims, concepts, workflows are found in file: !MISSING!


The SPINacc package includes:
* job
* main.py
* Tools/*
* DEF_*/


HOW TO RUN THE CODE:
1. copy the code to your own directory (this can be done from the command line with "git clone https://github.com/dsgoll123/SPINacc").

2. change the dirpython in job (L4) to the path where the code is copied, and the dirdef to the DEF_* directory where the configuration file (e.g., MLacc.def) is located.

3. adjust the configurations in the DEF_* directories: MLacc.def defines the tasks to do and the executation directory.

    3.1 MLacc.def defines the tasks to do and the executation directory: tasks are 1= clustering, 2=training , 3=extrapolation.

    3.2 varlist.json defines the specification of the input data: e.g. resolution, state variables to predict, etc.

4. qsub -q long job   



How to modify the varlist.json files:
You need to modify the sourcepath & sourcefile for the required variables.
-climate: climate forcing data, e.g. CRUJRA-1901-1920
-pred: other predictor variables used for the ML
	var1: NPP and LAI with pre-run for X years, *_stomate_history.nc
	var2 - var4: soil properties and/or nutrient-related variables, which vary with the model versions. In principle, those variables with original spatial resolution (half deg) are in the forcing files. When you apply the ML to a case with 2 deg resolution, you can use the values in the *_stomate_rest.nc or *_sechiba_rest.nc with pre-run for 1 or X years.
	For var2 - var4, if there is no output in *_rest.nc (e.g. N and P deposition), please use the values in *_stomate_history.nc with pre-run 1 year to replace.
-PFTmask: max PFT fractions, which was used to mask grids with extreme low PFT fractions. It is usually in the *_stomate_history.nc (VEGET_COV_MAX).
-resp: response variables, i.e. the equlibrium pools from the traditional spin-up , which were used for extracting site data and global evaluation, e.g. SOC. They are usually in the *_stomate_history.nc of the last year of the whole spin-up.

You can find the detailed information for each variable in the Trunk and CNP examples: DEF_Trunk/varlist.json, DEF_CNP/varlist.json 
You can create your varlist.json according to your case.



