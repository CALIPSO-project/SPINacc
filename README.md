# SPINacc
A spinup acceleration procedure for land surface models which is model independent.

Documentation of aims, concepts, workflows are found in file: !MISSING!


The SPINacc package includes:
* job
* main.py
* Tools/*
* DEF_*/


HOW TO RUN THE CODE:
1. copy the code to your own directory
2. change the dirpython in job (L4) to the path where the code is copied, and the dirdef to the DEF_* directory you are going to run with
3. adjust the configurations in the DEF_* directories: MLacc.def defines the tasks to do and the executation directory.
3.1 MLacc.def defines the tasks to do and the executation directory: tasks are 1= clustering, 2=training , 3=extrapolation 
3.2 varlist.json defines the specification of the input data: e.g. resolution, state variables to predict, etc.
4. qsub -q long job   
