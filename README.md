# SPINacc
A spinup acceleration procedure for land surface models which is model independent.

Documentation of aims, concepts, workflows are found in file: !MISSING!


The SPINacc package includes:
* job
* main.py
* Tools/*
* DEF_*/


HOW TO RUN THE CODE:
1. adjust the configurations in the DEF_* directories: MLacc.def defines the tasks to do and the executation directory.
1.1 MLacc.def defines the tasks to do and the executation directory: tasks are 1= clustering, 2=training , 3=extrapolation 
1.2 varlist.json defines the specification of the input data: e.g. resolution, state variables to predict, etc.


2. I can run it interactively in ipython with python3 or batch mode, but cannot run it interactively directly in the command line (I think this is due to my cshell is mainly configured with python2.7 and has some incompatibilities)

    2.1. in ipython, one need to first run 'import sys;sys.path.append(#the path you put the code#/Tools)', and then 'run -i main.py 'DEF_X' '

    2.2. in batch mode, one can just submit the job: qsub -q short job

    2.3 if you want to plot the results (i.e. itask = 100 or 200), one can only run it in ipython. (I don't know how to make plots in batch mode...)

