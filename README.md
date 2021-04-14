# SPINacc
A spinup acceleration procedure for land surface models which is model independent.

Documentation of underlying concepts and the workflow are found in file: !MISSING!

The SPINacc package includes:
* job
* main.py
* Tools/*
* DEF_*/ : folders which contains specifications for different model versions (CNP, trunk 2.0, MICT, CNP-MIMICS) as well as simulation setup (e.g. forcing data)

========================================================

HOW TO RUN THE CODE:

Prerequisite: Currently we only support the use of the tool on the LSCE IT infrastructure (i.e. obelix):

1. copy the code to your own directory (this can be done from the command line with "git clone https://github.com/dsgoll123/SPINacc").

2. change the dirpython in job (L4) to the path where the code is copied, and the dirdef to the DEF_* directory where the configuration file (e.g., MLacc.def) is located.

3. adjust the configurations in the DEF_* directories: MLacc.def defines the tasks to do and the executation directory.

    3.1 MLacc.def defines the tasks to do and the executation directory: tasks are 1= clustering, 2=training , 3=extrapolation.

    3.2 varlist.json defines the specification of the input data: e.g. resolution, state variables to predict, etc.

4. qsub -q long job   

=======================================================

INFORMATION FOR CODE DEVELOPERS:

1. HOW TO UPDATE THE CODE ON GITHUB: 
you need to do multiple steps:
First, "git add" to add all the files that you changed.
Second, "git commit" to commit them to your local copy (a difference between svn and git and is that git has a local copy that you commit to).
Third, "git push" to push them to the master repository (here).
This might help: https://git-scm.com/docs/gittutorial

2. USEFUL COMMANDS:
"git diff" will show you all the changes you have made.
"git pull" will update all your local files with what exists in the master code (here).
"git status" will show you what files have been modified.
