# Summary of Changes

- Converted dataset format to xarray.Dataset - merged auxil into packdata, etc.
- Reformat documentation and added more comments
- Rewrote tests as pytest files
- Reformated the config file as a python file
- Used ruff for reformatting and linting
- Added pre-commit checks and a Github workflow for CI checks
- Added input features (std of some variables like Qair, Psurf, etc.)
- Increase Nc values (by a factor of 2)
- Instead of averaging over the whole time span, only take monthly averages and separate data per year to increase dataset size
- Simplified readvar.py
- Added new ML algorithm options: XGBoost, RandomForest, MLP, Lasso, Stacking Ensemble
- Combined all ML evaluation results into a single CSV table
- Implemented multithread parallelization to train a ML model per target variable in parallelization
- Added standard scaling to preprocess the data before ML training
- Updated README.md and CONTRIBUTING.md
- Added explanation of the varlist.json file

## Performance Benchmark

### Separate Years
| Metric | Score              | Algorithm |
|--------|--------------------|-----------|
| R2     | 0.6528848053359372 | bt        |
| slope  | 0.9542832780095561 | bt        |
| R2     | 0.6530125681385772 | rf        |
| slope  | 0.9540960891785613 | rf        |
| R2     | 0.6512312928704228 | gbm       |
| slope  | 0.950301142771462  | gbm       |
| R2     | 0.3712954690899892 | lasso     |
| slope  | 0.9877068027413212 | lasso     |
| R2     | 0.6525452816395577 | stack     |
| slope  | 0.9525996582374604 | stack     |

### Averaged Years
| Metric | Score              | Algorithm |
|--------|--------------------|-----------|
| R2     | 0.31926695871812644| bt        |
| slope  | 0.9591009540297856 | bt        |
| R2     | 0.321636483649443  | rf        |
| slope  | 0.9590895662312189 | rf        |
| R2     | 0.328685618778433  | gbm       |
| slope  | 0.9583401949919101 | gbm       |
| R2     | 0.09302916905772492| lasso     |
| slope  | 0.9930868709808638 | lasso     |
| R2     | 0.3225905817064779 | stack     |
| slope  | 0.9753542956777028 | stack     |
