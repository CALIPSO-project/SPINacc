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

### Separated Years
| Algorithm | R2                 | slope              |
|-----------|--------------------|--------------------|
| bt        | 0.6528848053359372 | 0.9542832780095561 |
| rf        | **0.6530125681385772** | 0.9540960891785613 |
| gbm       | 0.6512312928704228 | 0.950301142771462  |
| lasso     | 0.3712954690899892 | **0.9877068027413212** |
| stack     | 0.6525452816395577 | 0.9525996582374604 |

### Averaged Years
| Algorithm | R2                 | slope              |
|-----------|--------------------|--------------------|
| bt        | 0.31926695871812644| 0.9591009540297856 |
| rf        | 0.321636483649443  | 0.9590895662312189 |
| gbm       | **0.328685618778433**  | 0.9583401949919101 |
| lasso     | 0.09302916905772492| **0.9930868709808638** |
| stack     | 0.3225905817064779 | 0.9753542956777028 |
