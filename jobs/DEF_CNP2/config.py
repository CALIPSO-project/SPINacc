"""Configuration file for the MLacc tasks."""

logfile = "log.MLacc_CNP2"
tasks = [
#    1,
#    2,
#    3,
    4,
#    5,
]  # 1=test clustering, 2=clustering, 3=compress forcing, 4=ML, 5=evaluation

kmeans_clusters = 4
max_kmeans_clusters = 9
random_seed = 1000
algorithms = [
    "bt",
    # "rf",
    # "gbm",
    # "nn",
    # "ridge",
    # "best",
]  # bt: BaggingTrees, rf: RandomForest, nn: MLPRegressor, gbm: XGBRegressor, lasso: Lasso, best: SelectBestModel

start_from_scratch = False

# To modify:
take_year_average = False  # Performance improved when set to True
smote_bat = True
model_out = False  # Optional
parallel = False  # True by default
sel_most_PFT_sites = False
old_cluster = True  # True by default
# obsolete take_unique = True

# Output and testing
results_dir = "./EXE_DIR_CNP2.TEST.UNIQUE/"
reference_dir = "./SPINacc-results/"
leave_one_out_cv = False
repro_test_task_1 = False
repro_test_task_2 = False
repro_test_task_3 = False
repro_test_task_4 = False
