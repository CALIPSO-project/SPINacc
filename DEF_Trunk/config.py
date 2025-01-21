"""Configuration file for the MLacc tasks."""

logfile = "log.MLacc_Trunk"
tasks = [
    1,
    2,
    # 3,
    4,
    5,
]  # 1=test clustering, 2=clustering, 3=compress forcing, 4=ML, 5=evaluation

kmeans_clusters = 4
max_kmeans_clusters = 9
random_seed = 1000
algorithms = [
    # "bt",
    # "rf",
    # "gbm",
    # "nn",
    # "ridge",
    "best",
]  # bt: BaggingTrees, rf: RandomForest, nn: MLPRegressor, gbm: XGBRegressor, lasso: Lasso, best: SelectBestModel

start_from_scratch = False

# To modify:
take_year_average = False  # Performance improved when set to True
smote_bat = True
model_out = False  # Optional
parallel = True  # True by default
take_unique = False

# Output and testing
results_dir = "./EXE_DIR/"
reference_dir = "/home/surface10/mrasolon/files_for_zenodo/reference/EXE_DIR/"
leave_one_out_cv = False
repro_test_task_1 = False
repro_test_task_2 = False
repro_test_task_3 = False
repro_test_task_4 = False
