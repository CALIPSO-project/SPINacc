"""Configuration file for the MLacc tasks."""

logfile = "log.MLacc_Trunk"
tasks = [
    # 2,
    4,
    5,
]  # 1=test clustering, 2=clustering, 3=compress forcing, 4=ML, 5=evaluation
results_dir = "./EXE_DIR/"
reference_dir = "/home/surface10/mrasolon/files_for_zenodo/reference/EXE_DIR/"
start_from_scratch = False
take_year_average = False
smote_bat = False
kmeans_clusters = 4
max_kmeans_clusters = 9
random_seed = 1000
algorithms = [
    "best",
    # "gbm",
    # "nn",
    # "bt",
    # "rf",
    # "lasso",
    # "stack",
]  # bt: BaggingTrees, rf: RandomForest, nn: MLPRegressor, gbm: XGBRegressor, lasso: Lasso, stack: StackingRegressor
leave_one_out_cv = False
repro_test_task_1 = False
repro_test_task_2 = False
repro_test_task_3 = False
repro_test_task_4 = False
