"""Configuration file for the MLacc tasks."""

logfile = "log.MLacc_Trunk"
tasks = [
    2,
    4,
]  # 1=test clustering, 2=clustering, 3=compress forcing, 4=ML, 5=evaluation
results_dir = "/home/surface10/mrasolon/SPINacc_24_01_05/EXE_DIR/"
reference_dir = "/home/surface10/mrasolon/files_for_zenodo/reference/EXE_DIR/"
start_from_scratch = True
kmeans_clusters = 4
max_kmeans_clusters = 9
random_seed = 1000
leave_one_out_cv = False
repro_test_task_1 = True
repro_test_task_2 = True
repro_test_task_3 = True
repro_test_task_4 = True
