# %%
from Tools import *
import xarray

dir_def = "DEF_Trunk/"
sys.path.append(dir_def)
import config

# Define task
resultpath = config.results_dir

# Read list of variables
with open(dir_def + "varlist.json", "r") as f:
    varlist = json.loads(f.read())

packdata = xarray.load_dataset(resultpath + "packdata.nc")
responseY = xarray.load_dataset(varlist["resp"]["sourcefile"], decode_times=False)

var_pred_name1 = varlist["pred"]["allname"]
var_pred_name2 = varlist["pred"]["allname_pft"]
var_pred_name = var_pred_name1 + var_pred_name2

IDx = np.load(resultpath + "IDx.npy", allow_pickle=True)
# generate PFT mask
PFT_mask, PFT_mask_lai = genMask.PFT(
    packdata, varlist, varlist["PFTmask"]["pred_thres"]
)

packdata.attrs.update(
    Nv_nopft=len(var_pred_name1),
    Nv_total=len(var_pred_name),
    var_pred_name=var_pred_name,
    Nlat=np.trunc((90 - IDx[:, 0]) / packdata.lat_reso).astype(int),
    Nlon=np.trunc((180 + IDx[:, 1]) / packdata.lon_reso).astype(int),
)

# %%
from collections import defaultdict

Yvar = varlist["resp"]["variables"]
PFT_mask, PFT_mask_lai = genMask.PFT(
    packdata, varlist, varlist["PFTmask"]["pred_thres"]
)
datasets = defaultdict(dict)
labx = ["Y"] + list(packdata.data_vars) + ["pft"]

for pool in Yvar.values():
    for ii in pool:
        for jj in ii["name_prefix"]:
            for kk in ii["loops"][ii["name_loop"]]:
                varname = jj + ("_%2.2i" % kk if kk else "") + ii["name_postfix"]
                if ii["name_loop"] == "pft":
                    ipft = kk
                if ii["dim_loop"] == ["null"] and ipft in ii["skip_loop"]["pft"]:
                    continue
                ivar = responseY[varname].data
                index = itertools.product(*[ii["loops"][ll] for ll in ii["dim_loop"]])
                for ind in index:
                    if "pft" in ii["dim_loop"]:
                        ipft = ind[ii["dim_loop"].index("pft")]
                    if ipft in ii["skip_loop"]["pft"]:
                        continue

                    # extract data
                    extr_var = extract_X.var(packdata, ipft)

                    # extract PFT map
                    pft_ny = extract_X.pft(packdata, PFT_mask_lai, ipft)
                    pft_ny = np.resize(pft_ny, (*extr_var.shape[:-1], 1))

                    # extract Y
                    pool_map = np.squeeze(ivar)[
                        tuple(i - 1 for i in ind)
                    ]  # all indices start from 1, but python loop starts from 0
                    pool_map[pool_map >= 1e20] = np.nan
                    # Y_map[ind[0]] = pool_map

                    if (
                        "format" in varlist["resp"]
                        and varlist["resp"]["format"] == "compressed"
                    ):
                        pool_arr = pool_map.flatten()
                    else:
                        pool_arr = pool_map[packdata.Nlat, packdata.Nlon]
                    extracted_Y = np.resize(pool_arr, (*extr_var.shape[:-1], 1))

                    extr_all = np.concatenate((extracted_Y, extr_var, pft_ny), axis=-1)
                    extr_all = extr_all.reshape(-1, extr_all.shape[-1])

                    df_data = DataFrame(
                        extr_all, columns=labx
                    )  # convert the array into dataframe
                    combine_XY = df_data.dropna().drop(["pft"], axis=1)

                    if jj == "biomass":
                        print(jj, varname, ind)
                        s, i = varname.split("_")
                        ll = f"{s}_{int(ind[0]):02d}"
                        ind = (int(i),)
                        print(jj, ll, ind)
                    else:
                        ll = varname

                    d = datasets[jj].setdefault(ll, {"Y": pd.DataFrame()})
                    d["X"] = combine_XY.drop(columns=["Y"])
                    y = combine_XY["Y"].rename(f"{ind[0]:02d}")
                    d["Y"] = pd.concat([d["Y"], y], axis=1)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

for pool, ds in datasets.items():
    # Plot the size of each dataset in datasets
    dataset_sizes = {k: len(v["X"]) for k, v in ds.items()}
    dataset_sizes_df = pd.DataFrame(
        list(dataset_sizes.items()), columns=["Dataset", "Size"]
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="Size",
        y="Dataset",
        data=dataset_sizes_df.sort_values(by="Size", ascending=False),
    )
    plt.title("Size of Each Dataset")
    plt.xlabel("Number of Samples")
    plt.ylabel("Dataset")
    plt.savefig(f"{pool}_dataset_sizes.png")

# %%
# Load the dataset
# Assuming datasets is a dictionary with keys as variable names and values as DataFrames
# Example: datasets = {('biomass', 14): df1, ('litter_12_be', 1): df2, ...}

# Select a specific dataset for analysis
for pool, ds in datasets.items():
    for k, v in ds.items():
        Y, X = v.values()
        print(X.shape, Y.shape)
        corr = X.corr()
        plt.figure(figsize=(12, 8))
        plt.imshow(corr, cmap="coolwarm", interpolation="none")
        plt.colorbar()
        plt.title(f"Correlation Matrix for {k}")
        plt.savefig(f"{k}_correlation_matrix.png")

# %%
# Load the dataset
# Assuming datasets is a dictionary with keys as variable names and values as DataFrames
# Example: datasets = {('biomass', 14): df1, ('litter_12_be', 1): df2, ...}

# Select a specific dataset for analysis
for pool, ds in datasets.items():
    for k, v in ds.items():
        Y, X = v.values()
        corrs = {}
        for c, y in Y.items():
            corr = X.corrwith(y, axis=0)
            corrs[c] = corr
        corr_matrix = pd.DataFrame(corrs)
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix)
        plt.title(f"Feature-Target Correlation Matrix for {k}")
        plt.savefig(f"{k}_xy_correlation_matrix.png")
        print(k)

# %%
# 4. Visualizations
for pool, ds in datasets.items():
    for k, v in ds.items():
        Y, X = v.values()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        # Train the model
        model = XGBRegressor()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error for {k}: {mse}")

        # Feature importance
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": importance}
        ).sort_values(by="Importance", ascending=False)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
        plt.title(f"Feature Importance for {k}")
        plt.savefig(f"{k}_feature_importance.png")

    # %%
