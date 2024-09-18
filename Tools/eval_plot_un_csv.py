import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



##@param[in]   data_path            config[5] resultpath
##@param[in]   npfts                number of PFT
##@param[in]   ipool                som, biomass, litter
##@param[in]   n_cnp                c-only:1 cn:2 cnp:3
##@param[in]   xTickLabel           The names of subpools
##@param[in]   subLabel             The names of sub-level iterm: ['Cpool']
# xTickLabel = ['Active','Passive','Slow','Surface']



def plot_metric(data_path, npfts, ipool, subLabel, xTickLabel):
   
   # reading the data 
   data = pd.read_csv(data_path + 'MLacc_results.csv' ) 
   # extract the data corresponding to one pool
   extracted_data = data[data['comp'] == ipool]
   if ipool=='som':
      # extract in 1D the metrics of interest
      r2_1D_array = extracted_data['R2'].to_numpy()
      dNRMSE_1D_array = extracted_data['dNRMSE'].to_numpy()
      slope_1D_array = extracted_data['slope'].to_numpy() 
      # check the ind_1 indexation in the .csv file 
      num_of_var=extracted_data['ind_1'].max()
      # reshape the metrics data  
      r2_2d = r2_1D_array.reshape(-1, num_of_var)
      dNRMSE_2d = dNRMSE_1D_array.reshape(-1, num_of_var)
      slope_2d = slope_1D_array.reshape(-1, num_of_var)
   if ipool=='biomass':
      # extract in 1D the metrics of interest 
      r2_1D_array = extracted_data['R2'].to_numpy()
      dNRMSE_1D_array = extracted_data['dNRMSE'].to_numpy()
      slope_1D_array = extracted_data['slope'].to_numpy()
      # check the ind_1 indexation in the .csv file 
      num_of_var=extracted_data['ind_1'].max()-1
      # reshape the metrics data  
      r2_2d = r2_1D_array.reshape(-1, num_of_var).T
      dNRMSE_2d = dNRMSE_1D_array.reshape(-1, num_of_var).T
      slope_2d = slope_1D_array.reshape(-1, num_of_var).T
   if ipool== 'litter':
       # Sorting the DataFrame by 'var' and 'ind_1' and extracting the sorted R2 values
       df_sorted = extracted_data.sort_values(by=['var', 'ind_1']).reset_index(drop=True)
       # Convert the metrics values to 1D arrays
       r2_1D_array = np.array(df_sorted['R2'])
       slope_1D_array = np.array(df_sorted['slope'])
       dNRMSE_1D_array = np.array(df_sorted['dNRMSE'])
       # check the ind_1 indexation in the .csv file 
       num_of_var=len(xTickLabel)
       # reshape the metrics data  
       r2_2d = r2_1D_array.reshape(-1, num_of_var)
       dNRMSE_2d = dNRMSE_1D_array.reshape(-1, num_of_var)
       slope_2d = slope_1D_array.reshape(-1, num_of_var) 
   
   subps=len(xTickLabel)
   yTickLabel = [
           "PFT02",
           "PFT03",
           "PFT04",
           "PFT05",
           "PFT06",
           "PFT07",
           "PFT08",
           "PFT09",
           "PFT10",
           "PFT11",
           "PFT12",
           "PFT13",
           "PFT14",
           "PFT15",
       ]
   yTickLabel = yTickLabel[0:npfts]


#----------------- colors of the maps setting-------------------------------------- 
   fonts = 7
   colors1 = plt.cm.YlGn(np.linspace(0, 1, 128))
   colors2 = plt.cm.YlGn_r(np.linspace(0, 1, 128))
   colors = np.vstack((colors1, colors2))
   mycolor_R2 = ["maroon", "tomato", "gold", "limegreen", "forestgreen"]
   mycolor_slope = [
           "maroon",
           "tomato",
           "gold",
           "limegreen",
           "forestgreen",
           "forestgreen",
           "limegreen",
           "gold",
           "tomato",
           "maroon",
       ]   
   mycolor_rmse = ["forestgreen", "limegreen", "gold", "tomato", "maroon"]
   mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
   mymap_R2 = mcolors.LinearSegmentedColormap.from_list("my_list", mycolor_R2, N=5)
   mymap_slope = mcolors.LinearSegmentedColormap.from_list(
          "my_list", mycolor_slope, N=10
      )
   mymap_rmse = mcolors.LinearSegmentedColormap.from_list("mylist", mycolor_rmse, N=5)
#-------------------------------- plotting ------------------------------------------
   # R2_Cpools
   # if n_cnp ==1:
   fig, axs = plt.subplots(nrows=3, figsize=(8, 18))
#axs[0].imshow(r2_2d, vmin=0.5, vmax=1, cmap=mymap_R2)
   for jj in range(0, subps):
       # print(jj)
       for ii in range(0, npfts):
           # print(R22_n[ii,jj])
          # axs[0].text(-0.5 + jj, ii, str(r2_2d[ii, jj]), size=fonts, color="k")
          axs[0].text(-0.5 + jj, ii, f"{r2_2d[ii, jj]:.2f}", size=fonts, weight="bold", color="k")

   my_x_ticks = np.arange(subps)
   axs[0].set_xticks(my_x_ticks)
   # axs[0].set_xticklabels([""])
   my_y_ticks = np.arange(npfts)
   axs[0].set_yticks(my_y_ticks)
   axs[0].set_yticklabels(yTickLabel)
   axs[0].set_title("R2_" + subLabel[0])
   fig.subplots_adjust(right=0.9)
   l = 0.92
   b = 0.66
   w = 0.015
   h = 0.22
   rect = [l, b, w, h]
   cbar_ax = fig.add_axes(rect)
   sc = axs[0].imshow(r2_2d, vmin=0.5, vmax=1, cmap=mymap_R2)
   plt.colorbar(sc, cax=cbar_ax)

#------------ the other metrics : slope and dNRMSE---------------
   # slope
   axs[1].imshow(slope_2d, vmin=0.75, vmax=1.25, cmap=mymap_slope)
   for jj in range(0, subps):
       for ii in range(0, npfts):
           axs[1].text(
               -0.5 + jj,
               ii,
               f"{slope_2d[ii, jj]:.2f}",
               size=fonts,
               color="k",
               weight="bold",
           )   
   my_x_ticks = np.arange(subps)
   axs[1].set_xticks(my_x_ticks)
   # axs[1].set_xticklabels([""])
   my_y_ticks = np.arange(npfts)
   axs[1].set_yticks(my_y_ticks)
   axs[1].set_yticklabels(yTickLabel)
   axs[1].set_title("slope_" + subLabel[0])
   fig.subplots_adjust(right=0.9)
   l = 0.92
   b = 0.39
   w = 0.015
   h = 0.22
   rect = [l, b, w, h]
   cbar_ax = fig.add_axes(rect)
   sc = axs[1].imshow(slope_2d, vmin=0.75, vmax=1.25, cmap=mymap_slope)
   plt.colorbar(sc, cax=cbar_ax)

# rmse
   axs[2].imshow(dNRMSE_2d, vmin=0, vmax=0.25, cmap=mymap_rmse)
   for jj in range(0, subps):
       for ii in range(0, npfts):
           axs[2].text(
               -0.5 + jj,
               ii,
               f"{dNRMSE_2d[ii, jj]:.2f}",
               size=fonts,
               color="k",
               weight="bold",
            )
   my_x_ticks = np.arange(subps)
   axs[2].set_xticks(my_x_ticks)
   axs[2].set_xticklabels(xTickLabel, rotation=60)
   my_y_ticks = np.arange(npfts)
   axs[2].set_yticks(my_y_ticks)
   axs[2].set_yticklabels(yTickLabel)
   axs[2].set_title("dNRMSE_" + subLabel[0])
   fig.subplots_adjust(right=0.9)
   l = 0.92
   b = 0.12
   w = 0.015
   h = 0.22
   rect = [l, b, w, h]
   cbar_ax = fig.add_axes(rect)
   sc = axs[2].imshow(dNRMSE_2d, vmin=0, vmax=0.25, cmap=mymap_rmse)
   plt.colorbar(sc, cax=cbar_ax)

   plt.savefig(data_path + "Eval_all_" + ipool + subLabel[0] + ".png")

