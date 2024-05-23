# Input Files details
## Coord Reference File
- **File Name:** SBG_FGSPIN.10Y.ORC22v8034_19101231_stomate_rest.nc
- **Usage:** Used as a reference file for coordinates, likely for geographic data alignment/validation in simulations or processing steps.
- **Variables:** Typically includes spatial coordinates (latitude, longitude), and time steps.

## Climate Data Files
- **Source Path:** /home/orchideeshare/igcmg/IGCM/SRF/METEO/CRUJRA/v2.2/twodeg/
- **File Names Pattern:** crujra_twodeg_v2.2_
- **Usage:** Provides historical climate data for driving ORCHIDEE simulations or for use in machine learning models.
- **Variables:** Temperature (Tair), rainfall (Rainf), snowfall (Snowf), air humidity (Qair), surface pressure (PSurf), downward shortwave radiation (SWdown), and longwave radiation (LWdown)- **Tools Usage:** Referenced in Tools/forcing.py for creating compressed and aligned forcing data.

## PFT Mask File
- **File Name:** FGSPIN.10Y.ORC22v8034_19010101_19101231_1Y_stomate_history.nc
- **Usage:** Employed to derive the plant functional type (PFT) mask, crucial for ecological models
- **Variable:** VEGET_COV_MAX represents maximum vegetation cover, used to threshold and classify different vegetation types.
- **Tools Usage:** Utilized in Tools/ML.py and Tools/train.py for feature extraction and model training processes, influencing the handling of ecological data.

## Data Files useful for prediction
- **File Name:** FGSPIN.10Y.ORC22v8034_19010101_19101231_1Y_stomate_history.nc
- **Variables:** NPP (Net Primary Productivity), LAI (Leaf Area Index)
- **Tools Usage:** Used in machine learning scripts (ML.py and train.py) for extracting features predicting ecological outputs based on historical data.

- **File Name:** SRF_FGSPIN.10Y.ORC22v8034_19101231_sechiba_rest.nc
- **Variables:** clay_frac (soil clay fraction)
- **Tools Usage:** The soil clay fraction variable is crucial for predicting soil-related properties and is used across various Tools scripts for training models simulating soil dynamics.


# Output Variables

- **File Name**: `SBG_FGSPIN.340Y.ORC22v8034_22501231_stomate_rest.nc`, which has been overwritten to add predicted variables (initially it is a file result of the conventional spinup after 340 years with the targeted variables)
- **Location**: Stored at `/home/surface5/vbastri/SPINacc_ref/GLOBAL340Y/`

## Soil Organic Matter (SOM) Pools
- **Variables**: Carbon content
- **Dimensions**:
  - `pft` (Plant Functional Type)
  - `pool` (Active, Passive, Slow)
- **Details**: Carbon content predictions for different SOM pools across various plant functional types.

## Biomass Pools
- **Variables**: Biomass content
- **Dimensions**:
  - `pool` (Leaf, Sapwood Above Ground (SapAB), Sapwood Below Ground (SapBE), Heartwood Above Ground (HeartAB), Heartwood Below Ground (HeartBE), Root, Fruit, Labile)
  - `pft` (Plant Functional Type)
- **Details**: Biomass predictions for different parts of plants across various functional types, segmented into specific pools like leaf biomass, root biomass, etc.

## Litter Pools
- **Variables**: Litter content
- **Dimensions**:
  - `pft` (Plant Functional Type)
  - `pool` (Structural Above Ground (StructuralAB), Structural Below Ground (StructuralBE), Woody Above Ground (WoodyAB), Woody Below Ground (WoodyBE))
- **Details**: Predictions for litter content, detailing the amount of organic material available on the forest floor, differentiated by type and location (above/below ground).
