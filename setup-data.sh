#! /bin/bash

# CHANGEME!
MYTWODEG="twodeg"
# Default is in the SPINacc home directory - CHANGEME if preferred.
MYFORCING="ORCHIDEE_forcing_data"

cd "$(dirname "$0")"

# Check if a DEF_Trunk path is provided as an argument; otherwise, use the default
if [ -n "$1" ]; then
  DEF_TRUNK_PATH="$1"
else
  DEF_TRUNK_PATH="DEF_Trunk"
fi

# Resolve the full path for varlist_json
varlist_json="$(readlink -f "$DEF_TRUNK_PATH")/varlist.json"



TWODEG_DATA=$(readlink -f $MYTWODEG)

FORCING_DATA=$(readlink -f $MYFORCING)

if [ -e "ORCHIDEE_forcing_data.zip" ]; then
  echo "File exists."
else
  echo "Downloading forcing data"
  wget https://zenodo.org/records/10514124/files/ORCHIDEE_forcing_data.zip
  # unzip Reproducibility_tests_reference.zip
  # will extract to ./ORCHIDEE_forcing_data/vlad_files/vlad_files
  unzip ORCHIDEE_forcing_data.zip -d $FORCING_DATA
fi

sed -i "s@/home/surface5/vbastri/SPINacc_ref@$FORCING_DATA/vlad_files/vlad_files/@g" $varlist_json
sed -i "s@/home/orchideeshare/igcmg/IGCM/SRF/METEO/CRUJRA/v2.2/twodeg/@$(readlink -f $TWODEG_DATA)/@g" $varlist_json
