#! /bin/bash

cd "$(dirname "$0")"

# CHANGEME!
MYTWODEG="twodeg"
# CHANGEME!
MYFORCING="ORCHIDEE_forcing_data"

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

varlist_json="$(readlink -f DEF_Trunk)/varlist.json"

sed -i "s@/home/surface5/vbastri/SPINacc_ref@$FORCING_DATA/vlad_files/vlad_files/@g" $varlist_json
sed -i "s@/home/orchideeshare/igcmg/IGCM/SRF/METEO/CRUJRA/v2.2/twodeg/@$(readlink -f $TWODEG_DATA)/@g" $varlist_json
