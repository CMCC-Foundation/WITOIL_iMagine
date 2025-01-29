# set folders
DIR_EXE="$PWD/src/model/bin/"
DIR_SRC="$PWD/src/model/"
NETCDF="$CONDA_PREFIX/"
# print message
echo "============================="
echo "  MEDSLIK-II SOFTWARE ...    "
echo "============================="
echo "MODEL FOLDER $DIR_SRC"
echo "EXEC FOLDER $DIR_EXE"
echo "NETCDF LIBRARIES TAKEN FROM $NETCDF"
echo "============================="
# compile model
echo " -- Compiling model ..."
mkdir -p "$DIR_EXE"
gfortran -I"$NETCDF/include"   -L"$NETCDF/lib"   "$DIR_SRC/simulation.for" -lnetcdf -lnetcdff -o "$DIR_EXE/simulation.exe"
# ***********************************************
# Compiling with enhanced warnings
# Please, decomment the following line if needed
# gfortran -g -fcheck=all -Wall -Wextra -I"$NETCDF/include" -L"$NETCDF/lib" "$DIR_SRC/simulation.for" -lnetcdf -lnetcdff -o "$DIR_EXE/simulation.exe"
# ***********************************************
# print message
echo "============================="
echo "END SUCCESSFULLY"
