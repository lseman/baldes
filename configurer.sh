#!/bin/bash

# Function to extract current cached values from CMakeCache.txt
get_cache_value() {
  local cache_file="build/CMakeCache.txt"
  local param_name=$1
  local default_value=$2

  if [ -f "$cache_file" ]; then
    local cache_value=$(grep -E "^$param_name:" "$cache_file" | cut -d "=" -f 2)
    if [ -n "$cache_value" ]; then
      echo "$cache_value"
    else
      echo "$default_value"
    fi
  else
    echo "$default_value"
  fi
}

# Initialize parameters with cached values or defaults
N_SIZE=$(get_cache_value "N_SIZE" "100")
R_SIZE=$(get_cache_value "R_SIZE" "50")
MAX_SRC_CUTS=$(get_cache_value "MAX_SRC_CUTS" "10")
BUCKET_CAPACITY=$(get_cache_value "BUCKET_CAPACITY" "20")
TIME_INDEX=$(get_cache_value "TIME_INDEX" "1")
DEMAND_INDEX=$(get_cache_value "DEMAND_INDEX" "2")
MAIN_RESOURCES=$(get_cache_value "MAIN_RESOURCES" "Time")
N_ADD=$(get_cache_value "N_ADD" "0")

# Initialize ON/OFF options with cached values or defaults
RIH=$(get_cache_value "RIH" "OFF")
RCC=$(get_cache_value "RCC" "OFF")
SRC3=$(get_cache_value "SRC3" "OFF")
SRC=$(get_cache_value "SRC" "OFF")
GET_TBB=$(get_cache_value "GET_TBB" "OFF")
UNREACHABLE_DOMINANCE=$(get_cache_value "UNREACHABLE_DOMINANCE" "OFF")
SORTED_LABELS=$(get_cache_value "SORTED_LABELS" "OFF")
MCD=$(get_cache_value "MCD" "OFF")
FIX_BUCKETS=$(get_cache_value "FIX_BUCKETS" "OFF")
AVX=$(get_cache_value "AVX" "OFF")
IPM=$(get_cache_value "IPM" "OFF")
TR=$(get_cache_value "TR" "OFF")
STAB=$(get_cache_value "STAB" "ON")
AUGMENTED=$(get_cache_value "AUGMENTED" "ON")
GET_SUITESPARSE=$(get_cache_value "GET_SUITESPARSE" "OFF")
EXACT_RCC=$(get_cache_value "EXACT_RCC" "OFF")
EVRP=$(get_cache_value "EVRP" "OFF")
WITH_PYTHON=$(get_cache_value "WITH_PYTHON" "OFF")
MTW=$(get_cache_value "MTW" "OFF")
PSTEP=$(get_cache_value "PSTEP" "OFF")

# Function to display a submenu for enabling/disabling options
configure_options() {
  local options=$(whiptail --title "Enable/Disable Options" --checklist \
    "Select ON/OFF options:" 20 78 12 \
    "RIH" "Enable RIH compilation option" $RIH \
    "RCC" "Enable RCC compilation option" $RCC \
    "SRC3" "Enable 3SRC compilation option" $SRC3 \
    "SRC" "Enable SRC compilation option" $SRC \
    "GET_TBB" "Enable TBB compilation option" $GET_TBB \
    "UNREACHABLE_DOMINANCE" "Enable Unreachable Dominance option" $UNREACHABLE_DOMINANCE \
    "SORTED_LABELS" "Enable Sorted Labels option" $SORTED_LABELS \
    "MCD" "Enable MCD option" $MCD \
    "FIX_BUCKETS" "Enable Fixed Buckets option" $FIX_BUCKETS \
    "AVX" "Enable AVX compilation option" $AVX \
    "IPM" "Enable IPM compilation option" $IPM \
    "TR" "Enable TR compilation option" $TR \
    "STAB" "Enable STAB compilation option" $STAB \
    "AUGMENTED" "Enable Augmented compilation option" $AUGMENTED \
    "GET_SUITESPARSE" "Enable SuiteSparse option" $GET_SUITESPARSE \
    "EXACT_RCC" "Enable Exact RCC option" $EXACT_RCC \
    "EVRP" "Enable EVRPTW compilation option" $EVRP \
    "WITH_PYTHON" "Enable Python bindings" $WITH_PYTHON \
    "MTW" "Enable MTW compilation option" $MTW \
    "PSTEP" "Enable PSTEP compilation option" $PSTEP \
    3>&1 1>&2 2>&3)

  # Parse the selected options and toggle ON/OFF values
  RIH=$(echo $options | grep -q '"RIH"' && echo "ON" || echo "OFF")
  RCC=$(echo $options | grep -q '"RCC"' && echo "ON" || echo "OFF")
  SRC3=$(echo $options | grep -q '"SRC3"' && echo "ON" || echo "OFF")
  SRC=$(echo $options | grep -q '"SRC"' && echo "ON" || echo "OFF")
  GET_TBB=$(echo $options | grep -q '"GET_TBB"' && echo "ON" || echo "OFF")
  UNREACHABLE_DOMINANCE=$(echo $options | grep -q '"UNREACHABLE_DOMINANCE"' && echo "ON" || echo "OFF")
  SORTED_LABELS=$(echo $options | grep -q '"SORTED_LABELS"' && echo "ON" || echo "OFF")
  MCD=$(echo $options | grep -q '"MCD"' && echo "ON" || echo "OFF")
  FIX_BUCKETS=$(echo $options | grep -q '"FIX_BUCKETS"' && echo "ON" || echo "OFF")
  AVX=$(echo $options | grep -q '"AVX"' && echo "ON" || echo "OFF")
  IPM=$(echo $options | grep -q '"IPM"' && echo "ON" || echo "OFF")
  TR=$(echo $options | grep -q '"TR"' && echo "ON" || echo "OFF")
  STAB=$(echo $options | grep -q '"STAB"' && echo "ON" || echo "OFF")
  AUGMENTED=$(echo $options | grep -q '"AUGMENTED"' && echo "ON" || echo "OFF")
  GET_SUITESPARSE=$(echo $options | grep -q '"GET_SUITESPARSE"' && echo "ON" || echo "OFF")
  EXACT_RCC=$(echo $options | grep -q '"EXACT_RCC"' && echo "ON" || echo "OFF")
  EVRP=$(echo $options | grep -q '"EVRP"' && echo "ON" || echo "OFF")
  WITH_PYTHON=$(echo $options | grep -q '"WITH_PYTHON"' && echo "ON" || echo "OFF")
  MTW=$(echo $options | grep -q '"MTW"' && echo "ON" || echo "OFF")
  PSTEP=$(echo $options | grep -q '"PSTEP"' && echo "ON" || echo "OFF")
}

# Use whiptail for a menu-based interface
while true; do
  SELECTION=$(whiptail --title "BALDES Configuration" --menu "Choose a parameter to configure:" 20 78 12 \
    "N_SIZE" "Set N_SIZE (Current: $N_SIZE)" \
    "R_SIZE" "Set R_SIZE (Current: $R_SIZE)" \
    "MAX_SRC_CUTS" "Set MAX_SRC_CUTS (Current: $MAX_SRC_CUTS)" \
    "BUCKET_CAPACITY" "Set BUCKET_CAPACITY (Current: $BUCKET_CAPACITY)" \
    "TIME_INDEX" "Set TIME_INDEX (Current: $TIME_INDEX)" \
    "DEMAND_INDEX" "Set DEMAND_INDEX (Current: $DEMAND_INDEX)" \
    "MAIN_RESOURCES" "Set MAIN_RESOURCES (Current: $MAIN_RESOURCES)" \
    "N_ADD" "Set N_ADD (Current: $N_ADD)" \
    "" "" \
    "ENABLE/DISABLE" "Configure ON/OFF options" \
    "" "" \
    "Generate" "Finish and generate CMake options" 3>&1 1>&2 2>&3)

  RETVAL=$?

  # Check if the user pressed OK or Cancel
  if [ $RETVAL -ne 0 ]; then
    echo "Configuration cancelled. Exiting without running CMake."
    exit 0
  fi

  case $SELECTION in
  "N_SIZE")
    N_SIZE=$(whiptail --inputbox "Enter value for N_SIZE" 8 78 "$N_SIZE" --title "N_SIZE" 3>&1 1>&2 2>&3)
    ;;
  "R_SIZE")
    R_SIZE=$(whiptail --inputbox "Enter value for R_SIZE" 8 78 "$R_SIZE" --title "R_SIZE" 3>&1 1>&2 2>&3)
    ;;
  "MAX_SRC_CUTS")
    MAX_SRC_CUTS=$(whiptail --inputbox "Enter value for MAX_SRC_CUTS" 8 78 "$MAX_SRC_CUTS" --title "MAX_SRC_CUTS" 3>&1 1>&2 2>&3)
    ;;
  "BUCKET_CAPACITY")
    BUCKET_CAPACITY=$(whiptail --inputbox "Enter value for BUCKET_CAPACITY" 8 78 "$BUCKET_CAPACITY" --title "BUCKET_CAPACITY" 3>&1 1>&2 2>&3)
    ;;
  "TIME_INDEX")
    TIME_INDEX=$(whiptail --inputbox "Enter value for TIME_INDEX" 8 78 "$TIME_INDEX" --title "TIME_INDEX" 3>&1 1>&2 2>&3)
    ;;
  "DEMAND_INDEX")
    DEMAND_INDEX=$(whiptail --inputbox "Enter value for DEMAND_INDEX" 8 78 "$DEMAND_INDEX" --title "DEMAND_INDEX" 3>&1 1>&2 2>&3)
    ;;
  "MAIN_RESOURCES")
    MAIN_RESOURCES=$(whiptail --inputbox "Enter value for MAIN_RESOURCES" 8 78 "$MAIN_RESOURCES" --title "MAIN_RESOURCES" 3>&1 1>&2 2>&3)
    ;;
  "N_ADD")
    N_ADD=$(whiptail --inputbox "Enter value for N_ADD" 8 78 "$N_ADD" --title "N_ADD" 3>&1 1>&2 2>&3)
    ;;
  "ENABLE/DISABLE")
    configure_options
    ;;
  "Generate")
    break
    ;;
  *)
    break
    ;;
  esac
done

# Generate the CMake command with selected parameters and options
cat <<EOF
add_definitions(-DN_SIZE=$N_SIZE)
add_definitions(-DR_SIZE=$R_SIZE)
add_definitions(-DMAX_SRC_CUTS=$MAX_SRC_CUTS)
add_definitions(-DBUCKET_CAPACITY=$BUCKET_CAPACITY)
add_definitions(-DTIME_INDEX=$TIME_INDEX)
add_definitions(-DDEMAND_INDEX=$DEMAND_INDEX)
add_definitions(-DMAIN_RESOURCES=$MAIN_RESOURCES)
add_definitions(-DN_ADD=$N_ADD)
option(RIH "Enable RIH compilation option" $RIH)
option(RCC "Enable RCC compilation option" $RCC)
option(SRC3 "Enable 3SRC compilation option" $SRC3)
option(SRC "Enable SRC compilation option" $SRC)
option(GET_TBB "Enable TBB compilation option" $GET_TBB)
option(UNREACHABLE_DOMINANCE "Enable Unreachable Dominance compilation option" $UNREACHABLE_DOMINANCE)
option(SORTED_LABELS "Enable Sorted Labels compilation option" $SORTED_LABELS)
option(MCD "Enable MCD compilation option" $MCD)
option(FIX_BUCKETS "Enable Fixed Buckets compilation option" $FIX_BUCKETS)
option(AVX "Enable AVX compilation option" $AVX)
option(IPM "Enable IPM compilation option" $IPM)
option(TR "Enable TR compilation option" $TR)
option(STAB "Enable STAB compilation option" $STAB)
option(AUGMENTED "Enable Augmented compilation option" $AUGMENTED)
option(GET_SUITESPARSE "Enable SuiteSparse compilation option" $GET_SUITESPARSE)
option(EXACT_RCC "Enable Exact RCC compilation option" $EXACT_RCC)
option(EVRP "Enable EVRPTW compilation option" $EVRP)
option(WITH_PYTHON "Enable Python bindings" $WITH_PYTHON)
option(MTW "Enable MTW compilation option" $MTW)
option(PSTEP "Enable PSTEP compilation option" $PSTEP)
EOF

# Optionally, run cmake command with these options
echo "Running CMake with the selected parameters..."
cmake -S . - B build -DN_SIZE=$N_SIZE -DR_SIZE=$R_SIZE -DMAX_SRC_CUTS=$MAX_SRC_CUTS -DBUCKET_CAPACITY=$BUCKET_CAPACITY -DTIME_INDEX=$TIME_INDEX -DDEMAND_INDEX=$DEMAND_INDEX -DMAIN_RESOURCES=$MAIN_RESOURCES -DN_ADD=$N_ADD \
  -DRIH=$RIH -DRCC=$RCC -DSRC3=$SRC3 -DSRC=$SRC -DGET_TBB=$GET_TBB -DUNREACHABLE_DOMINANCE=$UNREACHABLE_DOMINANCE -DSORTED_LABELS=$SORTED_LABELS -DMCD=$MCD -DFIX_BUCKETS=$FIX_BUCKETS -DAVX=$AVX \
  -DIPM=$IPM -DTR=$TR -DSTAB=$STAB -DAUGMENTED=$AUGMENTED -DGET_SUITESPARSE=$GET_SUITESPARSE -DEXACT_RCC=$EXACT_RCC -DEVRP=$EVRP -DWITH_PYTHON=$WITH_PYTHON -DMTW=$MTW -DPSTEP=$PSTEP ..
