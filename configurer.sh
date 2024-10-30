#!/bin/bash

# Function to extract current cached values from CMakeCache.txt
get_cache_value() {
  local cache_file="build/CMakeCache.txt"
  local param_name=$1
  local default_value=$2

  if [ -f "$cache_file" ]; then
    grep -E "^$param_name:" "$cache_file" | cut -d "=" -f 2 || echo "$default_value"
  else
    echo "$default_value"
  fi
}

# Initialize parameters with cached values or defaults
declare -A config_params=(
  ["N_SIZE"]="100" ["R_SIZE"]="50" ["BUCKET_CAPACITY"]="20"
  ["TIME_INDEX"]="1" ["N_ADD"]="0"
)

for param in "${!config_params[@]}"; do
  config_params[$param]=$(get_cache_value "$param" "${config_params[$param]}")
done

declare -A options=(
  ["RIH"]="OFF" ["RCC"]="ON" ["SRC"]="ON"
  ["SORTED_LABELS"]="OFF" ["MCD"]="OFF" ["FIX_BUCKETS"]="OFF"
  ["AVX"]="OFF" ["IPM"]="ON" ["TR"]="OFF" ["STAB"]="OFF" ["AUGMENTED"]="ON"
  ["EXACT_RCC"]="OFF" ["WITH_PYTHON"]="OFF"
  ["PSTEP"]="OFF" ["GUROBI"]="OFF" ["HIGHS"]="ON"
)

for opt in "${!options[@]}"; do
  options[$opt]=$(get_cache_value "$opt" "${options[$opt]}")
done

# Function to display a submenu for enabling/disabling options
configure_options() {
  local options_list=()
  for opt in "${!options[@]}"; do
    options_list+=("$opt" "Enable ${opt} option" "${options[$opt]}")
  done

  local selected_options=$(whiptail --title "Enable/Disable Options" --checklist \
    "Select ON/OFF options:" 20 78 12 "${options_list[@]}" 3>&1 1>&2 2>&3)

  # Parse selected options and update ON/OFF values
  for opt in "${!options[@]}"; do
    if echo "$selected_options" | grep -q "\"$opt\""; then
      options[$opt]="ON"
    else
      options[$opt]="OFF"
    fi
  done
}

# Use whiptail for a menu-based interface
while true; do
  SELECTION=$(whiptail --title "BALDES Configuration" --menu "Choose a parameter to configure:" 20 78 12 \
    "N_SIZE" "Set N_SIZE (Current: ${config_params[N_SIZE]})" \
    "R_SIZE" "Set R_SIZE (Current: ${config_params[R_SIZE]})" \
    "BUCKET_CAPACITY" "Set BUCKET_CAPACITY (Current: ${config_params[BUCKET_CAPACITY]})" \
    "TIME_INDEX" "Set TIME_INDEX (Current: ${config_params[TIME_INDEX]})" \
    "N_ADD" "Set N_ADD (Current: ${config_params[N_ADD]})" \
    "" "" \
    "ENABLE/DISABLE" "Configure ON/OFF options" \
    "" "" \
    "Generate" "Finish and generate CMake options" 3>&1 1>&2 2>&3)

  RETVAL=$?

  if [ $RETVAL -ne 0 ]; then
    echo "Configuration cancelled. Exiting without running CMake."
    exit 0
  fi

  case $SELECTION in
  "N_SIZE")
    config_params[N_SIZE]=$(whiptail --inputbox "Enter value for N_SIZE" 8 78 "${config_params[N_SIZE]}" --title "N_SIZE" 3>&1 1>&2 2>&3)
    ;;
  "R_SIZE")
    config_params[R_SIZE]=$(whiptail --inputbox "Enter value for R_SIZE" 8 78 "${config_params[R_SIZE]}" --title "R_SIZE" 3>&1 1>&2 2>&3)
    ;;
  "BUCKET_CAPACITY")
    config_params[BUCKET_CAPACITY]=$(whiptail --inputbox "Enter value for BUCKET_CAPACITY" 8 78 "${config_params[BUCKET_CAPACITY]}" --title "BUCKET_CAPACITY" 3>&1 1>&2 2>&3)
    ;;
  "TIME_INDEX")
    config_params[TIME_INDEX]=$(whiptail --inputbox "Enter value for TIME_INDEX" 8 78 "${config_params[TIME_INDEX]}" --title "TIME_INDEX" 3>&1 1>&2 2>&3)
    ;;
  "N_ADD")
    config_params[N_ADD]=$(whiptail --inputbox "Enter value for N_ADD" 8 78 "${config_params[N_ADD]}" --title "N_ADD" 3>&1 1>&2 2>&3)
    ;;
  "ENABLE/DISABLE")
    configure_options
    ;;
  "Generate")
    break
    ;;
  esac
done

# Generate the CMake command with selected parameters and options
cmake_command="cmake -S . -B build"
for param in "${!config_params[@]}"; do
  cmake_command+=" -D$param=${config_params[$param]}"
done
for opt in "${!options[@]}"; do
  cmake_command+=" -D$opt=${options[$opt]}"
done

echo "Running CMake with the selected parameters..."
echo "$cmake_command"
$cmake_command
