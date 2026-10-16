#!/usr/bin/env bash

# Ensure we're running in bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script requires bash to run. Please use: bash $0"
    exit 1
fi

set -e  # Exit on error

# Constants
CACHE_FILE="build/CMakeCache.txt"
CONFIG_H_IN="config.h.in"
SCRIPT_NAME=$(basename "$0")
LOG_FILE="build/config.log"

# Custom color scheme for whiptail
export NEWT_COLORS='
root=,black
window=,lightgray
border=black,lightgray
textbox=black,lightgray
button=black,cyan
listbox=black,lightgray
actlistbox=white,cyan
actsellistbox=white,black
'

# Helper functions
error() {
    echo -e "\033[0;31mError: $1\033[0m" >&2
    exit 1
}

info() {
    echo -e "\033[0;32m$1\033[0m"
}

warn() {
    echo -e "\033[1;33mWarning: $1\033[0m"
}

# Function to extract options and parameters from config.h.in
parse_config_h_in() {
    local config_file=$1

    if [ ! -f "$config_file" ]; then
        error "Config file $config_file not found!"
    fi

    # Extract #cmakedefine options
    options=($(grep -oP '(?<=#cmakedefine )\w+' "$config_file"))

    # Extract @VARIABLE@ parameters
    parameters=($(grep -oP '(?<=@)\w+(?=@)' "$config_file"))

    # Print extracted options and parameters for debugging
    info "Extracted options: ${options[*]}"
    info "Extracted parameters: ${parameters[*]}"
}

# Function to get cached value from CMakeCache.txt
get_cache_value() {
    local param_name=$1
    local default_value=$2

    if [ -f "$CACHE_FILE" ]; then
        local value
        value=$(grep -E "^$param_name:" "$CACHE_FILE" | cut -d "=" -f 2) || true
        if [ -n "$value" ]; then
            echo "$value"
        else
            echo "$default_value"
        fi
    else
        echo "$default_value"
    fi
}

# Validate numeric input
validate_numeric() {
    local value=$1
    local param_name=$2
    local min_value=${3:-0}

    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        error "$param_name must be a positive integer"
    fi

    if [ "$value" -lt "$min_value" ]; then
        error "$param_name must be greater than or equal to $min_value"
    fi
}

# Parse config.h.in to extract options and parameters
parse_config_h_in "$CONFIG_H_IN"

# Initialize options and parameters
declare -A config_options
declare -A config_parameters

# Set default values for options (OFF by default)
for opt in "${options[@]}"; do
    config_options[$opt]=$(get_cache_value "$opt" "OFF")
done

# Set default values for parameters (extract from config.h.in or use defaults)
declare -A default_parameters=(
    ["R_SIZE"]="1"
    ["N_SIZE"]="102"
    ["BUCKET_CAPACITY"]="100"
    ["TIME_INDEX"]="0"
    ["DEMAND_INDEX"]="0"
    ["N_ADD"]="10"
    ["HGS_TIME"]="2"
)

for param in "${parameters[@]}"; do
    config_parameters[$param]=$(get_cache_value "$param" "${default_parameters[$param]:-0}")
done

# Main menu interface
while true; do
    SELECTION=$(whiptail --title "Configuration Menu" \
        --menu "Choose an option to configure:" 20 78 12 \
        "OPTIONS" "Configure build options" \
        "PARAMETERS" "Configure numeric parameters" \
        "Generate" "Finish and generate CMake options" \
        3>&1 1>&2 2>&3) || {
        warn "Configuration cancelled. Exiting without running CMake."
        exit 0
    }

    case $SELECTION in
        "OPTIONS")
            while true; do
                OPT_SELECTION=$(whiptail --title "Build Options" \
                    --menu "Configure options:" 24 78 15 \
                    $(for opt in "${options[@]}"; do echo "$opt" "${config_options[$opt]}"; done) \
                    "Back" "Return to main menu" \
                    3>&1 1>&2 2>&3) || break

                [ "$OPT_SELECTION" = "Back" ] && break

                # Toggle the selected option
                if [ "${config_options[$OPT_SELECTION]}" = "ON" ]; then
                    config_options[$OPT_SELECTION]="OFF"
                else
                    config_options[$OPT_SELECTION]="ON"
                fi
            done
            ;;
        "PARAMETERS")
            while true; do
                PARAM_SELECTION=$(whiptail --title "Numeric Parameters" \
                    --menu "Choose a parameter to configure:" 20 78 10 \
                    $(for param in "${parameters[@]}"; do echo "$param" "${config_parameters[$param]}"; done) \
                    "Back" "Return to main menu" \
                    3>&1 1>&2 2>&3) || break

                [ "$PARAM_SELECTION" = "Back" ] && break

                new_value=$(whiptail --inputbox "Enter value for $PARAM_SELECTION" \
                    8 78 "${config_parameters[$PARAM_SELECTION]}" \
                    --title "$PARAM_SELECTION" \
                    3>&1 1>&2 2>&3) || continue

                validate_numeric "$new_value" "$PARAM_SELECTION" 1
                config_parameters[$PARAM_SELECTION]=$new_value
            done
            ;;
        "Generate")
            break
            ;;
    esac
done

# Generate and execute CMake command
cmake_command="cmake -S . -B build"

# Add options
for opt in "${!config_options[@]}"; do
    cmake_command+=" -D$opt=${config_options[$opt]}"
done

# Add parameters
for param in "${!config_parameters[@]}"; do
    cmake_command+=" -D$param=${config_parameters[$param]}"
done

# Create build directory if it doesn't exist
mkdir -p build

# Print configuration summary
info "Running CMake with the following configuration:"
echo "Options:"
for opt in "${!config_options[@]}"; do
    echo "  $opt = ${config_options[$opt]}"
done
echo "Parameters:"
for param in "${!config_parameters[@]}"; do
    echo "  $param = ${config_parameters[$param]}"
done

info "\nExecuting: $cmake_command"
eval "$cmake_command"
