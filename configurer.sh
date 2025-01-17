#!/usr/bin/env bash

# Ensure we're running in bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script requires bash to run. Please use: bash $0"
    exit 1
fi

set -e  # Exit on error

# Constants
CACHE_FILE="build/CMakeCache.txt"
SCRIPT_NAME=$(basename "$0")

# Color constants (only if terminal supports it)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

# Helper functions
error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${GREEN}$1${NC}"
}

warn() {
    echo -e "${YELLOW}Warning: $1${NC}"
}

# Function to extract current cached values from CMakeCache.txt
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

# Initialize numeric parameters
declare -A config_params
config_params=(
    ["N_SIZE"]="102"
    ["R_SIZE"]="1"
    ["MAX_SRC_CUTS"]="50"
    ["BUCKET_CAPACITY"]="100"
    ["TIME_INDEX"]="0"
    ["N_ADD"]="10"
    ["HGS_TIME"]="2"
    ["GUROBI_VERSION_MAJOR"]="120"
)

# Load cached values for numeric parameters
for param in "${!config_params[@]}"; do
    config_params[$param]=$(get_cache_value "$param" "${config_params[$param]}")
done

# Initialize boolean options
declare -A options
options=(
    ["RIH"]="OFF"
    ["RCC"]="ON"
    ["SRC"]="ON"
    ["UNREACHABLE_DOMINANCE"]="OFF"
    ["SORTED_LABELS"]="ON"
    ["MCD"]="OFF"
    ["FIX_BUCKETS"]="ON"
    ["IPM"]="ON"
    ["TR"]="OFF"
    ["EXACT_RCC"]="OFF"
    ["EVRP"]="OFF"
    ["WITH_PYTHON"]="OFF"
    ["MTW"]="OFF"
    ["SCHRODINGER"]="OFF"
    ["CLIQUER"]="OFF"
    ["VERBOSE"]="OFF"
    ["JEMALLOC"]="ON"
    ["GUROBI"]="OFF"
    ["HIGHS"]="ON"
    ["NSYNC"]="OFF"
    ["CHOLMOD"]="OFF"
    ["IPM_ACEL"]="OFF"
    ["BALDES"]="ON"
    ["HGS"]="OFF"
    ["TCMALLOC"]="OFF"
)

# Load cached values for boolean options
for opt in "${!options[@]}"; do
    options[$opt]=$(get_cache_value "$opt" "${options[$opt]}")
done

# Function to configure build type
configure_build_type() {
    local current_build_type
    current_build_type=$(get_cache_value "CMAKE_BUILD_TYPE" "Debug")
    local build_type
    build_type=$(whiptail --title "Build Type" \
        --menu "Choose build type:" 15 60 4 \
        "Debug" "Build with debugging information" \
        "Release" "Build with optimizations" \
        "RelWithDebInfo" "Release with debug info" \
        "MinSizeRel" "Minimum size release" \
        3>&1 1>&2 2>&3) || return 1

    mkdir -p build
    echo "CMAKE_BUILD_TYPE=$build_type" > build/BuildType.txt
    info "Build type set to: $build_type"
}



# Main menu interface
while true; do
    SELECTION=$(whiptail --title "BALDES Configuration" \
        --menu "Choose an option to configure:" 20 78 12 \
        "BUILD_TYPE" "Set build type" \
        "PARAMETERS" "Configure numeric parameters" \
        "OPTIONS" "Configure build options" \
        "Generate" "Finish and generate CMake options" \
        3>&1 1>&2 2>&3) || {
        warn "Configuration cancelled. Exiting without running CMake."
        exit 0
    }

    case $SELECTION in
        "BUILD_TYPE")
            configure_build_type
            ;;
        "PARAMETERS")
            while true; do
                PARAM_SELECTION=$(whiptail --title "Numeric Parameters" \
                    --menu "Choose a parameter to configure:" 20 78 10 \
                    "N_SIZE" "Set N_SIZE (Current: ${config_params[N_SIZE]})" \
                    "R_SIZE" "Set R_SIZE (Current: ${config_params[R_SIZE]})" \
                    "MAX_SRC_CUTS" "Set MAX_SRC_CUTS (Current: ${config_params[MAX_SRC_CUTS]})" \
                    "BUCKET_CAPACITY" "Set BUCKET_CAPACITY (Current: ${config_params[BUCKET_CAPACITY]})" \
                    "TIME_INDEX" "Set TIME_INDEX (Current: ${config_params[TIME_INDEX]})" \
                    "N_ADD" "Set N_ADD (Current: ${config_params[N_ADD]})" \
                    "HGS_TIME" "Set HGS_TIME (Current: ${config_params[HGS_TIME]})" \
                    "Back" "Return to main menu" \
                    3>&1 1>&2 2>&3) || break

                [ "$PARAM_SELECTION" = "Back" ] && break

                new_value=$(whiptail --inputbox "Enter value for $PARAM_SELECTION" \
                    8 78 "${config_params[$PARAM_SELECTION]}" \
                    --title "$PARAM_SELECTION" \
                    3>&1 1>&2 2>&3) || continue

                validate_numeric "$new_value" "$PARAM_SELECTION" 1
                config_params[$PARAM_SELECTION]=$new_value
            done
            ;;
        "OPTIONS")
            while true; do
                OPT_SELECTION=$(whiptail --title "Build Options" \
                    --menu "Configure options:" 24 78 15 \
                    "RIH" "RIH [${options[RIH]}]" \
                    "RCC" "RCC [${options[RCC]}]" \
                    "SRC" "SRC [${options[SRC]}]" \
                    "UNREACHABLE_DOMINANCE" "Unreachable Dominance [${options[UNREACHABLE_DOMINANCE]}]" \
                    "SORTED_LABELS" "Sorted Labels [${options[SORTED_LABELS]}]" \
                    "MCD" "MCD [${options[MCD]}]" \
                    "FIX_BUCKETS" "Fixed Buckets [${options[FIX_BUCKETS]}]" \
                    "IPM" "IPM [${options[IPM]}]" \
                    "TR" "TR [${options[TR]}]" \
                    "EXACT_RCC" "Exact RCC [${options[EXACT_RCC]}]" \
                    "WITH_PYTHON" "Python Bindings [${options[WITH_PYTHON]}]" \
                    "VERBOSE" "Verbose Output [${options[VERBOSE]}]" \
                    "JEMALLOC" "Use Jemalloc [${options[JEMALLOC]}]" \
                    "TCMALLOC" "Use TCMalloc [${options[TCMALLOC]}]" \
                    "GUROBI" "Gurobi [${options[GUROBI]}]" \
                    "HIGHS" "HiGHS [${options[HIGHS]}]" \
                    "NSYNC" "nsync [${options[NSYNC]}]" \
                    "BALDES" "BALDES [${options[BALDES]}]" \
                    "Back" "Return to main menu" \
                    3>&1 1>&2 2>&3) || break

                [ "$OPT_SELECTION" = "Back" ] && break

                # Toggle the selected option
                if [ "${options[$OPT_SELECTION]}" = "ON" ]; then
                    options[$OPT_SELECTION]="OFF"
                else
                    options[$OPT_SELECTION]="ON"
                fi

                # Handle special cases
                if [ "$OPT_SELECTION" = "IPM" ]; then
                    if [ "${options[IPM]}" = "OFF" ]; then
                        info "IPM is disabled, enabling STAB"
                        options["STAB"]="ON"
                    else
                        info "IPM is enabled, disabling STAB"
                        options["STAB"]="OFF"
                    fi
                fi
            done
            ;;
        "Generate")
            break
            ;;
    esac
done

# Generate and execute CMake command
cmake_command="cmake -S . -B build"

# Add build type if set
if [ -f "build/BuildType.txt" ]; then
    build_type=$(cat build/BuildType.txt)
    cmake_command+=" $build_type"
fi

# Add configuration parameters
for param in "${!config_params[@]}"; do
    cmake_command+=" -D$param=${config_params[$param]}"
done

# Add options
for opt in "${!options[@]}"; do
    cmake_command+=" -D$opt=${options[$opt]}"
done

# Create build directory if it doesn't exist
mkdir -p build

# Print configuration summary
info "Running CMake with the following configuration:"
if [ -f "build/BuildType.txt" ]; then
    echo "Build Type: $(cat build/BuildType.txt)"
fi
echo "Parameters:"
for param in "${!config_params[@]}"; do
    echo "  $param = ${config_params[$param]}"
done
echo "Options:"
for opt in "${!options[@]}"; do
    echo "  $opt = ${options[$opt]}"
done

info "\nExecuting: $cmake_command"
eval "$cmake_command"
