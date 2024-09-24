#!/bin/bash

# Base directories
HOME_DIRECTORY=$(git rev-parse --show-toplevel)
BASE_BACKGROUND_DIR="$HOME_DIRECTORY/DCASE_2023_Challenge_Task_7_Dataset/eval"
# BASE_EVAL_DIR="$HOME_DIRECTORY/rvggshesults/audio/mamba_fast_500/epoch-500_step-637037/conditioned_100" # MAMBA500
BASE_EVAL_DIR="$HOME_DIRECTORY/results/audio/og/conditioned_100"

MODEL_NAME="vggish"


# Function to execute the script with substituted arguments
execute_script() {
    local background_dir="$1"
    local eval_dir="$2"
    local model_name="$3"
    local folder_names=(
        "DogBark"
        "Footstep"
        "GunShot"
        "Keyboard"
        "MovingMotorVehicle"
        "Rain"
        "Sneeze_Cough"
    )

    for folder_name in "${folder_names[@]}"; do
        class_background_dir="${background_dir}/${folder_name}"
        class_eval_dir="${eval_dir}/${folder_name}"
        fadtk "$model_name" "$class_background_dir" "$class_eval_dir"
    done
}
# Call the function with directory paths as parameters
execute_script "$BASE_BACKGROUND_DIR" "$BASE_EVAL_DIR" "$MODEL_NAME"
