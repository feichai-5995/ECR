#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

scenes=( "7scenes_chess")
# scenes=( "7scenes_chess" "7scenes_fire" "7scenes_heads" "7scenes_office" "7scenes_pumpkin" "7scenes_redkitchen" "7scenes_stairs")
# scenes=("12scenes_apt1_kitchen" "12scenes_apt1_living" "12scenes_apt2_bed" "12scenes_apt2_kitchen" "12scenes_apt2_living" "12scenes_apt2_luke" "12scenes_office1_gates362" "12scenes_office1_gates381" "12scenes_office1_lounge" "12scenes_office1_manolis" "12scenes_office2_5a" "12scenes_office2_5b")


training_exe="${REPO_PATH}/train_ace_test.py"
testing_exe="${REPO_PATH}/test_ace_test.py"

datasets_folder="${REPO_PATH}/datasets/7scenes"
out_dir="${REPO_PATH}/output_test/loss_100_50_1/7scenes"
# mkdir -p "$out_dir"

# datasets_folder="${REPO_PATH}/datasets/12scenes"
# out_dir="${REPO_PATH}/output_final/12scenes"

for scene in ${scenes[*]}; do
  for id in {2..4};do
    python $training_exe "--scene" "$datasets_folder/$scene" "--output_map_file" "$out_dir/$scene"_"$id.pt"
    python $testing_exe "--scene" "$datasets_folder/$scene" "--network" "$out_dir/$scene"_"$id.pt"  2>&1 | tee "$out_dir/log_${scene}.txt"
  done
done

# for scene in ${scenes[*]}; do
#   echo "${scene}: $(cat "${out_dir}/log_${scene}.txt" | tail -5 | head -1)"
# done