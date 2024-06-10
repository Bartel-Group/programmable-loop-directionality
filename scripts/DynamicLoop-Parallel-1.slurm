#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=3900
#SBATCH -p aglarge
#SBATCH --nodes=1
#SBATCH --ntasks=120
#SBATCH --mail-type=NONE
#SBATCH --mail-user=murp1677@umn.edu

JULIA_EXECUTABLE=~/Julia_Nest/julia-1.9.0/bin/julia
JULIA_SCRIPT=~/Cyclic_Dynamics/Code/LoopDynamics/CURRENT_LoopSimulation.jl
PARAMETER_FILE=$file_path

# Define a function to run Julia script for each set of parameters
run_julia() {
    local alpha_a=$1
    local alpha_b=$2
    local alpha_c=$3
    local beta_a=$4
    local beta_b=$5
    local beta_c=$6
    local gamma_BA=$7
    local gamma_CA=$8
    local delta_BA=$9
    local delta_CA=${10}
    local delBEa=${11}
    local JobID=${12}
    local BatchID=${13}
    local output_file="/home/dauenha0/murp1677/Cyclic_Dynamics/Results/SimulationOutputSet2/results_Batch$BatchID.csv"
    
    echo "Running simulation for BatchID: $BatchID, JobID: $JobID"  # Debug statement

    # Run Julia script and redirect stderr to separate file for error capture
    $JULIA_EXECUTABLE $JULIA_SCRIPT $alpha_a $alpha_b $alpha_c $beta_a $beta_b $beta_c $gamma_BA $gamma_CA $delta_BA $delta_CA $delBEa $JobID $BatchID "$output_file" 2> "$output_file.error"
}

# Read parameters from CSV file and run Julia script for each set of parameters
tail -n +2 "$PARAMETER_FILE" | while IFS=, read -r BatchID JobID alpha_a alpha_b alpha_c beta_a beta_b beta_c gamma_BA gamma_CA delta_BA delta_CA delBEa; do 
    run_julia "$alpha_a" "$alpha_b" "$alpha_c" "$beta_a" "$beta_b" "$beta_c" "$gamma_BA" "$gamma_CA" "$delta_BA" "$delta_CA" "$delBEa" "$JobID" "$BatchID" &
done

# Wait for all jobs to finish
wait
