#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=3900
#SBATCH -p aglarge
#SBATCH --nodes=1
#SBATCH --ntasks=120
#SBATCH --mail-type=NONE
#SBATCH --mail-user=murp1677@umn.edu

JULIA_EXECUTABLE=~/Julia_Nest/julia-1.9.0/bin/julia
JULIA_SCRIPT=~/Users/maddymurphy/Documents/programmable-loop-directionality/scripts/loop-simulation/LoopSimulation.jl
PARAMETER_FILE=$file_path

# Define a function to execute Julia script for a single parameter set
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
    local output_file="/home/dauenha0/murp1677/Cyclic_Dynamics/Results/SimulationOutputSet2/results_Batch${BatchID}.csv"

    $JULIA_EXECUTABLE $JULIA_SCRIPT \
        $alpha_a $alpha_b $alpha_c $beta_a $beta_b $beta_c \
        $gamma_BA $gamma_CA $delta_BA $delta_CA $delBEa \
        $JobID $BatchID $output_file
}

# Read parameters from the parameter file and run Julia in background for each set of parameters
tail -n +2 "$PARAMETER_FILE"|while IFS=, read -r BatchID JobID alpha_a alpha_b alpha_c beta_a beta_b beta_c gamma_BA gamma_CA delta_BA delta_CA delBEa; do 
    run_julia $alpha_a $alpha_b $alpha_c $beta_a $beta_b $beta_c $gamma_BA $gamma_CA $delta_BA $delta_CA $delBEa $JobID $BatchID &
done

# Wait for all background jobs to finish
wait

