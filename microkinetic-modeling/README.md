# Microkinetic Modeling of a Three-Species Loop Reaction Network

--------------------

Overview of the data
--------------------

This folder includes scripts and data files used to generate the microkinetic model results for the three-species loop reaction network

--------------------------

SHARING/ACCESS INFORMATION
--------------------------

Terms of Use: Data Repository for the U of Minnesota (DRUM) By using these files, users agree to the Terms of Use. <https://conservancy.umn.edu/pages/drum/policies/#terms-of-use>

---------------------

SUBFOLDER/FILE OVERVIEW
---------------------

   Loop simulation and submission: Scripts used to run the three-species loop simulation from 14 input arguments. Callbacks.jl and CycleBaseFuncts.jl hold the necessary supporting functions for the loop simulation.

         A. Filename: Callbacks.jl
            Description: Subfunctions necessary to construct the callbacks used in the dynamic integration of a square waveform catalyst oscillation.
         
         B. Filename: CycleBaseFuncts.jl
            Description: Subfunctions necessary to run the dynamic cycle simulation. Contains the base functions to construct the reaction parameters, species balances, and rate equations needed to define the reaction.

         C. Filename: DynamicLoop-Parallel.slurm
            Description: Sample slurm script used to submit Set1 and Set2 (see `microkinetic-modeling/simulation-input`) simulations to the slurm workload manager at MSI. Reads a parameter file and submits each row as its own microkinetic simulation.
      
         D. Filename: LoopSimulation.jl
            Description: Main script used to run the three-species loop simulation. Micro kinetic model simulation of a three species loop reaction on a programmable catalyst surface that oscillates with a square waveform. The simulation takes 14 input arguments and exports the input parameters (ARGS 1-13) and the output loop turnover frequency to the filepath defined by ARGS 14. 14 ARGS passed into the simulation: ARGS[1-3] = alpha --> BEP slope for species A*, B*, C* ARGS[4-6] = beta --> BEP offset for species A*, B*, C* ARGS[7-8] = gamma --> Linear scaling slope relating BEa to BEb, BEa to BEc ARGS[9-10] = delta --> Linear scaling offset. equal binding energy of BEa and BEb, BEa and BEc ARGS[11] = delBEa --> amplitude of catalyst oscillation ARGS[12] = Job ID ARGS[13] = Batch ID ARGS[14] = file path --> location for saving output data
         
         E. Filename: MultipleBatchSubmission.py
            Description: Reads through a folder and runs the slurm script for each parameter file in the folder. Used to submit Set1 and Set2 parameters to MSI resources to be modeled by the loop simulation.
   \
   Simulation input: Contains the parameter files used to run the loop simulation. Set1 and Set2 are the two sets of parameters used in the loop simulation and have been further divided into batches.

         A. Filename: Set # / Batch #.csv (e.g. Set1/Batch1.csv)
            Description: To leverage parallel computing efforts, the input parameters were sectioned into sets and batches. The input-parameter folder includes the 174,312 input parameters for the screening process. Each of these parameter sets was submitted to the LoopSimulation.jl for micro kinetic modeling of the 3-species loop on a programmable catalyst surface.

   \
   Simulation organization: Jupyter notebook files used to generate the parameter sets, organize reruns, and compile data into one joint file.

         A. Filename: AggregateToOneCSV.ipynb
            Description: aggregate_csv_files reads all the files from the input_dir and rights the results to an output_file. Includes cells to separate steady-state and non steady-state data and find parameters to re-run. Includes cells to compute the rate constants from the input parameters and exports the resulting loop turnover frequency appended to rate constant data.

         B. Filename: CombinationGenerator.ipynb
            Description: For the identified high, medium, and low values of the input parameters, this notebook generates the complete set of combinations and batches them. The sets and batches are then written in that organized order to csv files. These files serve as inputs to the loop simulation, allowing the simualtions to be run in parallel. This notebook also serves to identify simulations that were not run and then regroups the simulations to isolate those that haven't been run yet. This allowed all simulations to be ran.

   \
   Simulation output: Contains the aggregated output files from the loop simulation for the original parameters and rate constants.

         A. Filename: final_simulation_output_rc.csv
            Description: The parameter inputs for the parameter screen were used to compute the rate constants governing each elementary reactions in the system. This file contains the rate constants for each simulation and the resulting loop turnover frequency output and steady state condition of each.
         
         B. Filename: final_simulation_output.csv
            Description: Parameter inputs for the parameter screen and the resulting loop turnover frequency output and steady state condition of each simulation. Steady state condition denotes whether or not the simulation converged upon a dynamic steady state solution.
