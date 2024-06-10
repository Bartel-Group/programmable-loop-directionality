# Murphy et. al Catalytic Resonance Theory: Forecasting the Flow of Programmable Catalytic Loops
## SCRIPTS README

This readme.md file was generated on 2024-06-10 by Madeline Murphy

-------------------
General Information
-------------------

Figures published in Murphy et. al Catalytic Resonance Theory: Forecasting the Flow of Programmable Catalytic Loops

Principal Investigator Contact Information
    Name: Paul Dauenhauer
            University of Minnesota
            hauer@umn.edu
            Room 484
            421 Washington Avenue SE 
            Minneapolis, MN 55455
    Name: Chris Bartel
            University of Minnesota
            cbartel@umn.edu
            Room 485
            421 Washington Avenue SE 
            Minneapolis, MN 55455

Co-investigator Contact Information:
    Madeline Murphy
            University of Minnesota
            murp1677@umn.edu
            
Date Published or finalized for release

Date of data collection: 2023-07 through 2024-03

--------------------
Overview of the data
--------------------
This folder includes all the files (scripts and notebooks) used to submit simulations to MSI, perform the micro kinetic model simulation, analyze the outputs, train ML models, and interpret the ML models.

        
--------------------------
SHARING/ACCESS INFORMATION
-------------------------- 

Licenses/restrictions placed on the data:

Links to publications that cite or use the data:

Data was derived using the following julia files to run simulations of the reaction on a dynamic catalyst surface:

Terms of Use: Data Repository for the U of Minnesota (DRUM) By using these files, users agree to the Terms of Use. https://conservancy.umn.edu/pages/drum/policies/#terms-of-use


---------------------
DATA & FILE OVERVIEW
---------------------

1. File List

General Data:

Loop Simulation: .jl script to run the three-species loop simulation from 14 input arguments. Two further .jl files hold the necessary supporting functions for the loop simulation [Callbacks.jl and CycleBaseFuncts.jl]
   A. Filename: Callbacks.jl
   Description: Micro kinetic model simulation of a three species loop reaction on a programmable catalyst surface that oscillates with a square waveform. The simulation takes 14 input arguments and exports the input parameters (ARGS 1-13) and the output loop turnover frequency to the filepath defined by ARGS 14.
        14 ARGS passed into the simulation: 
                ARGS[1-3] = alpha --> BEP slope for species A*, B*, C*
                ARGS[4-6] = beta --> BEP offset for species A*, B*, C*
                ARGS[7-8] = gamma --> Linear scaling slope relating BEa to BEb, BEa to BEc
                ARGS[9-10] = delta --> Linear scaling offset. equal binding energy of BEa and BEb, BEa and BEc
                ARGS[11] = delBEa --> amplitude of catalyst oscillation
                ARGS[12] = Job ID
                ARGS[13] = Batch ID
                ARGS[14] = file path --> location for saving output data
   B. Filename: CycleBaseFuncts.jl
   Description: Subfunctions necessary to run the dynamic cycle simulation. Contains the base functions to construct the reaction parameters, species balances, and rate equations needed to define the reaction. 
   C. Filename: Callbacks.jl
   Description: Subfunctions necessary to construct the callbacks used in the dynamic integration of a square waveform catalyst oscillation. 

Simulation Organization: .ipynb files used to generate the parameter sets, organize reruns, and compile data into one joint file.
   A. Filename: CombinationGenerator.ipynb
   Description: For the identified high, medium, and low values of the input parameters, this notebook generates the complete set of combinations and batches them. The sets and batches are then written in that organized order to csv files. These files serve as inputs to the loop simulation, allowing the simualtions to be run in parallel. This notebook also serves to identify simulations that were not run and then regroups the simulations to isolate those that haven't been run yet. This allowed all simulations to be ran.

   B. Filename: AggregateToOneCSV.ipynb
   Description: aggregate_csv_files reads all the files from the input_dir and rights the results to an output_file. Includes cells to separate steady-state and non steady-state data and find parameters to re-run. Includes cells to compute the rate constants from the input parameters and exports the resulting loop turnover frequency appended to rate constant data.

Slurm Submission
   A. Filename: DynamicLoop-Parallel.slurm and DynamicLoop-Parallel-1.slurm
   Description: Slurm script to submit Set2 and Set1 simulations to the slurm workload manager at MSI. Reads a parameter file and submits each row as its own microkinetic simulation.

   B. Filename: MultipleBatchSubmissions.py
   Desctription: Reads through a folder and runs the slurm script for each parameter file in the folder. Used to submit Set1 and Set2 parameters to MSI resources to be modeled by the loop simulation.  

Data Analysis: .ipynb filed used to generate holistic understanding of the dataset and to generate Figures 2 and 3.
   A. Filename: LoopDynamics_DataVisualization.ipynb
   Description: Jupyter notebook used to read the data, analyze the general trends, and generate FIgure 2 and 3 using the function from DataVis.ipynb

   B. Filename: DataVis.ipynb
   Description: Plotting functions used to generate Figure 2 and 3 to depict the distribution of the loop turnover frequencies in the final datset.


Machine Learning: .py and .ipynb files used to develop the machine learning methods to predict loop directionality and the magnitude of the loop turnover frequency. 
   A. Filename: 1-preprocess-splits.py
   Description: Preprocess the dataset for machine learning. Removes non-steady state data and relabels the features.

   B. Filename: 2-xgb_clf_random-cv.py
   Description: cross validation search of hyperparameters for the classification models.

   C. Filename: 2-xgb_reg_random-cv.py
   Description: cross validation search of hyperparameters for the regression models.

   D. Filename: 3-plot-cv-scores.py
   Description: train the models and identify the hyperparameters to produce the best model using the cross validation 
   defined in scripts 2. Print the testing performance scores of each model.

   E. Filename: check_errors.py
   Description: functions to check the errors on the regression and classification models.

   F. Filename: counterfactuals.ipynb
   Description: jupyter notebook used to digest the counterfactual results and generate Figure 5.

   G. Filename: counterfactuals.py
   Description: script used to generate counterfactual data for a specified transiton (i.e. negative loop turnover frequency to positve loop turnover frequency)

   H. Filename: feat-importances.py
   Description: script used to perform permutaion feature importance (PFI) and generate the resulting feature importance graphs. This script is used to generate Figure 4a, 4b, and 7a.

   I. Filename: model-training-and-performance.ipynb
   Description: notebook used to train each model and score its performance. This notebook uses weighted F1 scoring to score the classification models and median absolute error to score the regression models. Used to generate the parity plots (FIgure 6a and 6b) as well as the sequential feature importance plots in the supporting information.

   J. Filename: shap-values.py
   Description: script used to perform SHAP analysis and generate the resulting summary plots. This script is used to generate Figure 4c, 4d, 4e, and 7b.