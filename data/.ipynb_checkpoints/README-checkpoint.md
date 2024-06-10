# Murphy et. al Catalytic Resonance Theory: Forecasting the Flow of Programmable Catalytic Loops
## DATA README

This readme.md file was generated on 2024-06-10 by Madeline Murphy

-------------------
General Information
-------------------

Data used in Murphy et. al Catalytic Resonance Theory: Forecasting the Flow of Programmable Catalytic Loops

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
This data bank includes a csv file containing all the Parameter inputs for simulations ran in Julia using the script LoopSimulation.jl. Each simulation outputs the parameter set and the resulting loop turnover frequency computed from the simulation results. The input parameters, output turnover frequency, and steady state condition for each simulation are recorded in the simulation outputs file. The reaction parameters were used to determine the rate constants of each simulation, the results were exported to another csv file that included the rate constants, output turnover frequency, and steady state condition.
        
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

input-parameters
   A. Set 1 & Set 2
   Filename: SetA / BatchB (where A denotes the Set number and B denotes the batch number)
   Description: To leverage parallel computing efforts, the input parameters were sectioned into sets and batches. The input-parameter folder includes the 174,312 input parameters for the screening process. Each of these parameter sets was submitted to the LoopSimulation.jl for micro kinetic modeling of the 3-species loop on a programmable catalyst surface.
   
simulation-results
   C. Filename: final_simulation_outputs.csv   
      Description: Parameter inputs for the parameter screen and the resulting loop turnover frequency output and steady state condition of each simulation. Steady state condition denotes whether or not the simulation converged upon a dynamic steady state solution.
   
   D. Filename: final_simulation_outputs_rc.csv
       Description: The parameter inputs for the parameter screen were used to compute the rate constants governing each elementary reactions in the system. This file contains the rate constants for each simulation and the resulting loop turnover frequency output and steady state condition of each. 
