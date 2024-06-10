# Murphy et. al Catalytic Resonance Theory: Forecasting the Flow of Programmable Catalytic Loops
## FIGURES README

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
This folder include .tiff files for all the figure panels included in the manuscript. The scripts/notebooks used to create each figure are included in the scripts folder.

        
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

Figure 2: Complete Three-Species Loop Simulation Dataset
   A. Filename: LoopTOF_Histogram_logscale.tiff
   Description: A histogram with a log-scale x-axis and y-axis. Histogram plots the outputs of the converged loop simulations to show the loop turnover frequency values observed across the results.
   
   B. Filename: LoopTOF_PiePlot.tiff
   Description: A pie plot showing the distribution of simulation results across the three categories of behavior (zero loop TOF, positve loop TOF, and negative loop TOF)
   
Figure 3: Analysis of Output Loop TOF based on Applied Amplitude
   A. Filename: LoopTOF_Histogram_with_delbea.tiff
      Description: A histogram with a log-scale x-axis and y-axis. Histogram plots the outputs of the converged loop simulations to show the loop turnover frequency values observed across the results. The outputs are sectioned into different bar colors based on the amplitude of catalyst oscillation imposed in the simulation. 
      
   B. Filename: DelBEa_zero_tof.tiff
      Description: A histogram separating the simulations with a zero loop TOF based on the amplitude of catalyst oscillation imposed in the simulation. Demonstrates the realative number of zero loop TOF simulations across each input amplitude.
   
Figure 4: Permutation Feature Importance and SHAP Analysis of the XGBClassifier
   A. Filename: xgb_clf_op-feat-importances.tiff
   Description: Permutation feature importance (PFI) results of the OP-Classifier XGBoost machine learning model.

   B. Filename: xgb_clf_rc-feat-importances.tiff
   Description: PFI results of the RC-Classifier XGBoost machine learning model.
   
   C. Filename: clf-op-shap-summary-plot-dim0.tiff
   Description: SHAP summary plot for dimension zero of the output shap values. Dimension zero corresponds with class zero (zero loop TOF output).
   
   D. Filename: clf-op-shap-summary-plot-dim1.tiff
   Description: SHAP summary plot for dimension one of the output shap values. Dimension one corresponds with class one (positive loop TOF output).
   
   E. Filename: clf-op-shap-summary-plot-dim2.tiff
   Description: SHAP summary plot for dimension two of the output shap values. Dimension two corresponds with class two (negative loop TOF output).
   
Figure 5: Counterfactuals of the OP XGBClassifier Model
   A. 
   Filename: 
   Description: 
   
Figure 6: Parity Plots for the XGBRegressors
   A. Filename: 
   Description: 
   
Figure 7: Permutation Feature Importance and SHAP analysis on the XGBRegressor model
   A. 
   Filename: 
   Description: 