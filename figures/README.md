# Figures

--------------------

Overview of the data
--------------------

This folder includes .png files for figure panels generated specifically for this manuscript. The scripts/notebooks used to create each figure are included in the `machine-learning/scripts` folder.

--------------------------

SHARING/ACCESS INFORMATION
--------------------------

Terms of Use: Data Repository for the U of Minnesota (DRUM) By using these files, users agree to the Terms of Use. <https://conservancy.umn.edu/pages/drum/policies/#terms-of-use>

---------------------

SUBFOLDER/FILE OVERVIEW
---------------------

   Figure 2: Complete Three-Species Loop Simulation Dataset

         A. Filename: TOF_histogram.png
            Description: A histogram with a log-scale x-axis and y-axis. Histogram plots the outputs of the converged loop simulations to show the loop turnover frequency values observed across the results.
         
         B. Filename: TOF_piechart.png
            Description: A pie plot showing the distribution of simulation results across the three categories of behavior (zero loop TOF, positve loop TOF, and negative loop TOF)

   \
   Figure 3: Analysis of Output Loop TOF based on Applied Amplitude (BEA)

         A. Filename: BEA_histogram.png
            Description: A histogram with a linear x-axis and log-scale y-axis. Histogram plots the fraction of outputs of the converged loop simulations to show the loop turnover frequency values observed across the results. The outputs are sectioned into different line colors based on the amplitude of catalyst oscillation imposed in the simulation. The histogram is zoomed to emphasize the larger magnitude loop TOF values.

   \
   Figure 4: Permutation Feature Importance and SHAP Analysis of the XGBClassifier

         A. Filename: class-0-shap-op.png
          Description: SHAP summary plot for dimension zero of the output shap values. Dimension zero corresponds with class zero (zero loop TOF output).

         B. Filename: class-1-shap-op.png
            Description: SHAP summary plot for dimension one of the output shap values. Dimension one corresponds with class one (positive loop TOF output).

         C. Filename: class-2-shap-op.png
            Description: SHAP summary plot for dimension two of the output shap values. Dimension two corresponds with class two (negative loop TOF output).

         D. Filename: combined-clf-shap-op.png
            Description: Combined summary plot which appears in the manuscript. (Manual combination of the three SHAP summary plots for the output classes and the PFI results for the original parameter XGBClassifier model)

         E. Filename: op-clf-feature-importance.png
            Description: Permutation Feature Importance (PFI) results for the original parameter XGBClassifier.

   \
   Figure 5: Counterfactuals of the OP XGBClassifier Model

      A. Filename: mean-perturbations-barplot.png
         Description: counterfactual mean perturbations for the three transformations considered, for each feature, having been normalized to the range of the specified feature.

   \
   Figure 6: Parity Plots for the XGBRegressors

      A. Filename: op-parity-plot.png
         Description: log-scale (hex-bin) parity plot of the prediciton results of the Original Parameter XGB Regressor, with the color of the hex-bins representing the density of points (also shown in the sidebars of the plot).

      B. Filename: rc-parity-plot.png
         Description: log-scale (hex-bin) parity plot of the prediciton results of the Rate Constants XGB Regressor, with the color of the hex-bins representing the density of points (also shown in the sidebars of the plot).

   \
   Figure 7: Permutation Feature Importance and SHAP analysis on the XGBRegressor model

      A. Filename: combined-reg-shap-op.png
         Description: Combined summary plot which appears in the manuscript. (Manual combination of the single SHAP summary plot and the PFI results for the original parameter XGBRegression model)

      B. Filename: op-reg-feature-importance.png
         Description: Permutation Feature Importance (PFI) results for the original parameter XGBRegressor.

      C. Filename: reg-shap-op.png
         Description: SHAP summary plot for the output shap values of the XGBRegressor model.
