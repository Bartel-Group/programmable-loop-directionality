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
   Figure 4: Permutation Feature Importance and SHAP Analysis of the RF Classifier and XGB Classifier Models

         A. Filename: {model_architecture}-class-0-shap-op.png
          Description: SHAP summary plot for dimension zero of the output shap values. Dimension zero corresponds with class zero (zero loop TOF output).

         B. Filename: {model_architecture}-class-1-shap-op.png
            Description: SHAP summary plot for dimension one of the output shap values. Dimension one corresponds with class one (positive loop TOF output).

         C. Filename: {model_architecture}-class-2-shap-op.png
            Description: SHAP summary plot for dimension two of the output shap values. Dimension two corresponds with class two (negative loop TOF output).

         D. Filename: {model_architecture}-combined-op-clf-shap.png
            Description: Combined summary plot which appears in the manuscript. (Manual combination of the three SHAP summary plots for the output classes and the PFI results for the original parameter classifier model with the given architecture)

         E. Filename: {model_architecture}-op-clf-feature-importance.png
            Description: Permutation Feature Importance (PFI) results for the original parameter classifier with the given architecture.

   \
   Figure 5: Counterfactuals of the OP- RF Classifier and XGB Classifier Models

      A. Filename: {model_architecture}-mean-perturbations-barplot.png
         Description: Counterfactual mean perturbations for the three transformations considered, for each feature, having been normalized to the range of the specified feature. The bar plot shows the mean perturbation values for the classifier model with the given architecture.

   \
   Figure 6: Parity Plots for the RF Regressor and XGB Regressor Models

      A. Filename: {model_architecture}-combined-parity-plot.png
         Description: Combined parity plot which appears in the manuscript/SI. (Manual combination of the parity plots for the original parameter regressor and the rate constants regressor with the given architecture)
      
      B. Filename: {model_architecture}-op-parity-plot.png
         Description: Log-scale (hex-bin) parity plot of the prediciton results of the Original Parameter regressor with the given architecture. The color of the hex-bins represents the density of points (also shown in the sidebars of the plot).

      C. Filename: {model_architecture}-rc-parity-plot.png
         Description: Log-scale (hex-bin) parity plot of the prediciton results of the Rate Constants regressor with the given architecture. The color of the hex-bins representing the density of points (also shown in the sidebars of the plot).

   \
   Figure 7: Permutation Feature Importance and SHAP Analysis on the RF Regressor and XGB Regressor Models

      A. Filename: {model_architecture}-combined-op-reg-shap.png
         Description: Combined summary plot which appears in the manuscript/SI. (Manual combination of the single SHAP summary plot and the PFI results for the original parameter regression model with the given architecture)

      B. Filename: {model_architecture}-op-reg-feature-importance.png
         Description: Permutation Feature Importance (PFI) results for the original parameter regressor with the given architecture.

      C. Filename: {model_architecture}-reg-shap-op.png
         Description: SHAP summary plot for the output shap values of the regressor model with the given architecture.
