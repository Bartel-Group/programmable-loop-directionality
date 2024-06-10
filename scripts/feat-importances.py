import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import array

import xgboost
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.inspection import permutation_importance


DATA_DIR = '/home/dauenha0/murp1677/Cyclic_Dynamics/Code/ML_repo/programmable-catalysis/dev/data/'
FIG_DIR = "/home/dauenha0/murp1677/Cyclic_Dynamics/Code/ML_repo/programmable-catalysis/dev/figures"

def load_model(filename, data_dir=DATA_DIR):
    return joblib.load(os.path.join(data_dir, filename))

def load_data(filename, data_dir=DATA_DIR):
    data_file = os.path.join(data_dir, filename)
    return pd.read_csv(data_file)

def create_bar_plot(y_values, x_values, title="", xlabel="", ylabel="", figname=""):
    fig = plt.figure(figsize=(6,4))
    
    plt.barh(x_values, y_values)
    plt.title(title)
    plt.xlabel(xlabel, fontsize=12)
    #plt.xscale('log')
    plt.ylabel(ylabel, fontsize=12)    
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    
    return None

def feat_imp_direction(model, df, figname):
    """
    Inputs:
        model [XGB Classifier]: 
            Best XGB Classifier model isolated by the grid search
        df [DataFrame]: 
            Simulation data
            
    Outputs: 
    
    
    """
    # append direction to df
    df['direction'] = pd.cut(df['loop-tof'], bins=[-float('inf'), -1e-4, 1e-4, float('inf')], labels=[2, 0, 1])
    
    # identify targets and features
    targets = ['direction']
    non_features = ['steady-state-condition', 'loop-tof']
    
    columns = list(df)
    features = [f for f in columns if f not in targets if f not in non_features]
    
    # Label features and target
    X, y = df[features].values, df[targets].values

    # Split data --> reserving 15% for test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=22)

    # fit the model to training data
    fit = model.fit(X_train, y_train)

    # permuation importances
    result = permutation_importance(fit, X_train, y_train, n_repeats=10,random_state=22)
    
    n_features = len(features)
    
    # dictionary of importances and sorted importances
    feat_importances = {features[i]:result.importances_mean[i] for i in range(n_features)}
    sorted_imp = {k: v for k, v in sorted(feat_importances.items(), key=lambda item: item[1], reverse=True)}
    
    # plot the results
    create_bar_plot(list(sorted_imp.values()), list(sorted_imp.keys()), title="", xlabel="Feature Importance", ylabel="Feature Name", figname=figname)

    return None


def feat_imp_looptof(model, df, figname):
    """
    Inputs:
        model [XGB Classifier]: best estm model
        df [DataFrame]: simulation data
    Outputs: 
        train_mse [Float] : Mean Square Error of the model 
            prediction on the training set
        validation_mse [Float]: Mean Square Error of the model 
            prediction on the validation set
    
    """
    # identify targets and features
    targets = ['loop-tof']
    non_features = ['steady-state-condition', 'direction']
    
    columns = list(df)
    features = [f for f in columns if f not in targets if f not in non_features]
    
    # Label features and target
    df_filtered = df[df['loop-tof'] != 0]
    X, y = df_filtered[features].values, df_filtered[targets].values

    # Split data --> reserving 15% for test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=22)

    y_train = np.abs(y_train)
    y_test = np.abs(y_test)
    
    # fit the model to the log of the training data
    fit = model.fit(X_train, np.log(y_train))

    # permuation importances
    result = permutation_importance(fit, X_train, np.log(y_train), n_repeats=10,random_state=22)
    
    n_features = len(features)
    
    # dictionary of importances and sorted importances
    feat_importances = {features[i]:result.importances_mean[i] for i in range(n_features)}
    sorted_imp = {k: v for k, v in sorted(feat_importances.items(), key=lambda item: item[1], reverse=True)}
    
    # plot the results
    create_bar_plot(list(sorted_imp.values()), list(sorted_imp.keys()), title="", xlabel="Feature Importance", ylabel="Feature Name", figname=figname)

    return None


def main():
    # Original Parameters
    data = load_data("csvs/240313_op_steady.csv")
    
    # direction
    model = load_model("pkls/op_steady/xgb_clf_op-best-estm.pkl")
    feat_imp_direction(model, data, 'xgb_clf_op-feat-importances.tiff')
    
    #loop tof 
    model = load_model("pkls/op_steady/xgb_reg_op-best-estm.pkl")
    feat_imp_looptof(model, data, 'xgb_reg_op-feat-importances.tiff')
    
    # Rate Constants
    data = load_data("csvs/240313_rc_steady.csv")
    
    # direction
    model = load_model("pkls/rc_steady/xgb_reg_rc-best-estm.pkl")
    feat_imp_direction(model, data, 'xgb_clf_rc-feat-importances.tiff')

    #loop tof 
    model = load_model("pkls/rc_steady/xgb_reg_rc-best-estm.pkl")
    feat_imp_looptof(model, data, 'xgb_reg_rc-feat-importances.tiff')

    return None


if __name__ == "__main__":
    main()

    