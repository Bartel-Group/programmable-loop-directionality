import os
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

DATA_DIR = "/Users/noord014/MyDrive/phd/research/dmc-lab/bartel-group/programmable-catalysis/dev/data/pkls"
FIG_DIR = "/Users/noord014/MyDrive/phd/research/dmc-lab/bartel-group/programmable-catalysis/dev/figures"


def load_search_results(filename, data_dir=DATA_DIR):
    return joblib.load(os.path.join(data_dir, filename))


def get_cv_results(search_results):
    return search_results.cv_results_


def get_best_model_idx(cv_results):
    cv_ranking = cv_results["rank_test_score"]
    best_model_idx = np.where(cv_ranking == 1)[0][0]
    return best_model_idx


def get_best_train_val_scores(cv_results, best_model_idx):
    train_scores = []
    val_scores = []

    for i in range(0, 9):
        train_scores.append(cv_results[f"split{i}_train_score"][best_model_idx])
        val_scores.append(cv_results[f"split{i}_test_score"][best_model_idx])

    return train_scores, val_scores


def main():
    search_results = load_search_results("op_steady/xgb_clf_op-random-cv.pkl")
    cv_results = get_cv_results(search_results)
    best_model_idx = get_best_model_idx(cv_results)
    train_scores, val_scores = get_best_train_val_scores(cv_results, best_model_idx)

    fig, ax = plt.subplots()
    ax.plot(train_scores, label="Train")
    ax.plot(val_scores, label="Validation")
    ax.set_xlabel("CV Split")
    ax.set_ylabel("F1 Score (Weighted)")
    ax.legend()

    return None


if __name__ == "__main__":
    main()
