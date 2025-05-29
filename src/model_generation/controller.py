import logging

import pandas as pd
import numpy as np

from src.model_generation.traditional_models.RandomForest import RandomForest
from src.model_generation.traditional_models.SVM import SVM
from src.model_generation.traditional_models.DecisionTree import DecisionTree
# from src.model_generation.deep_learning.FCNN import FCNN

logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(cfg: dict,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray) -> None:
    """Run pipeline based on configuration file
    
    cfg (dict): configuration dict"""
    
    alg = [RandomForest(cfg),
           SVM(cfg),
           DecisionTree(cfg),
        # FCNN(cfg)
           ]
    
    return __get_best_model(alg,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            cfg)
    # rf = RandomForest(cfg)
    # rf.grid_search_fit(X_train, y_train)
    # rf_inf = rf.test(X_test, y_test)

    # svm = SVM(cfg)
    # svm.grid_search_fit(X_train, y_train)
    # svm_inf = svm.test(X_test, y_test)


    

def __get_best_model(algoritthms: list,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     cfg: dict) :
    """search best algorithm using rmse and r2 metrics"""
    best_rmse = np.inf
    best_r2 = -np.inf
    best_info_rmse = None
    best_info_r2 = None

    for alg in algoritthms:
        alg.grid_search_fit(X_train, y_train)
        alg_inf = alg.test(X_test, y_test)
        if best_rmse > alg_inf['rmse']:
            best_rmse = alg_inf['rmse']
            best_info_rmse = alg_inf

        if best_r2 < alg_inf['r2']:
            best_r2 = alg_inf['r2']
            best_info_r2 = alg_inf


        if cfg['run']['logging_level'] > 0:
            alg_name = alg_inf['name']
            logger.info(f'[Controller_MG] Best {alg_name}:{alg_inf}')

    return best_info_rmse, best_info_r2


