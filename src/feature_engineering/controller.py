import logging

import pandas as pd

from src.feature_engineering.feature_construction import normalize,split_dataframe, df2array


logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(cfg: dict,
                 df: pd.DataFrame) -> None:
    """Run pipeline based on configuration file
    
    cfg (dict): configuration dict"""
    df = normalize(df, cfg['run']['logging_level'])
    train_df, test_df = split_dataframe(df, cfg)

    X_train, y_train = df2array(train_df)
    X_test, y_test = df2array(test_df)

    if cfg['run']['logging_level'] > 0:
        logger.info(f'[Controller_FE] X_train shape:{X_train.shape}')
        logger.info(f'[Controller_FE] y_train shape:{y_train.shape}')
        logger.info(f'[Controller_FE] X_test shape:{X_test.shape}')
        logger.info(f'[Controller_FE] y_test shape:{y_test.shape}')

    return X_train, y_train, X_test, y_test