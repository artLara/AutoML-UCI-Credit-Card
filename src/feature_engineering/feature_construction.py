import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

from src.utils.io import save_csv
logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize(df: pd.DataFrame,
              report,
              log_level: int = 0) -> None:
    """
    Noramlization of data

    Keyword arguments:
      df(pd.Dataframe): dataframe of dataset
      report(Report): instance of Report
      log_level(int): loggin lavel

      Returns: DataFrame
    """

    scaler = MinMaxScaler() # or StandardScaler()
    columns_names = df.columns.tolist()
    data = scaler.fit_transform(df)
    df_norm = pd.DataFrame(columns=columns_names, data=data)
    if log_level > 0:
        logger.info('[Normalization] Dataset description:')
        print(df_norm.describe())

        logger.info('[Normalization] Dataset information:')
        print(df_norm.info())


    return df_norm


def split_dataframe(df: pd.DataFrame,
                    cfg: dict) -> None:
    """
    Noramlization of data

    Keyword arguments:
      df(pd.Dataframe): dataframe of dataset
      report(Report): instance of Report
      log_level(int): loggin lavel

      Returns: DataFrame
    """
    train_dir = cfg['data']['annotations']['train']
    test_dir = cfg['data']['annotations']['test']
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    if cfg['run']['logging_level'] > 0:
        logger.info(f'[SplitDataframe] train dir:{train_dir}')
        logger.info(f'[SplitDataframe] test dir:{test_dir}')
        print(train_df.head())
        print(test_df.head())

    save_csv(train_df, cfg['data']['annotations']['train'])
    save_csv(test_df, cfg['data']['annotations']['test'])

    return train_df, test_df


def df2array(df: pd.DataFrame) -> np.ndarray:
    y = df['default payment next month'].to_numpy()
    x = df.drop('default payment next month', axis=1).to_numpy()

    return x, y