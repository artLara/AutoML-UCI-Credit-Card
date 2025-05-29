"""Data information

ID: ID of each client, categorical variable
LIMIT_BAL: Amount of given credit in NT dollars.
SEX: Gender, categorical variable (1=male, 2=female)
EDUCATION: level of education, categorical variable (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
MARRIAGE: Marital status, categorical variable (1=married, 2=single, 3=others)
AGE: Age in years, numerical variable
"""

import logging

import pandas as pd


logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

def explore(df: pd.DataFrame,
            report,
            log_level: int = 0) -> None:
      """Explore UCI dataset.
      For this specific dataset, the data cleaning to modify is described in the
      next list:

      SEX: Male = 1 and Femal = 0. Delete unknown values
      EDUCATION: Delete unknown values
      MARRIAGE: Delete unknown values

      Keyword arguments:
      df(pd.Dataframe): drataframe of dataset
      
      Returns:
      None
      """
      report.write_text('Descripción del dataset original')
      columns_names = df.columns.tolist()
      data = df.describe().values.tolist()
      data.insert(0, columns_names)
      report.write_table(data)

      ##########Categorical data
      report.write_text('Información de los datos categoricos')
      cat_names = ['SEX', 'EDUCATION', 'MARRIAGE']
      cat_data = df[cat_names].nunique()
      cat_values = [df[col].unique() for col in cat_names]
      cat_values = [' '.join(list(map(str, num))) for num in cat_values]
      cat_data = pd.DataFrame(cat_data)
      cat_data['VALUES'] = cat_values
      cat_list = cat_data.copy().values.tolist()
      cat_list.insert(0, ['COUNTS', 'VALUES'])
      report.write_table(cat_list)

      if log_level > 0:
            logger.info('[Exploration] Dataset description:')
            print(df.describe())

            logger.info('[Exploration] Dataset information:')
            print(df.info())

            logger.info('[Exploration] Categorical data information:')
            print(cat_data)


def clean(df: pd.DataFrame,
          report,
          log_level: int = 0) -> pd.DataFrame:
      """Clean UCI dataset.
      For this specific dataset, the data cleaning to modify is described in the
      next list:

      SEX: Male = 1 and Femal = 0. Delete unknown values
      EDUCATION: Delete unknown values
      MARRIAGE: Delete unknown values

      Keyword arguments:
      df(pd.Dataframe): dataframe of dataset
      report(Report): instance of Report
      log_level(int): loggin lavel

      Returns: DataFrame
      """
      #Apply change to sex, education and marriage
      df = df.drop('ID', axis=1)
      df = df[df['MARRIAGE'] != 0]
      df = df[df['EDUCATION']!= 0]
      df = df[df['EDUCATION']!= 5]
      df = df[df['EDUCATION']!= 6]
      df[df['SEX']==2] = 0

      report.write_text('Información de los datos categoricos cambiados')
      cat_names = ['SEX', 'EDUCATION', 'MARRIAGE']
      cat_data = df[cat_names].nunique()
      cat_values = [df[col].unique() for col in cat_names]
      cat_values = [' '.join(list(map(str, num))) for num in cat_values]
      cat_data = pd.DataFrame(cat_data)
      cat_data['VALUES'] = cat_values
      cat_list = cat_data.copy().values.tolist()
      cat_list.insert(0, ['COUNTS', 'VALUES'])
      report.write_table(cat_list)
      
      if log_level > 0:
            logger.info('[Cleaning] Dataset information:')
            print(df.info())

            logger.info('[Cleaning] Categorical data information:')
            print(cat_data)
            
            
      return df
