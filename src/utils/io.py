import logging

import pandas as pd


def save_csv(df: pd.DataFrame,
             output_dir: str) -> None:      
    """Guarda la lista de rows en un CSV con columnas de la constante DF_COLUMNS
    
    Keyword arguments:
    data_rows(list): Lista de filas
    output_dir(str): Direccion donde se guardan los csv.
    """     
    csv_name = str(output_dir)
    df.to_csv(csv_name, sep=',', index=False)