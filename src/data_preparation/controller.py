from src.data_preparation.data_cleaning import explore, clean
import pandas as pd

def run_pipeline(cfg: dict, report) -> None:
    """Run pipeline based on configuration file
    
    cfg (dict): configuration dict"""
    ds_dir = cfg['data']['annotations']['full']
    df = pd.read_excel(ds_dir)
    explore(df, report, cfg['run']['logging_level'])
    df = clean(df, report, cfg['run']['logging_level'])

    return df
