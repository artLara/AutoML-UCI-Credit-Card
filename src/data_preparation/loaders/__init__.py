from . import icdar2015, segmentation, regression

def get_data_loader(config:str, 
                    split:str='',
                    specific_name: str = ''):
    name = config['data']['loader']
    if specific_name != '':
        name = specific_name

    if name == 'icdar2015ch1':
        return icdar2015.ICDAR2015Challenge1Reader(config, split)
    elif name == 'segmentation':
        return segmentation.SegmentationDataset(config, split)
    elif name == 'id_card_segmentation':
        return segmentation.IdCardMasksReader(config, split)
    elif name == 'regression':
        return regression.RegressionDataset(config, split)
    else:
        raise ValueError(f'data loader type {name} is not supported')