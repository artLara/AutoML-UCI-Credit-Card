from pathlib import Path

from tensorflow.keras.utils import Sequence, set_random_seed
import numpy as np
import pandas as pd
import cv2

from keypoint_detection.src import helpers
from keypoint_detection.src.data_processing import image_augmentation


class RegressionDataset(Sequence):
    def __init__(self, 
                 config,
                 split:str='train'):
        
        assert split in ['train', 'validation', 'test'], "split no válido, solo se permite ['train', 'validation', 'test']"
        assert config['training']['batch_size'] % 2 == 0, "el tamaño del batch debe de ser múltiplo de 2"
        assert config['data']['target_type'] in ['masks', 'heatmaps'], "target vtype no valido solo se permite ['masks', 'heatmaps']"
        assert 0 <= config['training']['rd_thresh'] <= 1, "rd_thresh debe ser entre 0 y 1"
        self.rd_thresh = config['training']['rd_thresh']
        self.target_type = config['data']['target_type']
        self.batch_size = config['training']['batch_size'] // 2
        self.data_dir = Path(config['data']['source']['full'])
        if config['data']['source'][split] is not None:
            self.data_dir = Path(config['data']['source'][split])

        assert self.data_dir.exists(), f'{self.data_dir} no existe'
        ann_file = Path(config['data']['annotations'][split])
        assert ann_file.exists(), f'{ann_file} no existe'
        self.data_df = pd.read_csv(ann_file)
        # self.data_df = self.data_df.drop('source', axis=1)
        self.input_shape = tuple(config['architecture']['input_shape'])
        self.output_shape = tuple(config['architecture']['output_shape'])
        self.num_epochs = config['training']['epochs']
        self.current_epoch = 0
        self.remove_corner_rate = config['training']['remove_corner_rate']
        self.black_mask = self.__get_black_mask()
        assert not self.black_mask is None, "Mask is none"
        self.NOT_DOCUMENT = 'none'
        self.split = split
        self.rgb_flag = False if self.output_shape == 1 else True
        set_random_seed(42)
        self.on_epoch_end()

    def __len__(self):
        return len(self.data_df) // self.batch_size

    def __get_black_mask(self):
        return np.zeros(self.input_shape, dtype=np.uint8)
    
    def on_epoch_end(self):
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)
        self.current_epoch += 1

    def __truncate_coord(self, coord: float):
        return min(1, max(0, coord))

    def __getitem__(self, index):  
        rows = self.data_df.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        idx_kps = self.data_df.columns.get_indexer(['x1'])[0]
        # print(f'idx_kps: {idx_kps}')
        images = []
        kps = []

        for _,row in rows.iterrows():
            file_name = Path(row['filename'])
            im_file = self.data_dir / file_name.parent / 'images' / file_name.name
            coords = row['x1':'y4'].values
            if row['class'] == self.NOT_DOCUMENT:
                im_file = self.data_dir / row['filename']
                coords = [0] * 8
            
            coords = list(map(self.__truncate_coord, coords))
            img = cv2.imread(str(im_file), cv2.IMREAD_COLOR)
            img = helpers.img_preprocessing(img)
            if self.split != 'test':
                img = image_augmentation.random_quality(img)
                img, coords = image_augmentation.random_afine_transform(img,
                                                                        coords,
                                                                        is_mask=False,
                                                                        rd_thresh=self.rd_thresh)

            images.append(img)
            kps.append(coords)
        
        images = np.array(images)
        kps = np.array(kps, dtype=np.float64)

        return images, kps