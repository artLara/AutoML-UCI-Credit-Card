from pathlib import Path
from itertools import chain

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from keypoint_detection.src import helpers
from keypoint_detection.src.data_processing import image_augmentation


class ICDAR2015Challenge1Reader(Sequence):
    def __init__(self, config:dict, split:str='train'):
        # se ocupa la mitad del batch 
        # porque la otra mitad se usa para imágenes negativas
        # para ese caso se espera que el batch_size sea multiplo de 2
        
        assert split in ['train', 'validation', 'test'], "split no válido, solo se permite ['train', 'validation', 'test']"
        assert config['training']['batch_size'] % 2 == 0, "el tamaño del batch debe de ser múltiplo de 2"

        self.input_shape = tuple(config['architecture']['input_shape'][:2])
        
        self.batch_size = config['training']['batch_size'] // 2
        self.data_dir = Path(config['data']['source']['full'])
        assert self.data_dir.exists(), f'{self.data_dir} no existe'
        ann_file = Path(config['data']['annotations'][split])
        assert ann_file.exists(), f'{ann_file} no existe'
        self.data_df = pd.read_csv(ann_file)
        self.data_df = self.data_df.drop('source', axis=1)
        
        negative_images_dir = Path(config['data']['extras']['negative_images_dir'])
        assert negative_images_dir.exists(), f'{negative_images_dir} no existe'

        self.negative_image_files = [file for file in negative_images_dir.rglob('*.*p*g')]

    def __getitem__(self, index):
        rows = self.data_df.iloc[index * self.batch_size : (index + 1) * self.batch_size]

        coord_cols = self.data_df.columns[-8:].values # 4 pares de coordenadas
        keypoints = rows[coord_cols].values
        
        # imágenes que si contienen un documento
        X = [helpers.read_image(self.data_dir / filepath) for filepath in rows['filepath']]

        for i in range(len(X)):
            h,w = X[i].shape[:2]
            coords = keypoints[i]
            # se escalan las coordenadas en el rango [0,1)
            x_coords = [coords[j]/w for j in range(0, 8, 2)]
            y_coords = [coords[j]/h for j in range(1, 8, 2)]
            scaled_coords = list(chain.from_iterable((zip(x_coords, y_coords))))
            
            keypoints[i] = scaled_coords

        # imágenes negativas (sin objeto target)
        batch_neg_imgs = np.random.choice(self.negative_image_files, self.batch_size)
        neg_imgs = [helpers.read_image(imfile) for imfile in batch_neg_imgs]
        
        X = X + neg_imgs
        neg_kp = [[0]*8]*self.batch_size
        y = list(keypoints) + neg_kp
        
        X = [image_augmentation.random_quality(img) for img in X]
        X, y = zip(*[image_augmentation.random_afine_transform(xy[0], xy[1], is_mask=False) for xy in zip(X, y)])
        X = [helpers.process_image(img, self.input_shape, True) for img in X]

        X = np.array(X)
        y = np.array(y)

        return X, y
    
    def __len__(self):
        return len(self.data_df) // (self.batch_size)
    
    def on_epoch_end(self):
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)

