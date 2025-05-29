from pathlib import Path

from tensorflow.keras.utils import Sequence, set_random_seed
import numpy as np
import pandas as pd
import cv2

from keypoint_detection.src import helpers
from keypoint_detection.src.data_processing import image_augmentation
from keypoint_detection.src.domain import enums


class SegmentationDataset(Sequence):
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
        self.rgb_flag = False if len(self.output_shape) == 2 else True
        set_random_seed(42)
        self.on_epoch_end()

    def __len__(self):
        return len(self.data_df) // self.batch_size

    def __get_black_mask(self):
        return np.zeros(self.output_shape, dtype=np.uint8)
    
    def on_epoch_end(self):
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)
        self.current_epoch += 1

    def __getitem__(self, index):  
        rows = self.data_df.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        images = []
        masks = []

        for _,row in rows.iterrows():
            file_name = Path(row['filename'])
            im_file = self.data_dir / row['filename']
            mask = self.black_mask
            if row['class'] != self.NOT_DOCUMENT:
                im_file = self.data_dir / file_name.parent / 'images' / file_name.name
                mask_file = self.data_dir / file_name.parent / self.target_type / file_name.name
                mask = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)
                
            img = cv2.imread(str(im_file), cv2.IMREAD_COLOR)
            img = helpers.img_preprocessing(img)
            mask = helpers.img_preprocessing(mask,
                                             rgb=self.rgb_flag)
                
            if self.split != 'test':
                img = image_augmentation.random_quality(img)
                img, mask = image_augmentation.random_afine_transform(img,
                                                                      mask,
                                                                      is_mask=True,
                                                                      rd_thresh=self.rd_thresh)
            if self.rgb_flag:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
            masks.append(mask)
            images.append(img)
        
        images = np.array(images)
        masks = np.array(masks)
                
        return images, masks
    


class IdCardMasksReader(Sequence):
    """Lee las imágenes de las tarjetasd e identificación 
    y las máscaras de segmentación del objeto completo (a diferencia de la segmentación de esquinas solamente)
    para una tarea de segemntación semántica"""
    def __init__(self, conf:dict, split:str, for_net_input=True):

        assert split in ["train", "validation", "test"], f"split '{split}' not supported try either 'train', 'validation' or 'test'"
        
        self.data_df = pd.read_csv(conf["data"]["annotations"][split])
        self.data_root_dir = Path(conf["data"]["source"]["full"])
        self.conf = conf
        self.batch_size = conf["training"]["batch_size"]
        self.input_shape = conf["architecture"]["input_shape"]
        self.for_net_input = for_net_input

    def __len__(self):
        return (len(self.data_df) // self.batch_size) + 1
    
    def on_epoch_end(self):
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):  
        rows = self.data_df.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        images = []
        masks = []

        for _,row in rows.iterrows():
            im_file = self.data_root_dir / row['filename']
            img = cv2.imread(str(im_file), cv2.IMREAD_COLOR)
            
            mask_file = self.data_root_dir / Path(row['filename']).parent.parent / 'masks' / Path(row['filename']).name
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

            if img.shape[0] > 700: # muy grande, se reescala para optimizar memoria
                """se reescala la imagen a 700px de alto manteniendo la relación de aspecto"""
                img = cv2.resize(img, (int(img.shape[1] * 700 / img.shape[0]), 700))
                mask = cv2.resize(mask, (int(mask.shape[1] * 700 / mask.shape[0]), 700))
            
            img, mask = image_augmentation.augment_image_and_mask(img, mask)

            if self.for_net_input:
                mask = helpers.process_image(mask, self.input_shape[:2],
                                             floating_model=True,
                                             scale_type=enums.ScaleType.ZERO_ONE)
                mask = np.expand_dims(mask, axis=-1)
                img = helpers.process_image(img, self.input_shape[:2],
                                             floating_model=True,
                                             scale_type=enums.ScaleType.ONE_ONE)
            
            masks.append(mask)
            images.append(img)
        
        images = np.array(images)
        masks = np.array(masks)        
        return images, masks
    
