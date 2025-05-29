import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
from tensorflow.keras import layers as ly
from tensorflow.keras import Input, Model
from keras.layers import Dense, Flatten
from keras import models

import numpy as np

logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


class FCNN():
    def __init__(self,
                 cfg: dict):
        
        self.epochs = cfg['algorithms']['regressor_nn']['epochs']
        self.batch_size = cfg['algorithms']['regressor_nn']['batch_size']
        self.model = self.__build(cfg)

    def grid_search_fit(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray) -> None:
        """Not implementation of grid search yet"""
        logger.info(f'[FCNN] Training...')
        X_train = np.array(X_train, dtype=np.float32)

        y_train=np.array(y_train)
        # y_train = y_train.reshape(len(y_train), 1)
        logger.info(f'[FCNN] {X_train[0].shape[0]}')
        logger.info(f'[FCNN] {y_train.shape}')

        history = self.model.fit(X_train,
                        validation_data=y_train,
                        epochs=self.epochs,
                        batch_size=self.batch_size)
        logger.info(f'[FCNN] Training complete')


    def predict(self,
                X_test):
        
        y_pred = self.model.predict(X_test)
        return y_pred

    def test(self,
             X_test: np.ndarray,
             y_test: np.ndarray) -> dict:
        """Test of the best FCNN found by Grid Search CV algorithm"""
        info = {}
        info['name'] = 'FCNN'
        # results = self.model.evaluate(X_test, return_dict=True)
        y_pred = self.predict(X_test)
        print(y_pred.shape)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # print(classification_report(y_test, y_pred))
        info['params'] = 'Check configuration yaml file'
        info['rmse'] = rmse
        info['r2'] = r2

        return info
    
    def __build(self, cfg:dict) -> Model:
        
        """
        Creates a Full Conected architecture.

        Args:
        cfg(dict): archivo de configuracion.

        Returns:
        A Keras Model instance.
        """
        input_shape = cfg['algorithms']['regressor_nn']['architecture']['input_shape'][0]
        fc_layers = cfg['algorithms']['regressor_nn']['architecture']['fc']
        loss = cfg['algorithms']['regressor_nn']['loss']
        metrics = cfg['algorithms']['metrics']
        
        # Full conected layers
        model = models.Sequential()
        model.add(Input(shape=np.array([input_shape])))
        for idx, fl in enumerate(fc_layers):
            neurons = fl['neurons']
            activation = fl['activation']            
            model.add(Dense(neurons, activation=activation))
            # if idx == 0:
            #     x = ly.Dense(units=neurons,
            #                  activation=activation,
            #                  input_dim=input_shape)
            #     continue

            # x = ly.Dense(units=neurons,
            #              activation=activation)(x)
        
        # model = Model(inputs = inputs,
        #             outputs = x,
        #             name = 'FC-NN') 
        
        print(metrics)
        model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
        
        model.summary()
        
        return model
