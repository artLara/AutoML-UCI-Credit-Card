import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, root_mean_squared_error, r2_score
import numpy as np

logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForest():
    def __init__(self,
                 cfg: dict):
        
        param_grid = {
            'n_estimators': cfg['algorithms']['random_forest']['n_estimators'],
            'max_features': cfg['algorithms']['random_forest']['max_features'],
            'max_depth' : cfg['algorithms']['random_forest']['max_depth'],
            'criterion' :cfg['algorithms']['random_forest']['criterion']
        }
        self.grid = GridSearchCV(RandomForestClassifier(),
                                 param_grid,
                                 refit = True,
                                 verbose = 0)

    def grid_search_fit(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray) -> None:
        """Train and search the best RandomForest using Grid Search CV algorithm"""
        logger.info(f'[Random_Forest] Training...')
        self.grid.fit(X_train, y_train)
        logger.info(f'[Random_Forest] Training complete')


    def predict(self,
                X_test):
        
        y_pred = self.grid.predict(X_test)
        return y_pred

    def test(self,
             X_test: np.ndarray,
             y_test: np.ndarray) -> dict:
        """Test of the best RandomForest found by Grid Search CV algorithm"""
        info = {}
        info['name'] = 'Random Forest'
        y_pred = self.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # print(classification_report(y_test, y_pred))
        info['params'] = self.grid.best_params_
        info['rmse'] = rmse
        info['r2'] = r2

        return info