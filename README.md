# optuna-learn

Tuning hyper-parameters based on Optuna is as easy as using scikit-learn.


## :hourglass_flowing_sand: Dependencies

optuna-learn requires:

- python >= 3.6
- scikit-learn 
- optuna 

## :rocket: Installation

```bash
pip install optuna-learn
```

## :zap: Quick Start

```python
from lightgbm import LGBMClassifier
from optlearn.opt import OptunaSearch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

opt = OptunaSearch(
        model=LGBMClassifier,
        optimize_direction='maximize',
        n_trials=100,
        params_dict={
            'n_estimators': ['categorical', 100, 200, 300, 500],
            'reg_alpha': ['float', 0.001, 10, False],
            'reg_lambda': ['float', 0.001, 100, False],
            'num_leaves': ['int', 2, 256],
        }
)

opt.fit(X_train, y_train)

y_pred = opt.predict(X_test)

accuracy_score(y_test, y_pred)
>>> 0.9967924528301886
```