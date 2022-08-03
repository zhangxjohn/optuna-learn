import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

class OptunaSearch:
    """

    Parameters
    ----------
    model: a class, e.g., LGBMRegressor.
    params_dict: dict, key is the name of the hyperparameter to be searched,
        and value is a list of the form [type_of_key, hyperparameter, property].
        For example,
        params_dict={
            'n_estimators': ['categorical', 100, 200, 300, 500],
            'reg_alpha': ['float', 0.001, 10, False],
            'num_leaves: ['int', 2, 256],
        }
    optimize_direction: str, 'minimize' or 'maximize'.
    n_trials: int, default 10.
    """

    def __init__(self,
                 model,
                 params_dict,
                 optimize_direction='minimize',
                 n_trials=10) -> None:
        self.model = model
        self.params_dict = params_dict
        self.n_trials = n_trials
        self.optimize_direction = optimize_direction
        self.best_estimator_ = None

    def objective(self, X, y):

        def func(trial):
            train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

            if self.params_dict is not None:
                params = {}
                for k, v in self.params_dict.items():
                    if v[0] == 'float':
                        assert len(v[1:]) == 3, f"{k} value suggest: [name, min, max, log_bool]"
                        params[k] = trial.suggest_float(k, v[1], v[2], log=v[3])
                    if v[0] == 'int':
                        assert len(v[1:]) == 2, f'{k} value suggest: [name, min, max]'
                        params[k] = trial.suggest_int(k, v[1], v[2])
                    if v[0] == 'categorical':
                        assert len(v[1:]) > 2, f'{k} value suggest: [name, v0, v1, v2, ...]'
                        params[k] = trial.suggest_categorical(k, v[1:])
            else:
                raise ValueError('params_dict can not be None')

            gbm = self.model(**params)
            gbm.fit(train_x, train_y)
            pred_y = gbm.predict(test_x)

            if self.optimize_direction == 'minimize':
                score = mean_squared_error(test_y, pred_y, squared=False)
            else:
                score = accuracy_score(test_y, pred_y)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return score

        return func

    def fit(self, X, y, **fit_params):
        study = optuna.create_study(direction=self.optimize_direction)
        study.optimize(self.objective(X, y), n_trials=self.n_trials)
        best_params = study.best_trial.params
        self.best_estimator_ = self.model(**best_params)
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)