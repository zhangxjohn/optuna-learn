import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, make_scorer

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
    n_trials: int, default 10.
    optimize_direction: str, 'min', 'minimize', 'max', 'maximize'.
    sampler: A sampler object that implements background algorithm for value suggestion.
        If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used during
        single-objective optimization and :class:`~optuna.samplers.NSGAIISampler` during
        multi-objective optimization. See also :class:`~optuna.samplers`.
    pruner: A pruner object that decides early stopping of unpromising trials. If :obj:`None`
        is specified, :class:`~optuna.pruners.MedianPruner` is used as the default. See
        also :class:`~optuna.pruners`.
    reward_func: function from sklearn.metrics.
    greater_is_better: bool, when reward_func is not None, it must be fixed.
    """
    def __init__(self,
                 model,
                 params_dict,
                 cv=3,
                 n_trials=10,
                 optimize_direction='minimize',
                 sampler=None,
                 pruner=None,
                 reward_func=None,
                 greater_is_better=None
        ) -> None:
        self.model = model
        self.params_dict = params_dict
        self.cv = cv
        self.n_trials = n_trials
        self.optimize_direction = self._optimize_direction(optimize_direction)
        self.sampler = sampler
        self.pruner = pruner
        self.reward_func = reward_func
        self.greater_is_better = greater_is_better

        self.best_estimator_ = None
        self.best_estimator_params = None

    def _optimize_direction(self, odir):
        odir = odir.lower()
        if odir in ['min', 'minimize']:
            odir = 'minimize'
        elif odir in ['max', 'maximize']:
            odir = 'maximize'
        else:
            raise ValueError("Only support: 'min', 'minimize', 'max', 'maximize'")
        return odir

    def objective(self, X, y):

        def func(trial):
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

            model = self.model(**params)

            if self.reward_func is not None:
                reward_func = self.reward_func
                if self.greater_is_better is None:
                    raise ValueError('greater_is_better must be False or True.')
                else:
                    greater_is_better = self.greater_is_better
            else:
                if self.optimize_direction == 'minimize':
                    reward_func = mean_squared_error
                    greater_is_better = False
                else:
                    reward_func = accuracy_score
                    greater_is_better = True

            if self.cv is None:
                train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
                model.fit(train_x, train_y)
                pred_y = model.predict(test_x)
                score = reward_func(test_y, pred_y)
            else:
                reward_scorer = make_scorer(reward_func, greater_is_better=greater_is_better)
                score = cross_val_score(model, X, y, n_jobs=-1, cv=self.cv, scoring=reward_scorer)
                score = score.mean()

            if trial.should_prune():
                raise optuna.TrialPruned()

            return score

        return func

    def fit(self, X, y, **fit_params):
        study = optuna.create_study(
            direction=self.optimize_direction,
            sampler=self.sampler,
            pruner=self.pruner)
        study.optimize(self.objective(X, y), n_trials=self.n_trials)
        self.best_estimator_params = study.best_trial.params
        self.best_estimator_ = self.model(**self.best_estimator_params)
        try:
            self.best_estimator_.fit(X, y, **fit_params)
        except:
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)