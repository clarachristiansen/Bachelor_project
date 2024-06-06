import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import xgboost as xgb
import ast

from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# Import data
data = pd.read_csv("Data/Cleaned_data.csv")

#Unique combinations of strækning/station
combinations = data[['visualiseringskode', 'station']].drop_duplicates().values

# Retrieve best model parameters for ARIMA models for strækning/station
best_model_parameters = {}
model_params_sarima = pd.read_csv('Data/Best_model_parameters_SARIMA_strækning_station.csv')
list(model_params_sarima.items())
for i in range(model_params_sarima.shape[0]):
    tuple_result = ast.literal_eval(model_params_sarima['Key'][i])
    list_result = ast.literal_eval(model_params_sarima['Values'][i])
    best_model_parameters[tuple_result] = list_result

# Customized cross validation with rolling window and XGboost
def custom_cross_val_predict(estimator, y, X=None, cv=None, verbose=0, averaging="mean", return_raw_predictions=False, initial=2555):
    """Generate cross-validated estimates for each input data point
    
    Parameters
    ----------
    estimator : tuple
        A tuple containing two estimators. The first estimator should be the ARIMA model
        and the second one should be the Random Forest model.

    y : array-like or iterable, shape=(n_samples,)
        The time-series array.

    X : array-like, shape=[n_obs, n_vars], optional (default=None)
        An optional 2-d array of exogenous variables.

    cv : BaseTSCrossValidator or None, optional (default=None)
        An instance of cross-validation. If None, will use a RollingForecastCV.
        Note that for cross-validation predictions, the CV step cannot exceed
        the CV horizon, or there will be a gap between fold predictions.

    verbose : integer, optional
        The verbosity level.

    averaging : str or callable, one of ["median", "mean"] (default="mean")
        Unlike normal CV, time series CV might have different folds (windows)
        forecasting the same time step. After all forecast windows are made,
        we build a matrix of y x n_folds, populating each fold's forecasts like
        so::

            nan nan nan  # training samples
            nan nan nan
            nan nan nan
            nan nan nan
              1 nan nan  # test samples
              4   3 nan
              3 2.5 3.5
            nan   6   5
            nan nan   4

        We then average each time step's forecasts to end up with our final
        prediction results.

    return_raw_predictions : bool (default=False)
        If True, raw predictions are returned instead of averaged ones.
        This results in a y x h matrix. For example, if h=3, and step=1 then:

            nan nan nan # training samples
            nan nan nan
            nan nan nan
            nan nan nan
            1   4   2   # test samples
            2   5   7
            8   9   1
            nan nan nan
            nan nan nan

        First column contains all one-step-ahead-predictions, second column all
        two-step-ahead-predictions etc. Further metrics can then be calculated
        as desired.

    Returns
    -------
    predictions : array-like, shape=(n_samples,)
        The predicted values.

    """

    def indexable(*iterables):
        """Internal utility to handle input types"""
        results = []
        for iterable in iterables:
            if not hasattr(iterable, "__iter__"):
                raise ValueError("Input {!r} is not indexable".format(iterable))
            results.append(iterable)
        return results

    def check_cv(cv, initial = 2555):
        """Internal utility to check cv"""
        if cv is None:
            from pmdarima.model_selection import RollingForecastCV
            cv = RollingForecastCV(initial=initial, step=1, h=1)
        return cv

    def check_endog(y, copy=True, preserve_series=False):
        """Internal utility to check endogenous variable"""
        from pmdarima.utils import check_endog
        return check_endog(y, copy=copy, preserve_series=preserve_series)

    def _check_averaging(averaging):
        """Internal utility to check averaging"""
        if averaging == "mean":
            return np.nanmean
        elif averaging == "median":
            return np.nanmedian
        elif callable(averaging):
            return averaging
        else:
            raise ValueError("Unknown averaging method: {}".format(averaging))

    def _fit_and_predict(fold, estimator_tuple, y, X, train, test, verbose=0):
        """Internal utility to fit and predict"""
        arima_model = estimator_tuple
        # Fit ARIMA model
        arima_model.fit(y[train]) # X=X.iloc[train, :]
        # Predict with ARIMA model
        train_predictions = arima_model.predict_in_sample()
        arima_pred = arima_model.predict(n_periods=len(test))

        # Calculate residuals for RF input
        arima_residuals_train = train_predictions - y[train]

        model = xgb.XGBRegressor(objective = 'reg:absoluteerror', booster = 'gbtree', max_depth=5, steps =20, learning_rate=0.1) # 'reg:squarederror'
        # Train the model
        #model = model.fit(D_train, steps, watchlist)
        model = model.fit(X.iloc[train,:], arima_residuals_train)
        # Predict the labels of the test set
        #preds = model.predict(D_test)
        preds = model.predict(X.iloc[test,:])
        # Overall prediction residuals = pred - true <=> true = pred - residuals
        overall_pred = np.array(max(min(1, arima_pred[0] - preds), 0)) # make sure it is in [0;1]

        return overall_pred, test, np.array(arima_pred) #arima_residuals_test

    y, X = indexable(y, X)
    y = check_endog(y, copy=False, preserve_series=True)
    cv = check_cv(cv, initial)
    avgfunc = _check_averaging(averaging)

    if cv.step > cv.horizon:
        raise ValueError("CV step cannot be > CV horizon, or there will be a gap in predictions between folds")

    prediction_blocks = [
        _fit_and_predict(fold,
                         estimator,
                         y,
                         X,
                         train=train,
                         test=test,
                         verbose=verbose,)  # TODO: fit params?
        for fold, (train, test) in enumerate(cv.split(y, X))]

    pred_matrix = np.ones((y.shape[0], len(prediction_blocks))) * np.nan
    arima_pred = []
    for i, (pred_block, test_indices, arima_block) in enumerate(prediction_blocks):
        pred_matrix[test_indices, i] = pred_block
        arima_pred.append(arima_block)

    if return_raw_predictions:
        predictions = np.ones((y.shape[0], cv.horizon)) * np.nan
        for pred_block, test_indices in prediction_blocks:
            predictions[test_indices[0]] = pred_block
        return predictions

    test_mask = ~(np.isnan(pred_matrix).all(axis=1))
    predictions = pred_matrix[test_mask]



    # Calculate CV score
    cv_scores = []
    cv_scores_arima = []
    for fold, (train, test) in enumerate(cv.split(y, X)):
        fold_predictions = pred_matrix[test, fold]
        fold_score = float(abs(y[test] - fold_predictions))
        fold_arima_score = float(abs(y[test] - arima_pred[fold]))
        cv_scores.append(fold_score)
        cv_scores_arima.append(fold_arima_score)

    # Compute overall CV score
    score = np.mean(cv_scores)
    arima_score = np.mean(cv_scores_arima)

    return avgfunc(predictions, axis=1), score,  arima_score, cv_scores, cv_scores_arima

# Get CV, errors and prediction for strækning/station
results_strækning_station = {}
combinations = data[['visualiseringskode', 'station']].drop_duplicates().values
MSEs = []
MSEs_arima = []
random_state = 42
initial_start = 2500
preds = {}
cv_scores = {}

for strækning, station in combinations[:10]:
    y = data[(data['visualiseringskode'] == strækning) & (data['station'] == station)]['togpunktlighed']
    X = data[(data['visualiseringskode'] == strækning) & (data['station'] == station)].iloc[:,1:]
    arima_model = pm.arima.ARIMA(order = best_model_parameters[strækning, station][0], seasonal_order=best_model_parameters[strækning, station][1])

    pred, mse, mse_arima, cv_score, cv_score_arima = custom_cross_val_predict((arima_model), y, X, cv=None, verbose=1, averaging="mean", return_raw_predictions=False, initial=initial_start)
    # output = prediction per fold. 
    MSEs.append(mse)
    MSEs_arima.append(mse_arima)
    preds[(strækning, station)] = pred
    cv_scores[(strækning, station)] = (cv_score, cv_score_arima)
    # initial = 2457 - 100 fold, 10 combinations, n_estimators=100 -> 61 min

# Plot and save error
x = np.arange(len(MSEs))
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in [['Arima', MSEs_arima], ['Full', MSEs]]:
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1
ax.legend(loc='upper left', ncols=2)
ax.set_xticks(x + width, combinations[:10])
ax.set_xlabel('[Strækning, Station]')
ax.set_ylabel('MAE')
ax.set_title('MAE per strækning per station')
ax.grid()
fig.savefig('Results/MAE_strækningstation_XGboost')

# Plotting functions for each strækning
def hist_strækning(strækning, cv_scores):
    cv_scores_full = [cv[0] for key, cv in cv_scores.items() if key[0] == strækning]
    cv_scores_arima = [cv[1] for key, cv in cv_scores.items() if key[0] == strækning]
    
    cv_scores_full = np.mean(cv_scores_full, axis=0)
    cv_scores_arima = np.mean(cv_scores_arima, axis=0)

    plt.hist(cv_scores_arima, label='ARIMA')
    plt.hist(cv_scores_full,label='Full')
    plt.title(f'Strækning {strækning}')
    plt.xlabel('MAE')
    plt.ylabel('Occurences')
    plt.legend(loc='upper right')
    plt.grid()
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) # only integer ticks i y-axis
    plt.savefig(f'Results/Strækning{strækning}_histogram_MAE_XGboost.png', bbox_inches='tight')
    plt.show()

def linechart_strækning(strækning, cv_scores):
    cv_scores_full = [cv[0] for key, cv in cv_scores.items() if key[0] == strækning]
    cv_scores_arima = [cv[1] for key, cv in cv_scores.items() if key[0] == strækning]
    
    cv_scores_full = np.mean(cv_scores_full, axis=0)
    cv_scores_arima = np.mean(cv_scores_arima, axis=0)

    plt.plot(cv_scores_arima, label='ARIMA')
    plt.plot(cv_scores_full,label='Full')
    plt.title(f'Strækning {strækning}')
    plt.xlabel('Data points')
    plt.ylabel('MAE')
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(f'Results/Strækning{strækning}_linechart_MAE_XGboost.png', bbox_inches='tight')
    plt.show()

# For Kystbanen
for strækning in data['visualiseringskode'].unique():
    hist_strækning(strækning, cv_scores)
    linechart_strækning(strækning, cv_scores)