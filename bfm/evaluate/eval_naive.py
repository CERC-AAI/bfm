import sys

import wandb
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

sys.path.append("./data/")
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from autogluon.timeseries.metrics import MAE, MAPE, MASE, RMSE, SMAPE, TimeSeriesScorer
from gluonts.dataset.arrow import ArrowFile
from gluonts.dataset.common import DatasetCollection, MetaData
from gluonts.time_feature import get_seasonality
from read_dataset import get_transformation


class MeanSquaredError(TimeSeriesScorer):
    greater_is_better_internal = False
    optimum = 0.0

    def compute_metric(self, data_future, predictions, target, **kwargs):
        return sklearn.metrics.mean_squared_error(
            y_true=data_future[target], y_pred=predictions["mean"]
        )


class WQL(TimeSeriesScorer):
    needs_quantile = True

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, q_pred, quantile_levels = self._get_quantile_forecast_score_inputs(
            data_future, predictions, target
        )
        values_true = y_true.values[:, None]  # shape [N, 1]
        values_pred = q_pred.values  # shape [N, len(quantile_levels)]

        return 2 * np.mean(
            np.nansum(
                np.abs(
                    (values_true - values_pred)
                    * ((values_true <= values_pred) - quantile_levels)
                ),
                axis=0,
            )
            / np.nansum(np.abs(values_true))
        )


class MeanQuantileLoss(TimeSeriesScorer):
    needs_quantile = True
    greater_is_better_internal = False
    optimum = 0.0

    def compute_metric(self, data_future, predictions, target, **kwargs):
        quantile_columns = [col for col in predictions if col != "mean"]
        total_quantile_loss = 0.0
        for q in quantile_columns:
            total_quantile_loss += sklearn.metrics.mean_pinball_loss(
                y_true=data_future[target], y_pred=predictions[q], alpha=float(q)
            )
        return total_quantile_loss / len(quantile_columns)


# Load dataset
test_path = "./data/moabb/moabb/BI2012_dl/train/"
train_ds, freq = get_transformation(test_path)

test_data = TimeSeriesDataFrame.from_iterable_dataset(train_ds)
prediction_length = 64
train_data_ag_splitted, test_data_ag_splitted = test_data.train_test_split(
    prediction_length=prediction_length
)
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length, quantile_levels=None, cache_predictions=False
).fit(train_data_ag_splitted, hyperparameters={"Naive": {}})
predictions = predictor.predict(train_data_ag_splitted)

fig = predictor.plot(
    data=test_data_ag_splitted,
    predictions=predictions,
    quantile_levels=None,
    max_history_length=200,
    max_num_item_ids=4,
    # item_ids=[70, 23, 4, 79],
)

mse = MeanSquaredError()
mse_score = mse(
    data=test_data_ag_splitted,
    predictions=predictions,
    prediction_length=predictor.prediction_length,
    target=predictor.target,
)
wql = WQL()
mae, mape, mase, rmse, smape = MAE(), MAPE(), MASE(), RMSE(), SMAPE()
wql_score = wql(
    data=test_data_ag_splitted,
    predictions=predictions,
    prediction_length=predictor.prediction_length,
    target=predictor.target,
)
mae_score = mae(
    data=test_data_ag_splitted,
    predictions=predictions,
    prediction_length=predictor.prediction_length,
    target=predictor.target,
)
mape_score = mape(
    data=test_data_ag_splitted,
    predictions=predictions,
    prediction_length=predictor.prediction_length,
    target=predictor.target,
)
mase_score = mase(
    data=test_data_ag_splitted,
    predictions=predictions,
    prediction_length=predictor.prediction_length,
    target=predictor.target,
)
rmse_score = rmse(
    data=test_data_ag_splitted,
    predictions=predictions,
    prediction_length=predictor.prediction_length,
    target=predictor.target,
)
smape_score = smape(
    data=test_data_ag_splitted,
    predictions=predictions,
    prediction_length=predictor.prediction_length,
    target=predictor.target,
)
print(f"{mse.name_with_sign} = {mse_score}")
log_dict = {
    mse.name_with_sign: mse_score,
    "WQl": wql_score,
    "MAE": mae_score,
    "MAPE": mape_score,
    "MASE": mase_score,
    "rmse": rmse_score,
    "smape": smape_score,
}
wandb.log(log_dict)
