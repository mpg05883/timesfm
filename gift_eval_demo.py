from gift_eval.data import Dataset
from dotenv import load_dotenv
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from src import timesfm

load_dotenv()

tfm = timesfm.timesfm_2p5_torch.TimesFM_2p5_200M_torch()
tfm.load_checkpoint()

dataset = Dataset(name="m4_hourly")

metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]

predictor = timesfm.TimesFmPredictor(
            tfm=tfm,
            prediction_length=dataset.prediction_length,
        )

res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=1024,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=get_seasonality(dataset.freq),
        )

mase = res["MASE[0.5]"][0]
crps = res["mean_weighted_sum_quantile_loss"][0]

print(f"MASE: {mase:.3f}, CRPS: {crps:.3f}")