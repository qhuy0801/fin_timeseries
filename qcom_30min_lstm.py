import os
from datetime import datetime

import pandas as pd

import wandb
from dotenv import load_dotenv

from application.main.services.lstm_training import train, inferent

load_dotenv()
wandb.login(key=os.environ["WANDB_KEY"])


sweep_config = {
    "project": "fin_timeseries",
    "name": "QCOM_30min",
    "method": "bayes",
    "metric": {
        "name": "epoch/val_binary_accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "sequence_length": {"values": [10, 20, 40, 60]},
        "feature_extracting_layer": {"values": [64, 32, 16]},
        "optimiser": {"values": ["adam", "sgd"]},
        "learning_rate": {"distribution": "uniform", "min": 5e-6, "max": 1e-4},
        "correlated_symbols": {"values": [None, ["AAPL"], ["SPY"], ["SPY", "AAPL"]]},
        "lstm_layers": {
            "values": [
                [64, 64, 64],
                [32, 32, 32],
                [16, 16, 16],
                [64, 64],
                [32, 32],
                [16, 16],
                [64, 32, 16],
                [64, 32],
                [32, 16],
            ]
        },
        "lstm_l1": {"values": [None, 0.01, 0.02]},
        "fc_layers": {
            "values": [
                None,
                [64, 64],
                [32, 32],
                [16, 16],
                [64, 32],
                [32, 16],
                [64],
                [32],
                [16],
            ]
        },
        "fc_l1": {"values": [None, 0.01, 0.02]},
        "fc_activation": {"values": ["tanh", "sigmoid"]},
        "lstm_dropout": {"values": [None, 0.1, 0.2, 0.3, 0.4]},
        "fc_dropout": {"values": [None, 0.1, 0.2, 0.3, 0.4]},
    },
}


def searching_train(config=None):
    # Date formatting
    date_format = "%Y-%m-%d"

    # Run
    with wandb.init(config=config) as run:
        config = wandb.config
        train(
            # General configuration
            func="TIME_SERIES_INTRADAY",
            interval="30min",
            target_symbol="QCOM",
            period=(
                datetime.strptime("2014-04-01", date_format),
                datetime.now(),
            ),
            # Correlated symbols
            correlated_symbols=config.correlated_symbols,
            # Indicator
            indicator_settings={"MACD": {}, "BBANDS": {}, "RSI": {}},
            to_generator=False,
            # Training settings
            batch_size=200,
            upsampling=True,
            model_name="trend_lstm",
            sequence_length=config.sequence_length,
            validation_size=0.1,
            epochs=20,
            wandb_log=run,
            # Model configuration
            optimiser=config.optimiser,
            learning_rate=config.learning_rate,
            feature_extracting_layer=config.feature_extracting_layer,
            lstm_layers=config.lstm_layers,
            lstm_l1=config.lstm_l1,
            fc_layers=config.fc_layers,
            fc_l1=config.fc_l1,
            fc_activation=config.fc_activation,
            lstm_dropout=config.lstm_dropout,
            fc_dropout=config.fc_dropout,
        )


def tuning_train():
    # Date formatting
    date_format = "%Y-%m-%d"

    # Run
    with wandb.init(project="fin_timeseries") as run:
        return train(
            # General configuration
            func="TIME_SERIES_INTRADAY",
            interval="30min",
            target_symbol="QCOM",
            period=(
                datetime.strptime("2018-10-01", date_format),
                datetime.now(),
            ),
            # Correlated symbols
            correlated_symbols=["AAPL"],
            # Indicator
            indicator_settings={"MACD": {}, "BBANDS": {}, "RSI": {}},
            to_generator=False,
            # Training settings
            batch_size=200,
            upsampling=True,
            model_name="trend_lstm",
            sequence_length=60,
            validation_size=0.15,
            epochs=200,
            wandb_log=run,
            # Model configuration
            optimiser="adam",
            learning_rate=4e-5,
            feature_extracting_layer=64,
            lstm_layers=[64, 32],
            lstm_l1=None,
            fc_layers=[32, 32],
            fc_l1=0.02,
            fc_activation="sigmoid",
            lstm_dropout=0.1,
            fc_dropout=0.4,
            model_registry_name="qcom_30mins_lstm",
        )


def inferent_process(
    artifact_path: str = "artifact", artifact_name: str = "driven-glade-282_ckpt:v0"
):
    # Date formatting
    date_format = "%Y-%m-%d"

    # Download the model
    api = wandb.Api()
    artifact = api.artifact(
        f'{os.environ["WANDB_ENTITY"]}/fin_timeseries/{artifact_name}',
        type="model_n_scaler",
    )
    artifact.download(artifact_path)

    model_path = f"{artifact_path}/model.keras"
    scaler_path = f"{artifact_path}/scaler.save"

    timestamp, y, y_pred, org_df = inferent(
        func="TIME_SERIES_INTRADAY",
        interval="30min",
        target_symbol="QCOM",
        period=(
            datetime.strptime("2023-10-01", date_format),
            datetime.now(),
        ),
        # Correlated symbols
        correlated_symbols=["AAPL"],
        # Indicator
        indicator_settings={"MACD": {}, "BBANDS": {}, "RSI": {}},
        sequence_length=60,
        # Models
        model_name="trend_lstm",
        model_path=model_path,
        scaler_path=scaler_path,
    )

    result_df = pd.DataFrame({
        "timestamp": timestamp,
        "y": y,
        "y_pred": y_pred,
    })

    result_df.to_csv("qcom_30mins_lstm_result.csv", index=False)
    org_df.to_csv("qcom_30mins_lstm_org.csv", index=False)


if __name__ == "__main__":
    # Create the sweep
    # wandb.sweep(sweep_config, project="fin_timeseries")

    # Use Sweep to perform hyper-parameter tuning
    # wandb.agent(
    #     sweep_id="ckq012aq",
    #     project="fin_timeseries",
    #     function=searching_train,
    #     count=20,
    # )

    # Tuning train
    # _artifact_name = tuning_train()

    # Inferent
    inferent_process()
