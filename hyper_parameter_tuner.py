import os
from datetime import datetime

import wandb
from dotenv import load_dotenv

from application.main.services.training_services import train

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


if __name__ == "__main__":
    # Create the sweep
    # wandb.sweep(sweep_config, project="fin_timeseries")

    # Use Sweep to perform hyper-parameter tuning
    wandb.agent(
        sweep_id="tthsme5k",
        project="fin_timeseries",
        function=searching_train,
        count=20,
    )
