import os
from datetime import datetime

import wandb
from dotenv import load_dotenv

from application.main.services.training_services import train

load_dotenv()
wandb.login(key=os.environ["WANDB_KEY"])


sweep_config = {
    "project": "fin_timeseries",
    "name": "QCOM_single_BBANDS_MACD",
    "method": "bayes",
    "metric": {
        "name": "accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "sequence_length": {"values": [20, 40, 60]},
        "feature_extracting_layer": {"values": [32, 64, 128]},
        "lstm_layers": {"values": [[128, 64, 32], [64, 32, 16], [64, 32], [32, 16]]},
        "lstm_l1": {"values": [None, 0.01, 0.02]},
        "fc_layers": {"values": [[32, 16, 8], [32, 16], [16, 8]]},
        "fc_l1": {"values": [None, 0.01, 0.02]},
        "fc_activation": {"values": ["tanh", "sigmoid"]},
        "dropout": {"values": [None, 0.2, 0.4]},
    },
}


def searching_train():
    # Init the run
    run = wandb.init(project="fin_timeseries")

    # Get the configuration for instance
    config = wandb.config

    # Date formatting
    date_format = "%Y-%m-%d"

    train(
        # General configuration
        func="TIME_SERIES_INTRADAY",
        interval="15min",
        target_symbol="QCOM",
        period=(
            datetime.strptime("2019-01-01", date_format),
            datetime.strptime("2024-01-01", date_format)
        ),
        correlated_symbols=["GOOGL"],
        indicator_funcs=None,
        batch_size=200,
        to_generator=False,
        model_name="trend_lstm",
        sequence_length=config.sequence_length,
        validation_size=0.2,
        epochs=10,
        wandb_log=run,
        # Indicator configuration
        time_period="20",
        series_type="close",
        nbdevup="1.5",
        nbdevdn="1.5",
        fastperiod="12",
        slowperiod="26",
        signalperiod="9",
        # Model configuration
        feature_extracting_layer=config.feature_extracting_layer,
        lstm_layers=config.lstm_layers,
        lstm_l1=config.lstm_l1,
        fc_layers=config.fc_layers,
        fc_l1=config.fc_l1,
        fc_activation=config.fc_activation,
        dropout=config.dropout,
    )

    # train(
    #     # General configuration
    #     func="TIME_SERIES_INTRADAY",
    #     interval="15min",
    #     target_symbol="QCOM",
    #     period=None,
    #     correlated_symbols=None,
    #     indicator_funcs=None,
    #     batch_size=200,
    #     to_generator=False,
    #     model_name="trend_lstm",
    #     sequence_length=60,
    #     validation_size=0.2,
    #     epochs=10,
    #     wandb_log=run,
    #     # Indicator configuration
    #     time_period="20",
    #     series_type="close",
    #     nbdevup="1.5",
    #     nbdevdn="1.5",
    #     fastperiod="12",
    #     slowperiod="26",
    #     signalperiod="9",
    #     # Model configuration
    #     feature_extracting_layer=32,
    #     lstm_layers=[32, 32, 32],
    #     lstm_l1=None,
    #     fc_layers=None,
    #     fc_l1=None,
    #     fc_activation="tanh",
    #     dropout=0.35,
    # )


if __name__ == "__main__":
    # Create the sweep
    # wandb.sweep(sweep_config, project="fin_timeseries")

    # Use Sweep to perform hyper-parameter tuning
    wandb.agent(
        sweep_id="dqotfmuu", project="fin_timeseries", function=searching_train, count=20
    )
