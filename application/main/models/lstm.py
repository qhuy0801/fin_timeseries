from typing import Optional, List

from keras import Sequential
from keras.src.layers import LSTM, BatchNormalization, Dropout, Dense
from keras.src.optimizers import Adam, SGD
from keras.src.regularizers import L1


def trend_lstm(
    feature_count: int = 10,
    sequence_length: int = 60,
    feature_extracting_layer: int = 128,
    optimiser: str = "adam",
    learning_rate: float = 1e-4,
    lstm_layers: Optional[List[int]] = None,
    lstm_l1: Optional[float] = None,
    fc_layers: Optional[List[int]] = None,
    fc_l1: Optional[float] = None,
    fc_activation: str = "tanh",
    lstm_dropout: Optional[float] = None,
    fc_dropout: Optional[float] = None,
    **kwargs,
):
    sequence = Sequential()

    # Input layer with normalisation
    sequence.add(
        LSTM(
            units=feature_extracting_layer,
            return_sequences=True,
            input_shape=(sequence_length, feature_count),
            kernel_regularizer=L1(lstm_l1) if lstm_l1 is not None else None,
        )
    )
    sequence.add(BatchNormalization())
    if lstm_dropout:
        sequence.add(Dropout(lstm_dropout))

    # Middle lstm layers
    for index, lstm_layer in enumerate(lstm_layers):
        sequence.add(
            LSTM(
                units=lstm_layer,
                return_sequences=True if index < len(lstm_layers) - 1 else False,
                kernel_regularizer=L1(lstm_l1) if lstm_l1 is not None else None,
            )
        )
        if lstm_dropout:
            sequence.add(Dropout(lstm_dropout))

    # Middle fully connected layers
    if fc_layers:
        for fully_connected_layer in fc_layers:
            sequence.add(
                Dense(
                    units=fully_connected_layer,
                    kernel_regularizer=L1(fc_l1) if lstm_l1 is not None else None,
                    activation=fc_activation,
                )
            )
            if fc_dropout:
                sequence.add(Dropout(fc_dropout))

    # Final output layer
    sequence.add(
        Dense(
            units=1,
            activation="sigmoid",
            kernel_regularizer=L1(fc_l1) if lstm_l1 is not None else None,
        )
    )

    if optimiser == "sgd":
        optimiser = SGD(learning_rate=learning_rate)
    else:
        optimiser = Adam(learning_rate=learning_rate)

    sequence.compile(optimizer=optimiser, loss="binary_crossentropy", metrics=["binary_accuracy"])
    print(sequence.summary())
    return sequence


if __name__ == "__main__":
    model = trend_lstm(lstm_layers=[128, 64, 32], fc_layers=[16, 8], dropout=0.1)
    model.summary()
