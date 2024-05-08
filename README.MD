# 30-Minute QCOM LSTM Model for Trend and Reversal Prediction

## Project Overview
This project focuses on developing an LSTM model to predict trends and reversals in the QCOM stock price on a 30-minute interval. The model outputs a signal indicating the probability that the next closing price will be lower or higher than the current closing price.

## Environment Setup
- **Python Version**: The project is built using Python 3.10.
- **Installation**:
  - Ensure that Python 3.10 is installed on your system.
  - Install all required Python packages using:
    ```bash
    pip install -r requirements.txt
    ```

## Main Application
- **File**: `qcom_30min_lstm.py`
- **Description**: This file contains the complete logic for model training, backtesting, and making predictions. Additionally, it includes functionalities for parameter tuning and running multivariate tests (sweeps).

## Usage Instructions
1. Open the `qcom_30min_lstm.py` file.
2. Choose the specific section you want to run:
   - **Sweep Configuration**: For setting up configurations for multivariate testing.
   - **Run Sweep**: To execute multivariate tests.
   - **Fine Tuning**: For model training with fine-tuning parameters.
   - **Inferencing**: To use the trained model for making predictions.
3. Comment out the sections of the code that you do not want to execute.
4. Run the file from your terminal or Python environment.

## Contributing
Feel free to fork this repository and submit pull requests to contribute to the development of the model. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

