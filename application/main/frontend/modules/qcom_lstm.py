import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import bctools as bc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from sklearn.metrics import accuracy_score

from application.main.utils.frontend.markdown import mermaid

data_org = pd.read_csv("application/main/frontend/resources/qcom_30mins_lstm_org.csv")
data_result = pd.read_csv(
    "application/main/frontend/resources/qcom_30mins_lstm_result.csv"
)

st.header(
    body="30-Minute QCOM: Training and Backtesting LSTM Model for Trend and Reversal Prediction Strategy",
    anchor="qcom_30min",
    help=None,
    divider="grey",
)

with st.expander(label="Overview"):
    st.subheader(body="Overview")

    st.markdown(
        body="""
    **Develop and Test Model:** The experiment involves developing a model that can detect trends and reversions in stock 
    prices. It aims to predict whether the closing price at the next timestep \( t+1 \) will be higher or lower than the 
    current closing price \( t \).
    
    **Trading Strategy Implementation:** Once the model is ready and optimized, it will be used to make trading decisions:
    - **Long Order:** Place a long order if the model predicts that the next closing price will be higher than the current 
    closing price.
    - **Short Order:** Close the long order and place a short order if the model predicts that the next closing price will 
    be lower.

    """
    )

    mermaid(
        """
            flowchart LR
        A(Input Data) -->|Pass to| B[LSTM Layer 1]
        B -->|Output to| C[LSTM Layer 2]
        C --> D{...}
        D --> E[LSTM Layer N]
        E -->|Flatten/Reshape| F[Dense Layer 1]
        F --> G[Dense Layer 2]
        G --> H{...}
        H --> I[Dense Layer M]
        I --> J(Output Layer)
        J -->|Class 1| K(Upward-trend)
        J -->|Class 0| L(Downward-trend)

            """
    )

    st.markdown(
        body="""
    **Model type**: Stacked LSTM paired with a fully-connected network is quite suitable for the task. This setup uses previous 
    timestamp data through the LSTM layers to predict future outcomes, helping the model recognize and learn from past 
    patterns. 
    
    Stacking the LSTM layers improves the model's ability to spot even minor differences in data patterns, 
    enhancing its predictive precision. 
    
    The fully-connected layer is crucial as it converts the LSTM's complex outputs into 
    clear, actionable decisions. This design helps the model effectively understand and utilize the time-related patterns in 
    the data.
        """
    )


with st.expander(label="Model building procedure"):
    st.subheader(body="Model building procedure", divider="grey")

    st.markdown(
        """
**Building trend predictive model in a machine learning approach**

**1. Model Construction**
   - Start by selecting the model architecture, type, input data, and target.
   - Generate necessary data features and layout a blueprint for the predictive model.
   - The architecture of the model has been previously defined and described.

**2. Parameter Tuning**
   - Even with a chosen model blueprint, numerous internal parameters require tuning to tailor the model for specific tasks.
   - A Bayesian search was conducted in a parallel cloud environment to rapidly identify optimal settings.
   - For more detailed information, refer to the **Model Selection Report** at the end of this page or in 
   [this link](https://api.wandb.ai/links/qhuy0168/pxwhp4d5).

**3. Fine Tuning and Training**
   - After acquiring the optimal parameter set, the model undergoes additional fine-tuning and extended training to achieve maximum convergence.
   - The final model, along with the relevant data scaler, is available for download at artifact section of 
   [this link](https://wandb.ai/qhuy0168/fin_timeseries/runs/xpl0zpzp?nw=nwuserqhuy0168).

**4. Signal Tuning for Binary Classification**
   - As the task involves binary classification, further tuning can be accomplished by adjusting the decision threshold to refine decision-making.
   
**5. Backtesting and Production**
   - Adjusted thresholds are tested during the backtesting phase with recent data to validate the model’s effectiveness.
   - Following successful backtesting, the model is ready for production deployment.
   
    """
    )

    mermaid(
        """
    flowchart LR
    A(Model Construction) -->|Proceed to| B(Model Selection)
    B -->|Proceed to| C(Model Training)
    C -->|Proceed to| D(Analyze Result and Choose Threshold)
    D -->|Proceed to| E(Backtesting)

    %% Feedback Loops
    C -->|Feedback to| B
    E -->|Feedback to| D

    %% Additional description for feedback loops
    style B stroke:#f66,stroke-width:2px
    style C stroke:#f66,stroke-width:2px
    style D stroke:#f66,stroke-width:2px
    style E stroke:#f66,stroke-width:2px

    """
    )

    st.markdown(
        body="""
    **Training and backtesting timeframe**
    According to random walk theory, consistently outperforming the market in the long run is viewed as nearly impossible. 
    Initial examination of long-term data supports this notion, revealing that while certain phenomena may recur in the 
    short term, they do not predictably extend into the long term. Additionally, preliminary analysis from a data science 
    perspective shows that the price data is **not stationary**, leading to a decline in predictive performance over time. 
    This also means that certain data manipulation techniques may become invalid. 
    
    Consequently, for strategies that span a medium timeframe, it is critical to select a training and validation period 
    that isn't too lengthy to avoid over-fitting. 
    
    Furthermore, to keep up with evolving market conditions, the model should be frequently retrained with fresh data to 
    ensure ongoing relevance and accuracy.
    
    Therefore, the duration for this experiment is chosen and listed below:
    """
    )

    st.markdown(
        body="""
    - **Training Data:** Train the model on stock price data from a 5-year period, spanning from October 2018 to October 2023.
    - **Backtesting:** Test the model's performance through backtesting from October 2023 to April 2024 to assess its 
    effectiveness in applying learned patterns to new data.
    """
    )


with st.container(border=True):

    st.subheader(body="Trained model performance", divider="grey")

    st.markdown(body="""
    The model focuses on predicting the closing prices of targeted assets for each timestep \( t \) and \( t+1 \), within a 30-minute interval. The probability signal, which indicates whether the price is likely to go up or down, is visualized in the third plot below:

- **Signal Interpretation:** If the signal value at timestep \( t \) is above 0.5 (or 50%), it suggests an upward trend in the closing value at \( t+1 \). This threshold is used to train the model to reliably predict positive movements in asset prices.

This approach allows the probabilistic outputs of the model to be effectively used as actionable trading signals.
    """)

    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.5, 0.2, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Closing price in backtesting timeframe",
            "Volume",
            "Trend probability (signal) produced by the model",
        ],
    )

    fig.add_trace(
        go.Scatter(x=data_org.timestamp, y=data_org.close),
        col=1,
        row=1,
    )
    fig.add_trace(
        go.Bar(x=data_org.timestamp, y=data_org.volume),
        col=1,
        row=2,
    )
    fig.add_trace(
        go.Scatter(x=data_result.timestamp, y=data_result.y_pred),
        col=1,
        row=3,
    )

    fig.update_layout(height=900, showlegend=False)
    fig.update_annotations(xshift=20)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(body="""
However, as mentioned previously, the **random walk nature of the market** means that the data is not stationary. Consequently, the output signal from the model is not expected to clearly deliver **two distinct distributions for the classes (up or down)**.

As visualized above, the probability of an upward or downward trend tends to **fluctuate between 0.35 and 0.55**, rather than spanning the two ends of the target value spectrum (0 and 1). This behavior indicates that the model's probability outputs are clustered around the center of the range, that the signal reflecting certaintiness instead of decisive predictions.

Therefore, it is crucial to analyze the results and fine-tune the class discrimination threshold before deploying the model into production. This adjustment will help in better aligning the model's sensitivity to actual market movements and enhance its practical utility.

    """)

with st.container(border=True):
    st.subheader(body="Threshold selection and strategy", divider="grey")
    with st.container():
        col1, col2 = st.columns(spec=[0.45, 0.55])
        with col1:
            roc_plot, _ = bc.curve_ROC_plot(
                true_y=data_result.y, predicted_proba=data_result.y_pred
            )
            roc_plot.update_layout(autosize=True)
            st.plotly_chart(roc_plot, use_container_width=True)
        with col2:
            st.markdown(body="""
- The ROC curve indicates that classification is not perfect (with an AUC of 1, which would signify flawless accuracy 
every time). However, the curve confirms that the model **performs significantly better than random guessing**.
- Similar to the signal probabilities visualized above, the ROC curve was calculated using a **classification threshold 
ranging between 0.3411 and 0.5389**.
- Based on the results observed from the ROC plot, the best classification thresholds for determining upward or downward 
trends are estimated to be **around 0.49 to 0.52**.
- The rationale for selecting this optimal range is that this particular portion of the curve is the furthest from the 
line representing a random guess, in terms of squared error.
            """)

    with st.container():
        col1, col2 = st.columns(spec=[0.45, 0.55])
        with col1:
            st.markdown(body="""
- The "Threshold vs. Accuracy" plot reveals that the model's accuracy peaks at thresholds of 0.4914 and 0.5097. 
- This phenomenon occurs because the probability distribution tends to cluster around the midpoint (0.5). Thus, slight 
adjustments around this central value significantly influence the model's performance.
- Deciding whether to increase or decrease the threshold depends on the strategic priority of the model: **whether 
increasing the number of positive predictions (indicating an upward trend) is more important than overall accuracy**.
- Consequently, final tuning of the model — **choosing the optimal threshold that yields the highest accuracy from the 
listed values — should be performed during the final backtesting phase**. This step ensures that the model is finely 
adjusted to meet specific strategy requirements before deployment in a production environment.           
            """)
        with col2:
            thresholds = np.linspace(0.48, 0.52, 1000)
            accuracies = [
                accuracy_score(data_result.y, data_result.y_pred >= t)
                for t in thresholds
            ]
            thresholds = pd.DataFrame({"threshold": thresholds, "accuracy": accuracies})
            fig = px.line(
                thresholds,
                x="threshold",
                y="accuracy",
                labels={"Threshold": "Threshold", "Accuracy": "Accuracy"},
                title="Accuracy vs. Threshold",
            )
            fig.update_layout(autosize=True)
            st.plotly_chart(fig, use_container_width=True)

with st.container(border=True):
    st.subheader(body="Performance and benchmarking", divider="grey")
    col1, col2 = st.columns(spec=[0.45, 0.55])
    with col1:
        thresholds = [0.4914, 0.5, 0.5097]
        for threshold in thresholds:
            data_result[f"action_{threshold}"] = np.where(
                data_result["y_pred"] >= threshold, 1, -1
            )
        data_org["return"] = data_org["close"].pct_change().shift(-1)
        data_merged = pd.merge(data_org, data_result, on="timestamp", how="inner")
        data_merged["timestamp"] = pd.to_datetime(data_merged["timestamp"])
        data_merged["buy_hold_return"] = (1 + data_merged["return"]).cumprod()
        for threshold in thresholds:
            data_merged[f"algo_return_{threshold}"] = (
                1 + (data_merged[f"action_{threshold}"] * data_merged["return"])
            ).cumprod()

        fig = go.Figure()

        # Loop through each threshold to add a line to the plot
        for threshold in thresholds:
            fig.add_trace(
                go.Scatter(
                    x=data_merged["timestamp"],
                    y=data_merged[f"algo_return_{threshold}"],
                    mode="lines",
                    name=f"Algo Return {threshold}",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=data_merged["timestamp"],
                y=data_merged[f"buy_hold_return"],
                mode="lines",
                name=f"Buy and hold returns",
            )
        )

        # Update the layout to add titles and axis labels
        fig.update_layout(
            title="Algorithm Return by Threshold",
            xaxis_title="Date",
            yaxis_title="Algorithm Return",
            legend_title="Thresholds",
            height=500,
            legend=dict(
                yanchor="top",
                y=1,
                xanchor="left",
                x=0,
            ),
        )

        # Display the figure
        st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(body="""
- In the backtesting environment, the final returns from the algorithmic decisions—based solely on the signals generated 
by the LSTM model—were analyzed to identify the optimal threshold value.
- The returns of the algorithm were **compared against a benchmark 'buy and hold' strategy**, which serves as a simple 
yet effective measure of market performance over time.
- The results indicated that a **lower threshold (0.4914)**, which makes the model more sensitive to detecting downward 
trends, led to **superior outcomes** compared to other thresholds that prioritize upward trend detection, and the 
standard threshold of 0.5.
- This phenomenon can be explained by the overall upward trend observed during the testing period. In such a market, a 
strategy that **effectively utilizes quality signals on downward trends** allows the model to generate returns that 
surpass those of a standard buy and hold strategy. This is due to the model's enhanced ability to capitalize on brief but 
significant market downturns for potential gains.

  &rarr; As a result, for a strategy that relies solely on the LSTM-generated signals, the model output with a 
  **threshold of 0.4914** should be adopted as the key decision-making parameter. This threshold setting has been 
  identified as **optimal for maximizing returns** based on the model's predictive accuracy during backtesting.
            """)

with st.expander("Model selection"):
    components.iframe(
        "https://wandb.ai/qhuy0168/fin_timeseries/reports/LSTM-Model-selection-of-QCOM-on-30-mins-interval--Vmlldzo3NzExMDI3",
        height=1024,
        scrolling=True,
    )

with st.expander("Raw data viewer"):
    col1, col2 = st.columns(spec=[0.5, 0.5])
    with col1:
        st.dataframe(data=data_org.drop(columns=["id"]), hide_index=True, use_container_width=True)
    with col2:
        st.dataframe(data=data_result, hide_index=True, use_container_width=True)

with st.expander("Technology used"):
    st.markdown(
        body="""
- **Numpy:** Used for array and numerical calculations.
- **Pandas:** Employed for data preprocessing, cleaning, and manipulation.
- **SQL:** 
  - **SQLite:** Used for database storage.
  - **SQLAlchemy:** Used for database queries and filtering.
- **Scikit-learn:** Utilized for data interpretation, calculations, metrics, and analysis.
- **Keras:** Applied for building the neural network model.
- **TA Library:** Used to calculate technical analysis indicators for feature engineering.
- **Alpha Vantage:** Sources real historical data (the model is tested on actual historical data, not sandbox or 
simulated data).
    """
    )
