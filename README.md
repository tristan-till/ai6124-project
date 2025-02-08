# AI6124 - Neuro Evolution and Fuzzy Intelligence Project (Group 17)

### Submitted By Group 17:
#### Goyal Hitesh Mat ID: G2405867G
#### Till Tristan Mat ID: G2403792H

### Stocks (01.01.2020 - 01.11.2024)

- **Primary Dataset: Exxon Mobil (XOM)**

  The daily closing price offered on the Yahoo Finance API is considered for this analysis. Yahoo Finance aggregates multiple data sources for a reliable estimate, despite the primary exchange of the Exxon Mobil stock being traded on the NYSE (New York Stock Exchange).

  ExxonMobil is a multinational energy company engaged in the exploration, production, and distribution of oil, natural gas, and petrochemical products. It is one of the largest publicly traded companies in the world, with operations spanning the entire energy sector, including refining, chemical manufacturing, and renewable energy initiatives.

- **Testing Dataset: Chevron (CVX)**

  Similarly, for the testing dataset, the Yahoo Finance API will be used, despite the stock being primarily traded on the NYSE. As described above, this API combines various sources to provide more profound insight into the stock's value.

  Chevron is a global energy corporation involved in the exploration, production, and refining of oil and natural gas, as well as the manufacture of petrochemicals. As one of the world’s largest oil companies, it operates across the energy spectrum, including investments in renewable energy and advanced technologies to reduce environmental impact.

- **Supplementary Data: S&P 500 (^SPX)**

  We assume that stocks within the energy sector are commodity-driven and hence reliant on external factors beyond a single company's control. In our proposal, we hypothesized the best possible prediction may require additional sources on global oil prices, energy consumption/demand, weather data, governmental policies, or geopolitical events.

  To limit the scope of input data to a manageable size, one such indicators, the S&P 500, is used as supplementary data to train our model for stock price prediction. It should act as a guidance for the overall market trend, allowing us to contextualize our target stock's movements. By incorporating the S&P 500, we can enhance our model's ability to identify relevant features that influence stock prices, ultimately improving the accuracy of our predictions.


### Features and Indicatores Used:

Our backbone and head models (referenced down below) are trained on different features. All features originate from the baseline data extracted using the Yahoo Finance API.

### Backbone Data

- **Closing Price and Normalised Closing Price Difference:**
  The actual closing price and the normalised closing price difference are used as features within the model. One tells the the model about the price itself, while the other represents how much change has taken place (acting as a differential of price). Their utility are widely discussed, as some financial research suggests stock prices move randomly on a day-to-day basis - for sake of completeness we decided to include these measures into our submission.
- **Volume and Normalised Volume Difference:**
  Volume information indicates the amount of momentum a stock has when moving in a particular direction. This information can help the model analyse the significance of the stock's movement in a particular direction. The Volume difference acts as a differential of volume. Similarly, it is hypothesized these values change randomly on a per-day basis but were included with equal reasoning to closing data.
- **Relative Strength Index (RSI):**
  According to [investopedia](https://www.investopedia.com/terms/r/rsi.asp), the RSI is a momentum oscillator that is widely used in technical analysis of stocks and commodities to identify changes in momentum and price direction.

  Typically, RSI operates on two thresholds, suggesting a stock may be *overbought* when the RSI-value surpasses a predefined upper bound (UB=70), resulting in a recommendation to sell. Alternatively, values trending below a predefined lower bound (LB=30) may appear to be *oversold*, hence being recommended to buy. To be less prone to variance, RSI is evaluated across different time intervals (short=5d, month=20d, quarter=60d), producing more robust buy/sell signals.

- **Rate of Change (ROC):**
  According to [investopedia](https://www.investopedia.com/terms/r/rateofchange.asp), the rate of change (ROC) is the speed at which a variable changes over a specific period of time. ROC is often used when speaking about momentum, and it can generally be expressed as a ratio between a change in one variable relative to a corresponding change in another; graphically, the rate of change is represented by the slope of a line. ROC will be evaulated across a 12-day time interval, to provide more robust price difference information of bi-weekly trends.

- **Moving Average Convergence/Divergence (MACD):**
  According to [investopedia](https://www.investopedia.com/terms/m/macd.asp), Moving Average Convergence/Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two **Exponential Moving Averages (EMAs)** of a security's price.

  EMA is a more robust moving average, as compared to Slow (SMA) or Fast Moving Average. MACD extracts useful information from different EMAs, similar to what we wanted to achieve with RSI over various intervals.

  MACD measures the relationship between two EMAs to indicate momentum and potential trade reversals, while the RSI seeks out overbought and oversold conditions by evaluating recent price action. These indicators are often used together to give analysts a more complete technical picture.

  Similarly, **Volume MACD (VMACD)**, encompasses the same functionality of MACD for volumetric trends. This feature is used as another indicator to understand high volume breakouts.

### Head Data

- **RSI-Signal**: Calculate the average of RSI in three time windows, short=7d, medium=30d, and long=90d
- **EMA-Signal**: EMA difference between two time windows short=12d - long=26d
- **MACD-Signal**: Logistic transformation to the difference between the MACD line and MACD-signal line
- **Portfolio-Liquidity**: The ratio of current cash divided by total portfolio value.

### Training Data
For training, data from **01.01.2020 to 01.11.2024** was considered.

**Backbone training data** used values up to 60 days in the past to predict the next days stock price. In total, the target stock and one supplementary stock were used, extracting 13 features per stock. Therefore, each input datapoint was of dimension 60x26.

**Head training data** does not use past values and only generates buy/sell signals on the current feature distribution. These include the aforementioned features for the target and supplementary stock, as well as the model prediction, resulting in a 1x8 feature vector as input.

## Hybrid AI Method: GRU-LSTM-Attention with MO-GenFIS

For the Hybrid AI model we decided to employ a GRU-LSTM-Attention backbone with Multi-Objective GenFIS Head. Conceptually, the **GRU-LSTM-Attention backbone** will process large quantities of context data to supply accurate predictions for a stock's price change within the next episode. This prediction is fed forward to the **Multi-Objetive GenFIS Head**, upon other information, to automatically generate buy/sell signals. Importantly, we hypothesized that with this setup, only the model backbone requires **finetuning and transfer learning**, as the learned fuzzy ruleset should transcend the logic of a single stock.

Visualization of the propsed hybrid architecture:

<img src="https://github.com/tristan-till/ai6124-project/blob/main/images/fuzzy-model.png?raw=true" alt="GRU-LSTM-Attention GenFIS Model Architecture" width="60%" height="auto" style="margin: 10px; align-center justify-center">

### GRU-LSTM-Attention

The backbone architecture was inspired by that in [14] from the literature review, which employed a similar structure as a backbone for a Reinforcement Learning (RL) agent. GRU, LSTM and Attention are each network structures and mechanisms to efficiently and effectively extract time-series information. All components complement each others strengths well and allow us to utilize a fine-to-coarse data extraction procedure:
* GRU-architectures excel at identifying short- and mid-term dependencies, while being an inexpensive option to comparable alternatives.
* LSTM is capable of modelling short-, medium- and long-term, providing a more robust understanding of the time-series itself.
* Attention dynamically focuses on the most relevant parts of the input data, mitigating the dilution of important features across bulky datasets.

The **architecture** for our backbone is comparatively large, as listed down below. In retrospect, the size of the architecture can likely be decreased, as the original decisions were made before our significant improvements in terms of feature engineering. The design was as follows:
- 256-hidden-states x 4-layer GRU
- 128-hidden-states x 4-layer LSTM
- 100-dimensional Attention block

**Training** was conducted for 250 epochs originally, using a batch-size of 32, learning rate of 1e-4, weight-decay of 5e-4, dropout of 0.2 and gradient clipping at 1.0. All parameters used can be found in the ai6124_project.utils.params file. For **finetuning and transfer learning** the number of epochs were reduced to 50. This value was empirically selected, as it produced satisfactory results.

### MO-GenFIS

The head architecture was based upon [8] from the literature review, utilizing a multi-objective approach. This should not only produce more robust and invariant buy/sell/hold signals, but add an interesting dynamic due to the inherent capabilities for explaining decisions in fuzzy-inference systems. Not only receiving multiple recommendations, but multiple explanations for the subsequent decision based on a deterministic ruleset could provide a great interface for a fruitful Human-AI-Interface. The GenFIS is embedded into a PortfolioManager class, capable of buying/selling different quantities of stocks, as opposed to either buying/selling all possible assets. This should allow more nuanced behaviour for situations with uncertainty and should ensure higher profitability in due course.

* 4 genes were established, encoding the entire genome: the values of the input-membership-functions, output-membership-functions, the fuzzy rule-base and their corresponding consequents.
* Objectives employed for this project were cumulative reward, Sharpe ratio, and maximum drawdown (all of which are explained below), determining the maximum volatility of a portfolio in the given circumstances.
* The heads were trained separately using a genetic algorithm, optimizing input membership functions, output membership functions, rules and consequences.
* The trained models were unified to generate a single trading decision in a *Aggregation Layer* using a majority-based voting system: if more than 50% of single-objective models vote to buy/sell, the average recommendation of all positive voters is executed. If no consensus is reached, the current balance is held.

The GenFIS **architecture** was designed as follows: for each input, three triangular input membership functions were learned. Triangular input membership functions were selected as they greatly reduced computational complexity, as fewer points must be evaluated for centroid calculations during defuzzification. In addition, a soft-constraint on the number of inferred rules was added, setting this value to 20, to allow for manageable training times and easier explainability. This approach diverges from traditional FIS implementations. Defuzzification was performed by using max-centroid calculation. The consequences of these rules map to each output, which itself uses three distinct output membership functions.

**Training** was not performed on the entire training dataset, but rather on a randomly selected, 50-step interval within the training data. For these 50 timesteps, the agent was initialized with 1.000$ Cash and 10 stocks to simulate trading behavior on an ongoing position. This should allow more robust models to emerge, which perform well in uptrending and downtrending environments, regardless of their current position. A population of size 100 agents was trained for 100 generations. For genetic procreation, a mutation rate of 0.2, elitism threshold of 0.2 and probability-based (using a softmax across the total reward at the end of an episode) crossover with two cuts per gene were used. As mentioned earlier, these models were not finetuned on the alternate stock to test the hypothesis of learning general rules, which can be applied to different stocks given suitable backbone inputs.

Exemplary training trajectories of the first 50 training epochs for the cumulative reward agent:

<div style="display: flex; justify-content: space-around;">
<img src="https://github.com/tristan-till/ai6124-project/blob/main/images/train_genfis.png?raw=true" width="30%">
<img src="https://github.com/tristan-till/ai6124-project/blob/main/images/train_genfis_2.png?raw=true" width="30%">
<img src="https://github.com/tristan-till/ai6124-project/blob/main/images/train_genfis_3.png?raw=true" width="30%">
</div>

### Benchmark methods

Backbone and head were evaluated on different benchmarking metrics and methods, to gain a better understanding of their individual functionality. The former used time-series prediction metrics for predicting the stock price change in at the next timestep. The head used those predictions, as well as additional data, to manage a stock portfolio. Therefore, the latter was benchmarked on portfolio trajectory and financial gain during the testing period. Both types of benchmarking methods used simplified heuristics, like random walk for price predictions, and benchmark AI-models.

The data is split with a simple train, validation, test split of ratio 70%, 15%, 15% but without shuffling. The split is visible in the visualizations in the beginning.

### Backbone Metrics:
* **Mean Squared Error (MSE)**:
  Measures the average squared difference between predicted and actual values. Lower values indicate better performance, with a strong penalty for large errors.
* **Mean Absolute Error (MAE)**:
  Calculates the average of absolute differences between predicted and actual values. Unlike MSE, it is less sensitive to large errors.
* **Mean Absolute Percentage Error (MAPE)**:
  Represents the average of absolute percentage errors between predicted and actual values, often used to express accuracy as a percentage.
* **Hit Rate**:
  Evaluates the frequency of correct directional predictions, such as whether the predicted trend matches the actual trend.

### Head Metrics:
* **Cumulative Return**:
  Measures the total percentage return generated over a specified time period, reflecting overall profitability.
* **Sharpe Ratio**:
  Assesses risk-adjusted return by dividing excess returns (above a risk-free rate) by the standard deviation of returns, indicating reward per unit of risk.
* **Maximum Drawdown**:
  Calculates the largest peak-to-trough decline in portfolio value over a period, representing the worst-case loss scenario.

### Backbone Benchmarking Heuristics:
* **Zero-Change Benchmark**:
  Assumes no change in value from one period to the next, serving as a baseline for evaluating predictions.
* **Mean-Zero-Change Benchmark**:
  Predicts future values as the average of previous values, assuming mean reversion.
* **Previous-Day-Change Benchmark**:
  Uses the previous day's change as the prediction for the next period, reflecting a persistence-based heuristic.

### Head Benchmarking Heuristics:
* **Risk-Free-Rate Return**:
  Benchmarks performance against the return of risk-free assets, such as treasury bills, to assess opportunity cost.
* **Mean-Cumulative-Average Return**:
  Computes the average cumulative return over multiple time horizons, serving as a long-term performance benchmark.
* **Buy-&-Hold Return**:
  Measures the return achieved by holding an asset over the entire period without active trading, used as a baseline for active strategy comparisons.

As a **backbone benchmark model** an LSTM was used with 64 hidden states and 16 layers. The **head benchmark model** was a manually configured fuzzy inference system, applying potentially sensible rules for a reasonable trading system. Substantial effort was put into finding a sensible solution - all decisions originated from general, human assumptions. As this yielded terrible results unsuitable for a benchmark, these were refined using trial and error. This portion of the analysis took large amounts of time and was very insightful for what makes a *good model* work well. Nonetheless, referencing all parameters goes beyond the scope of this notebook, hence why we invite you to consult the linked github repository for further information!


## Results
### The Results in a Glance

Backbone Results

|     |  Benchmark Method | MSE    | MAE    | MAPE    | Hit Rate |
| --- | ------------      | ------ | ------ | ------- | -------- |
| XOM | Zero Change       | 0.9798 | 0.7768 | 100.00% | 47.74%   |
|     | Mean Change       | 0.9787 | 0.7757 | 103.16% | 52.26%   |
|     | Prev. Day         | 1.8351 | 1.0444 | 405.35% | 50.97%   |
|     | LSTM              | 0.9804 | 0.7771 | 101.16% | 47.74%   |
|     | **GRU-LSTM-Attention**  | 0.9441 | 0.7492 | 124.72% | 61.29%   |
| CVX | Zero Change       | 0.6848 | 0.6153 | 100.00% | 56.13%   |
|     | Mean Change       | 0.6826 | 0.6217 | 115.96% | 43.87%   |
|     | Prev. Day         | 1.1260 | 0.7927 | 285.34% | 60.00%   |
|     | LSTM              | 0.6849 | 0.6148 |  98.95% | 56.13%   |
|     | **GRU-LSTM-Attention** | 0.6695 | 0.6127 | 123.32% | 53.55%   |

Head Results

|     |  Benchmark Method | Cumulative Return | Sharpe Ratio | Max. Drawdown |
| --- | --------------    | ----------------- | ------------ | ------------- |
| XOM | Risk-Free Rate    | 2.14%             | 0.00%        | 0.00%         |
|     | XOM-MCA           | 2.37%             | 0.64%        | 7.47%         |
|     | XOM-B&H           | 4.61%             | 1.85%        | 10.68%        |
|     | Custom-FIS        | 1.29%             | \-0.93%      | 5.77%         |
|     | CR-FIS            | 6.54%             | 4.51%        | 5.10%         |
|     | SR-FIS            | 3.04%             | 1.16%        | 6.70%         |
|     | MD-FIS            | 1.39%             | \-1.44%      | 3.30%         |
| CVX | MO-FIS            | 6.49%             | 3.63%        | 5.10%         |
|     | Risk-Free Rate    | 2.14%             | 0.00%        | 0.00%         |
|     | XOM-MCA           | \-1.11%           | \-2.58%      | 9.46%         |
|     | XOM-B&H           | \-1.78%           | \-1.64%      | 14.86%        |
|     | Custom-FIS        | 0.80%             | \-2.47%      | 3.91%         |
|     | CR-FIS            | 1.32%             | \-0.47%      | 9.22%         |
|     | SR-FIS            | 2.81%             | 1.09%        | 6.10%         |
|     | MD-FIS            | 1.34%             | \-1.31%      | 4.22%         |
|     | MO-FIS            | 2.98%             | 1.14%        | 6.77%         |


The results above show that within its testing environment, the combined model of backbone and head **work surprisingly well**.
  * Our backbone shows higher predictive accuracy than not only the heuristic baselines, but also the baseline model.
  * It not only manages to outperform general heuristic benchmarks on rising stocks, but generates returns above the risk-free rate in a down-trending environment without specifically retraining for this purpose.
  * The use of multi-objective FIS, all optimized for a certain criterion, seemed to provide a more well-balanced resulting model, inheriting positive features from all child-models.
  * The consensus-based decisions seem to sacrifice very little performance in Cumulative Return and Sharpe Ratio, likely due to the fact that models optimized on those objectives may have an easier time to find consensus for a given decision.

Nonetheless, due to the the results of testing the head using different backbone models, led to valid questions about the **utility of the current setup**.
  * Varying the model predictions from their original values to constant zeros, perfect predictions, inverse predictions and random values led to our model performing equally well on the zero-benchmark as well as the perfect prediction benchmark.
  * However, using inverse predictions or random predictions significantly reduces predictive accuracy.

The conclusion does tie in well with our original hypothesis.
  * An efficient agent should not buy/sell on a day-to-day basis but act on **medium- and long-term dependencies**.
  * In fact, once the value gained by a price change is lower than the transaction fee, this behaviour is effectively penalized, in addition to the inherent uncertainty of prediction lowering the expected value of such an action.
  * Furthermore, when visualizing the testing results, the backbone model is showcased **not** trying to greedily predict the price delta of the next point in time, rather producing a smooth estimate of the current upwards and downwards trend.
  * Due to the inherent difficulty of predicting stock prices, neither a zero-prediction, nor a perfect prediction are too far off to influence a buy/sell decision within the fuzzy rule-base.
  * Fuzzy rules are used during prediction and errors within them can be fatal as seen by our other benchmarks.
  * However, due to the solid foundation of highly engineered input features they add to the existing pool of rich features for medium- and long-term dependencies which help the agent find ideal points to buy/sell.

Furthermore, we are also not convinced our fuzzy head can reliably *beat the market* in its current state.
  * Despite our best efforts, it is highly unlikely to encapsulate the intricacies and interactions of different mechanisms within a limited ruleset, even more so at our current limit of 20 rules.
  * Even with larger scaling, fuzzy systems have shown to reach their limits in terms of processing highly non-linear and high-dimensional data, calling for a powerful, bulky model to extract crucial information beforehand.

----

On a sidenote, when discussing trading with professionals in finance and asset management, they will likely reference another benchmark we purposefully omitted from our experiments as it is notoriously challenging to beat: the **S&P-500** or any related index for that matter.
  * Our model has shown that it can reliably trade on a single stock during up- and downtrends and generate an above average return
  * However, it cannot distinguish whether a buy decision is actually sensible within the context of the entire market.
  * In particular, over the same testing period, trading according to the monthly cumulative average heuristic would yield 8.5%, likely coinciding with lower risk and portfolio drawdown.
  * For the sake of a project/report, using XOM and CVX may be preferable as it is more likely to produce interesting discussions and findings.
  * However, it is crucial to acknowledge that if real money were on the line, strategies like these are more sensible and should be considered during benchmarking.
  * Further experiments may prove/disprove our model's capabilities on stocks/indices like these to eliminate this concern.

## Analysis of Results

The results above show that within its testing environment, the combined model of backbone and head **work surprisingly well**.
  * Our backbone shows higher predictive accuracy than not only the heuristic baselines, but also the baseline model.
  * It not only manages to outperform general heuristic benchmarks on rising stocks, but generates returns above the risk-free rate in a down-trending environment without specifically retraining for this purpose.
  * The use of multi-objective FIS, all optimized for a certain criterion, seemed to provide a more well-balanced resulting model, inheriting positive features from all child-models.
  * The consensus-based decisions seem to sacrifice very little performance in Cumulative Return and Sharpe Ratio, likely due to the fact that models optimized on those objectives may have an easier time to find consensus for a given decision.

Nonetheless, due to the the results of testing the head using different backbone models, led to valid questions about the **utility of the current setup**.
  * Varying the model predictions from their original values to constant zeros, perfect predictions, inverse predictions and random values led to our model performing equally well on the zero-benchmark as well as the perfect prediction benchmark.
  * However, using inverse predictions or random predictions significantly reduces predictive accuracy.

The conclusion does tie in well with our original hypothesis.
  * An efficient agent should not buy/sell on a day-to-day basis but act on **medium- and long-term dependencies**.
  * In fact, once the value gained by a price change is lower than the transaction fee, this behaviour is effectively penalized, in addition to the inherent uncertainty of prediction lowering the expected value of such an action.
  * Furthermore, when visualizing the testing results, the backbone model is showcased **not** trying to greedily predict the price delta of the next point in time, rather producing a smooth estimate of the current upwards and downwards trend.
  * Due to the inherent difficulty of predicting stock prices, neither a zero-prediction, nor a perfect prediction are too far off to influence a buy/sell decision within the fuzzy rule-base.
  * Fuzzy rules are used during prediction and errors within them can be fatal as seen by our other benchmarks.
  * However, due to the solid foundation of highly engineered input features they add to the existing pool of rich features for medium- and long-term dependencies which help the agent find ideal points to buy/sell.

Furthermore, we are also not convinced our fuzzy head can reliably *beat the market* in its current state.
  * Despite our best efforts, it is highly unlikely to encapsulate the intricacies and interactions of different mechanisms within a limited ruleset, even more so at our current limit of 20 rules.
  * Even with larger scaling, fuzzy systems have shown to reach their limits in terms of processing highly non-linear and high-dimensional data, calling for a powerful, bulky model to extract crucial information beforehand.

----

On a sidenote, when discussing trading with professionals in finance and asset management, they will likely reference another benchmark we purposefully omitted from our experiments as it is notoriously challenging to beat: the **S&P-500** or any related index for that matter.
  * Our model has shown that it can reliably trade on a single stock during up- and downtrends and generate an above average return
  * However, it cannot distinguish whether a buy decision is actually sensible within the context of the entire market.
  * In particular, over the same testing period, trading according to the monthly cumulative average heuristic would yield 8.5%, likely coinciding with lower risk and portfolio drawdown.
  * For the sake of a project/report, using XOM and CVX may be preferable as it is more likely to produce interesting discussions and findings.
  * However, it is crucial to acknowledge that if real money were on the line, strategies like these are more sensible and should be considered during benchmarking.
  * Further experiments may prove/disprove our model's capabilities on stocks/indices like these to eliminate this concern.

## Future work

The results section already uncovered some major flaws within the current setup: a prediction head targeted at predicting the price change in the next timestep unlikely to fully converge. Instead the model could learn to evaluate a short-, medium- and long-term price estimate, given three predefined confidence thresholds. If this produces satisfactory results, further research can be done removing **all** other supplementary datapoints and only using the backbone prediction, as well as portfolio information like liquidity, for generating buy/sell/signals.

Moreover, instead of constraining the model to one stock, it should be able to select from a wide variety of stocks within a certain sector i.e. trading on energy-stocks like XOM, CVX, CP, COP or CL=F simultaneously. Then, it would learn to differentiate the degree of profitability between different stocks, instead of greedily acting upon the movement within a single stock. When selecting a profitable sector, this could provide the first step to reliably beating a market average like the S&P-500. This would also allow the introduction of traditional alphas and betas (optimizing excess return compared to a benchmark index like SPX), which has the potential to further increase trading decisions.

Obviously, instead of adding multi-stock output we could increase the number of possible actions from buy/sell/hold to include short- and long-options trading, which could significantly increase the models profitability. Moreover, these may also require a more sophisticated logic in the aggregation layer beyond a simple majority-based *first-past-the-post* system. The aggregation layer itself could employ a computationally intelligent method, optimized during training, as a neural network, fuzzy based system or other AI/ML-architecture.

However, scaling up the model automatically causes the model's explainability to diminish. Another potential approach for improvement would be integrating our current model into an explainable Human-AI interface. Here, a stock broker could receive trading decisions based on different objectives, while receiving a comprehensible and understandable explanation. We still assume the combination of human and AI to outperform an autonomous AI agent. Analyzing this interaction is likely to yield even more fruitful insights on the shortcomings of the model, which, in turn, can be optimized as well.

## Thank You!

We found this project as a great learning opportunity and thoroughly enjoyed it even with all its challenges. If Neural Networks have taught us one thing, it is to learn from our mistakes. So, please feel free to reach out to us hitesh003@e.ntu.edu.sg and tr0001ll@e.ntu.edu.sg for any extra feedback you may have for us. We are extremely grateful to have an opportunity to work on such a fun, challenging and interesting project and will be even more grateful of any feedback you may have for us.


Todo:
* General
    X - Replace "x1, x2, ..." etc. from fis.explain() with actual input
    - Fix Hit Rate as Backbone Benchmark
    - Add S&P Monthly Cumulative Average Benchmark (same code for other stock already exists, reuse that function)
    - Separate plot paths for CR, SR and MD agents
* Notebook
    - Add comments/headers on where what is
    - Add data analysis (why did we choose those inputs for our backbone/head?)
    - Format Results (Add tables, graphs, buy/sell decisions etc.)
    - Reflection on results
* ReadMe
    - In-depth analysis/explanation of functionality
* Presentation Slides
    - Summarize findings
    - Literature Review I (Stock parameters)
    - Literature Review II (LSTM, GRU-LSTM-Attention, GenFIS, Multi-Objective Models)
    - Add proper visualizations
