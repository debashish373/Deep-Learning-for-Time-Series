
# Deep Learning for Time Series Prediction


Time series prediction of stock price movements poses a big challenge for most finance practitioners due to the highly stochastic nature of price changes. Traditional statistical time series approaches like ARIMA have been in use for long but it’s difficult to implement non-linearity if/when present in the data. On the other hand, advances in Deep Learning have opened a plethora of new possibilities for time series predictions especially when autoregressive methods fail or are of limited use. This project employs a special type of Deep Neural Nets called the Recurrent Neural Networks (LSTMs) to predict the direction of price movements for five European stocks from five different industries for varying window sizes, where a classification approach for 5D forward return predictions is used. Three different LSTM architectures were used and compared for accuracies. A comparison of LSTMs with tree based has also been carried out, and further possibilities of viewing the project as a regression problem were checked.
## Usage/Examples

```python
pip install -r requirements.txt

cd src

run main.py



  