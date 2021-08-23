
# Deep Learning for Time Series Prediction


This project employs Long Short Term Memory Networks (LSTMs) to predict the direction of price movements for five European stocks from five different industries for varying window sizes, where a classification approach for 5D forward return predictions is used. Three different LSTM architectures have been used and compared for accuracies. A comparison of LSTMs with tree based models has also been carried out, and further possibilities of viewing the project as a regression problem were checked.

Feature selection was carried out using SOM/KMeans

Hyperparameters for the LSTM architectures were optimized using Optuna.

Further research on improving AUC scores are ongoing.
## Usage/Examples

```python
pip install -r requirements.txt

cd src

python main.py --ticker TTE.PA --timestep 10 --forward 5 --epochs 10 --fs SOM



  
