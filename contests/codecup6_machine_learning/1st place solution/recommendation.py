import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate
import surprise as sp

train_data = pd.read_csv('train.csv', parse_dates=["date"])
test_data = pd.read_csv('test.csv', parse_dates=["date"])

# Prepare data

reader = Reader()
data = Dataset.load_from_df(train_data[['userId', 'itemId', 'rating']], reader)

# Use matrix factorization to implement collaborative filtering

alg = sp.SVD()
cross_validate(alg, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Prediction

test_data["prediction"] = test_data.apply(lambda x: alg.predict(x["userId"], x["itemId"]).est, axis=1)
test_data["prediction"] = np.clip(test_data["prediction"], 1, 5)
test_data["prediction"].to_csv('output.csv', index=False)
