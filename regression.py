import requests
import pandas as pd
import scipy
import numpy
import sys
import io
from sklearn.linear_model import LinearRegression 
TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    s=response.content
    d=pd.read_csv(io.StringIO(s.decode('utf-8')))
    d=d.T
    d.reset_index(level=0,inplace=True)
    d=d[1:]
    d.columns=['area','price']
    model=LinearRegression()
    model.fit(d['area'],d['price'])
    area=area.reshape(-1,1)
    print(model.coef_)
    print(model.intercept_)
    return model.predict(area)

    # YOUR IMPLEMENTATION HERE
    ...


if __name__ == "__main__":
    
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
