import numpy as np
from sklearn.metrics import mean_squared_error
import train

X_train, X_test, y_train, y_test = train.X_train, train.X_test, train.y_train, train.y_test
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")