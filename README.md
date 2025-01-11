
# Air Quality Prediction with Random Forest Regressor

This project uses air quality data to predict the concentration of Carbon Monoxide (`CO(GT)`) based on various sensor readings and environmental factors. The data is preprocessed, scaled, and then used to train a Random Forest Regressor model. The trained model is saved for future predictions.

## Requirements

Before running the code, ensure you have the following dependencies installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`

You can install them using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Dataset

The dataset `airdata.csv` contains the following columns:

- `CO(GT)`: Carbon monoxide concentration (target variable)
- `PT08.S1(CO)`, `NMHC(GT)`, `C6H6(GT)`, `PT08.S2(NMHC)`, `NOx(GT)`, `PT08.S3(NOx)`, `NO2(GT)`, `PT08.S4(NO2)`, `PT08.S5(O3)`: Various sensor readings for pollutants and gases
- `T`: Temperature
- `RH`: Relative Humidity
- `AH`: Absolute Humidity
- `Datetime`: Date and time of the observation

## Steps

1. **Data Preprocessing**:
   - Load the dataset.
   - Clean the data by converting columns to numeric types and handling missing values.
   - Drop irrelevant columns such as `C6H6(GT)`.
   - Combine the `Date` and `Time` columns into a single `Datetime` column.
   - Scale the features using `MinMaxScaler`.

2. **Model Training**:
   - Split the data into training and testing sets.
   - Train a Random Forest Regressor model to predict `CO(GT)` values.

3. **Model Evaluation**:
   - Evaluate the model's performance using Mean Squared Error (MSE).
   - Plot the actual vs predicted values of `CO(GT)`.

4. **Model Saving**:
   - Save the trained model using `joblib` for future use.

## Code Walkthrough

### Data Preprocessing

```python
data = pd.read_csv("airdata.csv", delimiter=";")
data = data.loc[:, ~data.columns.str.contains('^Unnamed|;;')]
# Convert columns to numeric
numeric_columns = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', ...]
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
```

### Feature Scaling

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['Datetime']))
data_scaled = pd.DataFrame(scaled_features, columns=data.columns[:-1])
```

### Model Training

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = data_scaled.drop(columns=['CO(GT)'])
y = data_scaled['CO(GT)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Model Evaluation

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

### Model Saving

```python
import joblib
joblib.dump(model, "air_quality_model.pkl")
```

## Visualizations

The program plots a scatter plot comparing actual vs predicted values of `CO(GT)`:

```python
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel("Actual CO(GT)")
plt.ylabel("Predicted CO(GT)")
plt.title("Actual vs Predicted CO(GT)")
plt.show()
```

## Model Usage

Once the model is saved, you can load it for future predictions:

```python
model = joblib.load("air_quality_model.pkl")
predictions = model.predict(X_test)
```

## Conclusion

This project demonstrates how to preprocess air quality data, train a Random Forest Regressor model, and save the model for future predictions. The model can be used to predict the concentration of `CO(GT)` based on various environmental and sensor data.
