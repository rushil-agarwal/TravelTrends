import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from keras import models
from keras import layers
from sklearn.neural_network import MLPRegressor

def train_regression_model(data, features, targets):
    X = data[features]
    Y = data[targets]

    X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(Y_test, predictions)

    error = mean_squared_error(Y_test, predictions)

    return model, r2, error

def get_season(month):
    m = month.month
    return (
        'Winter' if m in [12, 1, 2] else
        'Spring' if m in [3, 4, 5] else
        'Summer' if m in [6, 7, 8] else
        'Fall'
    )

def forecast_occupancy(data):
    forecasts = []
    forecast_months = 6
    max_month = data['month'].max()
    room_types = data['room_type'].unique()

    for room in room_types:
        df = data[data['room_type'] == room].copy()
        df['month_num'] = df['month'].map(lambda x: x.toordinal())
        df['month'] = pd.to_datetime(df['month'])
        df['month_sin'] = np.sin(2 * np.pi * df['month'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'].dt.month / 12)
        df['quarter'] = df['month'].dt.quarter
        df['year'] = df['month'].dt.year
        if 'lead_time' not in df.columns:
            df['lead_time'] = 50
        if 'adr' not in df.columns:
            df['adr'] = 100  
        if 'is_repeated_guest' not in df.columns:
            df['is_repeated_guest'] = 0

        time_features = ['month_num', 'month_sin', 'month_cos', 'quarter', 'year', 'lead_time', 'adr', 'is_repeated_guest'] + [col for col in df.columns if col.startswith('season_')]

        X = df[time_features].values
        y = df['bookings'].values.reshape(-1, 1)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        model = models.Sequential()
        model.add(layers.Dense(128, input_dim=X.shape[1], activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X_scaled, y_scaled, epochs=200, verbose=0)

        future_months = [max_month + pd.DateOffset(months=i) for i in range(1, forecast_months + 1)]
        future_data = pd.DataFrame({
            'month': future_months,
            'room_type': room,
            'month_num': [m.toordinal() for m in future_months],
            'month_sin': [np.sin(2 * np.pi * m.month / 12) for m in future_months],
            'month_cos': [np.cos(2 * np.pi * m.month / 12) for m in future_months],
            'quarter': [m.quarter for m in future_months],
            'year': [m.year for m in future_months],
            'lead_time': [50] * forecast_months,
            'adr': [100] * forecast_months,
            'is_repeated_guest': [0] * forecast_months
        })

        for season_col in [col for col in df.columns if col.startswith('season_')]:
            future_data[season_col] = 0

        for i, m in enumerate(future_months):
            season = get_season(m)
            season_col = f"season_{season}"
            if season_col in future_data.columns:
                future_data.at[i, season_col] = 1

        X_future = future_data[time_features].values
        X_future_scaled = scaler_X.transform(X_future)
        preds_scaled = model.predict(X_future_scaled)
        preds = scaler_y.inverse_transform(preds_scaled)

        result = pd.DataFrame({
            'month': future_months,
            'room_type': room,
            'predicted_bookings': preds.flatten()
        })

        forecasts.append(result)

    return pd.concat(forecasts)




def forecast_amenity_usage(monthly):
    features = ['adr', 'lead_time', 'stay_length', 'total_guests', 'total_revenue',
            'is_family', 'is_repeated_guest', 'month_sin', 'month_cos', 'month_num']
    target = 'amenity_utilization_ratio'
    X = monthly[features]
    y = monthly[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=6)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    forecast_months = X_test.copy()
    forecast_months['predicted_amenity_usage'] = y_pred
    forecast_months['month'] = monthly.loc[X_test.index, 'month']

    return forecast_months
