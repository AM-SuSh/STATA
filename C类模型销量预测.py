import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 模型相关
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Lambda, Multiply
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

# 读取并筛选数据
def load_and_aggregate(path, pids):
    df = pd.read_csv(path, parse_dates=['date_added'])
    df = df[df['odoo_product_id'].isin(pids)]
    df['date'] = df['date_added'].dt.floor('D')
    agg = df.groupby(['odoo_product_id', 'date']).agg({
        'quantity': 'sum',
        'total': 'sum',
        'discount_total': 'sum'
    }).reset_index()
    return agg

# 填补缺失日期
def fill_dates(prod_df):
    idx = pd.date_range(prod_df['date'].min(), prod_df['date'].max(), freq='D')
    prod_df = prod_df.set_index('date').reindex(idx).fillna(0)
    prod_df.index.name = 'date'
    prod_df = prod_df.reset_index()
    prod_df['odoo_product_id'] = prod_df['odoo_product_id'].ffill().bfill()
    return prod_df

# 构造滞后与滚动特征
def make_features(df, lags=[1, 7, 14], windows=[7, 14]):
    df = df.sort_values('date').copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['quantity'].shift(lag)
    for w in windows:
        df[f'roll_mean_{w}'] = df['quantity'].shift(1).rolling(w).mean()
        df[f'roll_std_{w}'] = df['quantity'].shift(1).rolling(w).std()
    df = df.dropna()
    return df

# 生成时序序列
def create_sequences(X, y, time_steps=7):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# 计算RMSE
def calculate_rmse(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return np.sqrt(mse)

def train_and_predict_for_pid(pid, agg_data, TIME_STEPS=12):
    tmp = agg_data[agg_data['odoo_product_id'] == pid][['date', 'quantity', 'total', 'discount_total', 'odoo_product_id']]
    tmp_filled = fill_dates(tmp)
    data_feat = make_features(tmp_filled)

    if data_feat.empty:
        return None  # 无足够数据跳过

    data_feat = data_feat.set_index('date')

    features = [c for c in data_feat.columns if c not in ['quantity', 'odoo_product_id']]
    X = data_feat[features].values
    y = data_feat['quantity'].values

    if len(X) <= TIME_STEPS + 1:
        return None  # 数据太少跳过

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, TIME_STEPS)

    split_ratio = 0.7
    lstm_split = int(len(X_lstm) * split_ratio)
    X_lstm_train, X_lstm_test = X_lstm[:lstm_split], X_lstm[lstm_split:]
    y_lstm_train, y_lstm_test = y_lstm[:lstm_split], y_lstm[lstm_split:]

    X_rf, y_rf = X_scaled[TIME_STEPS:], y_scaled[TIME_STEPS:]
    X_rf_train, X_rf_test = X_rf[:lstm_split], X_rf[lstm_split:]
    y_rf_train, y_rf_test = y_rf[:lstm_split], y_rf[lstm_split:]

    # 构建自注意力LSTM模型
    input_layer = Input(shape=(TIME_STEPS, X_lstm.shape[2]))
    lstm_out = LSTM(64, return_sequences=True)(input_layer)
    drop1 = Dropout(0.2)(lstm_out)

    attention = Dense(1, activation='tanh')(drop1)
    attention = Lambda(lambda x: K.softmax(x, axis=1))(attention)
    context = Multiply()([drop1, attention])
    context = Lambda(lambda x: K.sum(x, axis=1))(context)

    dense1 = Dense(16, activation='relu')(context)
    output = Dense(1)(dense1)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_lstm_train, y_lstm_train,
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )

    pred_lstm_scaled = model.predict(X_lstm_test).flatten()
    pred_lstm = target_scaler.inverse_transform(pred_lstm_scaled.reshape(-1, 1)).flatten()

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_rf_train, y_rf_train)
    pred_rf_scaled = rf.predict(X_rf_test)
    pred_rf = target_scaler.inverse_transform(pred_rf_scaled.reshape(-1, 1)).flatten()

    y_test_actual = target_scaler.inverse_transform(y_lstm_test.reshape(-1, 1)).flatten()

    weight_lstm = 0.4
    weight_rf = 0.6
    pred_ensemble = pred_lstm * weight_lstm + pred_rf * weight_rf

    mse = mean_squared_error(y_test_actual, pred_ensemble)
    r2 = r2_score(y_test_actual, pred_ensemble)
    rmse = calculate_rmse(y_test_actual, pred_ensemble)

    return {
        'odoo_product_id': pid,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2
    }

def main():
    # 读含类别的表，筛选C类产品ID
    category_path = r"C:\Users\86137\Desktop\x_order_product_actual_amount.csv"
    df_cat = pd.read_csv(category_path)
    c_class_pids = df_cat[df_cat['ABC Category'] == 'C']['odoo_product_id'].unique()

    data_path = r"C:\Users\86137\Desktop\库存\x_order_product.csv"
    agg = load_and_aggregate(data_path, c_class_pids)

    results = []
    count = 0
    max_count = 200  # 最多跑50个产品

    for pid in c_class_pids:
        if count >= max_count:
            break
        print(f"正在预测产品ID {pid} ...")
        res = train_and_predict_for_pid(pid, agg)
        if res:
            results.append(res)
            count += 1
        else:
            print(f"产品ID {pid} 数据不足，跳过。")

    df_results = pd.DataFrame(results)
    print("\n所有C类产品预测指标：")
    print(df_results)

    avg_metrics = df_results[['RMSE', 'MSE', 'R2']].mean()
    print("\n所有C类产品预测指标平均值：")
    print(avg_metrics)

    output_path = r"C:\Users\86137\Desktop\c_class_products_forecast_metrics.csv"
    df_results.to_csv(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
