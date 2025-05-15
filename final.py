# -*- coding: utf-8 -*-
"""
Created on Tue May 13 13:21:17 2025

@author: 86178
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.dates as mdates

# 数据预处理函数
def load_and_aggregate(path, pids):
    df = pd.read_csv(path, parse_dates=['date_added'])
    df = df[df['odoo_product_id'].isin(pids)]
    df["date"] = pd.to_datetime(df["date_added"].dt.date)
    agg_df = df.resample('D', on='date').agg({
        'quantity': 'sum',
        'odoo_inventory': 'last',
        'total': 'sum',
        'discount_total': 'sum'
    }).fillna(0)
    
    # 添加周末标识
    agg_df['is_weekend'] = (agg_df.index.weekday >= 5).astype(int)
    return agg_df

def prepare_data(file_path, product_ids):
    daily_df = load_and_aggregate(file_path, product_ids)
    
    # 时间特征
    daily_df['day_of_month_sin'] = np.sin(2*np.pi*daily_df.index.day/31)
    daily_df['day_of_month_cos'] = np.cos(2*np.pi*daily_df.index.day/31)
    daily_df['month_sin'] = np.sin(2*np.pi*daily_df.index.month/12)
    daily_df['month_cos'] = np.cos(2*np.pi*daily_df.index.month/12)
    
    # 滞后特征（确保使用历史数据）
    for lag in range(1, 8):
        daily_df[f'lag_{lag}'] = daily_df['quantity'].shift(lag)
    
    # 统计特征（使用历史窗口）
    daily_df['ma_7'] = daily_df['quantity'].rolling(7).mean().shift(1)
    daily_df['std_7'] = daily_df['quantity'].rolling(7).std().shift(1)
    daily_df['ema_3'] = daily_df['quantity'].ewm(span=3).mean().shift(1)
    
    # 交互特征
    daily_df['price_ratio'] = daily_df['total'] / (daily_df['quantity'] + 1e-6)
    
    # 删除无效值
    daily_df = daily_df.dropna()
    
    features = [
        'total', 'discount_total', 'odoo_inventory',
        'day_of_month_sin', 'day_of_month_cos',
        'month_sin', 'month_cos', 'is_weekend',
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
        'ma_7', 'std_7', 'ema_3', 'price_ratio'
    ]
    
    return daily_df[features].values, daily_df['quantity'].values, daily_df.index

# 数据标准化
def scale_data(X, y):
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, X_scaler, y_scaler

# 时间窗口生成
def create_time_windows(X, y, window_size=7):
    X_windowed = []
    y_windowed = []
    for i in range(len(X) - window_size):
        # 获取7天特征并转置为(特征数, 时间步)
        window = X[i:i+window_size].T  # (特征数, 窗口大小)
        X_windowed.append(window)
        y_windowed.append(y[i+window_size])  # 预测第8天
    return np.array(X_windowed), np.array(y_windowed)

# CNN模型架构
class CNNModel(nn.Module):
    def __init__(self, input_channels, seq_length):
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 输入形状: (batch, 特征数, 时间步)
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 时间步长减半
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

# 训练流程
def train_model(model, train_loader, epochs=300):
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    history = {'train_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_loss = train_loss/len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f}')
    
    model.load_state_dict(torch.load('best_model.pth'))
    return history

# 可视化函数
def plot_results(y_true, y_pred, dates):
    plt.figure(figsize=(20, 10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    
    plt.plot(dates, y_true, label='Actual', color='#1f77b4', linewidth=2)
    plt.plot(dates, y_pred, label='Predicted', color='#ff7f0e', linestyle='--', linewidth=2)
    
    plt.title('7-Day Window CNN Prediction', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Quantity', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 主程序
def main():
    # 数据准备
    csv_path = r"C:\Users\86178\Desktop\x_order_product.csv"
    X, y, dates = prepare_data(csv_path, [16420])
    
    # 数据标准化
    X_scaled, y_scaled, X_scaler, y_scaler = scale_data(X, y)
    
    # 生成时间窗口
    X_windowed, y_windowed = create_time_windows(X_scaled, y_scaled)
    dates_windowed = dates[7:]  # 对齐时间戳
    
    # 转换为张量
    X_tensor = torch.FloatTensor(X_windowed)
    y_tensor = torch.FloatTensor(y_windowed).unsqueeze(1)
    
    # 创建数据加载器（保持时间顺序）
    train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    input_channels = X_windowed.shape[1]  # 特征数
    seq_length = X_windowed.shape[2]      # 时间步长
    model = CNNModel(input_channels, seq_length)
    
    # 训练模型
    history = train_model(model, train_loader, epochs=300)
    
    # 全量预测
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_tensor).numpy().flatten()
    
    # 反标准化
    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_orig = y_scaler.inverse_transform(y_windowed.reshape(-1, 1)).flatten()
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_orig, preds))
    mae = mean_absolute_error(y_orig, preds)
    r2 = r2_score(y_orig, preds)
    
    print("\n================ Final Results ================")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    print("==============================================")
    
    # 可视化完整结果
    plot_results(y_orig, preds, dates_windowed)

if __name__ == '__main__':
    main()