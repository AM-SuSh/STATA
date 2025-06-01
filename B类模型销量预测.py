
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import holidays
from statsmodels.tsa.api import STL
import os
import warnings
import random

# 设置随机种子以确保结果可重复
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------- 数据预处理函数 -----------------
def load_single_product_data(path, pid):
    """加载单个产品的数据"""
    df = pd.read_csv(path, parse_dates=['date_added'])
    df = df[df['odoo_product_id'] == pid]  # 筛选单个产品
    df['date'] = df['date_added'].dt.floor('D')
    agg = df.groupby(['date'])['quantity'].sum().to_frame()
    cn_holidays = holidays.CountryHoliday('CN', years=range(2010, 2024))
    agg['is_holiday'] = agg.index.map(lambda x: x in cn_holidays).astype(int)
    return agg

def fill_missing_dates(df):
    idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(idx)
    df['quantity'] = df['quantity'].interpolate(method='time')
    df['is_holiday'] = df['is_holiday'].fillna(0)
    return df

def remove_outliers(df, column='quantity', threshold=1.5):
    rolling_median = df[column].rolling(7, min_periods=1).median()
    diff = np.abs(df[column] - rolling_median)
    mad = diff.rolling(7, min_periods=1).median()
    cutoff = threshold * mad
    df[column] = np.where(diff > cutoff, rolling_median, df[column])
    return df

# ----------------- 优化后的特征工程 -----------------
def create_advanced_features(df):
    df = df.copy()
    # 基础时间特征
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['dayofmonth'] = df.index.day
    
    # 节假日扩展特征
    df['pre_holiday'] = df['is_holiday'].shift(-1).fillna(0)
    df['post_holiday'] = df['is_holiday'].shift(1).fillna(0)
    df['holiday_3day'] = df['is_holiday'].rolling(3, min_periods=1).max()
    
    # 交互特征
    df['weekend_holiday'] = df['is_weekend'] * df['is_holiday']
    
    # 周期特征增强
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear/365)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear/365)
    df['sin_month'] = np.sin(2 * np.pi * df['month']/12)
    df['cos_month'] = np.cos(2 * np.pi * df['month']/12)
    
    # 滞后特征扩展
    lags = [1,2,3,7,14,21,28]
    for lag in lags:
        df[f'lag_{lag}'] = df['quantity'].shift(lag)
    
    # 滑动窗口特征改进
    windows = [3,7,14,21]
    for w in windows:
        df[f'mean_{w}'] = df['quantity'].rolling(w).mean()
        df[f'std_{w}'] = df['quantity'].rolling(w).std()
        df[f'min_{w}'] = df['quantity'].rolling(w).min()
        df[f'max_{w}'] = df['quantity'].rolling(w).max()
        df[f'ewm_{w}'] = df['quantity'].ewm(span=w).mean()
    
    # STL分解 - 添加随机种子确保可重复性
    try:
        stl = STL(df['quantity'], period=7, seasonal=13)
        res = stl.fit()
        df['trend'] = res.trend
        df['seasonal'] = res.seasonal
        df['residual'] = res.resid
    except Exception as e:
        print(f"STL分解失败: {str(e)}")
        df['trend'] = df['quantity'].rolling(14).mean()
        df['seasonal'] = 0
        df['residual'] = 0
    
    # 趋势相关特征
    df['trend_diff'] = df['trend'].diff()
    df['trend_accel'] = df['trend_diff'].diff()
    
    # 删除中间列
    df = df.drop(columns=['dayofweek', 'month', 'quarter'])
    
    return df.dropna()

# ----------------- 模型架构保持不变 -----------------
class SalesPredictor(nn.Module):
    def __init__(self, input_dim, lstm_dim=128, cnn_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2))
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_dim*2 + 64, 
            num_heads=4,
            dropout=0.2,
            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim*2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :]
        cnn_input = x.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_input)
        cnn_feat = cnn_out.mean(dim=-1)
        combined = torch.cat([lstm_feat, cnn_feat], dim=1)
        attn_in = combined.unsqueeze(1)
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)
        return self.fc(attn_out.squeeze(1))

# ----------------- 数据管道保持不变 -----------------
class SalesDataset(Dataset):
    def __init__(self, data, window=42):
        self.data = data
        self.window = window
        
    def __len__(self):
        return len(self.data) - self.window
    
    def __getitem__(self, idx):
        features = self.data[idx:idx+self.window, :-1]
        target = self.data[idx+self.window, -1]
        return torch.FloatTensor(features), torch.FloatTensor([target])

# ----------------- 训练函数保持不变 -----------------
def train_model(model, train_loader, val_loader, epochs=100):
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X)
                val_loss += criterion(pred, y).item()
        
        avg_val = val_loss/len(val_loader)
        scheduler.step(avg_val)
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Epoch {epoch+1:03d} | Val Loss Improved: {avg_val:.4f}')
        else:
            print(f'Epoch {epoch+1:03d} | Val Loss: {avg_val:.4f}')
            
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# ----------------- 单个产品预测函数 -----------------
def predict_single_product(pid, order_path, window=56, epochs=100):
    """对单个产品进行预测并返回R²分数和预测数据"""
    print(f"\n{'='*50}")
    print(f"开始处理产品: {pid}")
    
    # 加载和处理数据
    raw = load_single_product_data(order_path, pid)
    if len(raw) < window + 30:  # 确保足够的数据点
        print(f"产品 {pid} 数据不足({len(raw)}条)，跳过预测")
        return None, None, None, None, None
    
    filled = fill_missing_dates(raw)
    cleaned = remove_outliers(filled)
    featured = create_advanced_features(cleaned)
    
    # 改用平方根变换
    featured['sqrt_quantity'] = np.sqrt(featured['quantity'])
    featured = featured.drop(columns=['quantity'])
    
    # 标准化
    scaler_x = RobustScaler()
    scaler_y = RobustScaler()
    y = scaler_y.fit_transform(featured[['sqrt_quantity']])
    X = scaler_x.fit_transform(featured.drop(columns=['sqrt_quantity']))
    processed = np.hstack([X, y])
    
    # 创建数据集 - 固定数据分割
    dataset = SalesDataset(processed, window)
    if len(dataset) < 100:  # 确保足够的数据点
        print(f"产品 {pid} 有效数据不足({len(dataset)}条)，跳过预测")
        return None, None, None, None, None
    
    # 固定训练测试分割
    train_size = int(0.85 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(
        dataset, 
        [train_size, len(dataset)-train_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    
    # 初始化模型
    model = SalesPredictor(input_dim=X.shape[1])
    
    # 训练
    trained_model = train_model(model, train_loader, val_loader, epochs)
    
    # 评估
    full_loader = DataLoader(dataset, batch_size=64)
    preds, actuals = [], []
    trained_model.eval()
    with torch.no_grad():
        for X, y in full_loader:
            pred = trained_model(X)
            # 反变换
            sqrt_pred = scaler_y.inverse_transform(pred.numpy())
            pred_inv = sqrt_pred ** 2
            sqrt_actual = scaler_y.inverse_transform(y.numpy())
            actual_inv = sqrt_actual ** 2
            preds.extend(pred_inv.flatten())
            actuals.extend(actual_inv.flatten())
    
    # 计算R²
    r2 = r2_score(actuals, preds)
    print(f"产品 {pid} 预测完成 - R²: {r2:.4f}")
    
    # 未来预测
    future_steps = 30
    last_window = processed[-window:, :-1]
    
    future_preds = []
    for _ in range(future_steps):
        with torch.no_grad():
            input_tensor = torch.FloatTensor(last_window).unsqueeze(0)
            pred = trained_model(input_tensor)
            
            # 反变换
            pred_scaled = pred.numpy().reshape(-1, 1)
            sqrt_pred = scaler_y.inverse_transform(pred_scaled)[0][0]
            pred_inv = sqrt_pred ** 2
            
            future_preds.append(pred_inv)
            
            # 更新窗口
            new_value = np.sqrt(pred_inv)
            new_row = np.append(last_window[-1, 1:], new_value)
            last_window = np.vstack([last_window, new_row])[1:]
    
    # 创建预测结果
    future_dates = pd.date_range(featured.index[-1], periods=future_steps+1, freq='D')[1:]
    future_df = pd.DataFrame({
        '日期': future_dates,
        '预测销量': future_preds
    })
    
    return r2, actuals, preds, featured.index[window:], future_df

# ----------------- 主程序修改 -----------------
def main():
    # 加载产品分类数据
    abc_path = r"C:\Users\86178\Desktop\库存优化\数据分类\数据分类\x_order_product_actual_amount.csv"
    order_path = r"C:\Users\86178\Desktop\库存优化\x_order_product.csv"
    
    abc_df = pd.read_csv(abc_path)
    a_ids = abc_df[abc_df['ABC Category'] == 'B']['odoo_product_id'].unique()
    
    # 选择前100个B类产品（固定数量以便结果可重复）
    num_products = min(100, len(a_ids))
    selected_ids = a_ids[:num_products]
    print(f"选择了 {num_products} 个B类产品进行预测")
    
    # 存储结果 - 使用两个列表分别存储产品ID和R²分数
    success_ids = []   # 成功预测的产品ID
    r2_scores = []     # 对应的R²分数
    best_r2 = -float('inf')
    best_pid = None
    best_actuals = None
    best_preds = None
    best_dates = None
    best_future = None
    
    # 创建结果目录
    os.makedirs("product_predictions", exist_ok=True)
    
    # 对每个产品进行预测
    for i, pid in enumerate(selected_ids):
        r2, actuals, preds, dates, future_df = predict_single_product(pid, order_path)
        
        if r2 is not None:
            # 添加到结果列表
            success_ids.append(pid)
            r2_scores.append(r2)
            
            # 更新最佳产品
            if r2 > best_r2:
                best_r2 = r2
                best_pid = pid
                best_actuals = actuals
                best_preds = preds
                best_dates = dates
                best_future = future_df
    
    # 计算平均R²
    if r2_scores:
        avg_r2 = np.mean(r2_scores)
        print(f"\n{'='*50}")
        print(f"所有产品预测完成!")
        print(f"成功预测产品数量: {len(success_ids)}/{len(selected_ids)}")
        print(f"平均R²: {avg_r2:.4f}")
        print(f"最佳产品ID: {best_pid}, R²: {best_r2:.4f}")
        
        # 保存最佳产品结果
        if best_pid:
            # 保存未来预测
            future_path = f"product_predictions/best_product_{best_pid}_future.csv"
            best_future.to_csv(future_path, index=False, encoding='utf-8-sig')
            print(f"最佳产品未来预测已保存到: {future_path}")
            
            # 绘制最佳产品图表
            plt.figure(figsize=(16, 8))
            plt.plot(best_dates, best_actuals, label='实际销量', color='#1f77b4', linewidth=2, alpha=0.9)
            plt.plot(best_dates, best_preds, label='模型预测', color='#ff7f0e', linestyle='--', linewidth=1.5)
            
            plt.title(f'产品 {best_pid} 销量预测 (R²={best_r2:.3f})', fontsize=16)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('销量', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left', fontsize=12)
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            
            # 保存图表
            plot_path = f"product_predictions/best_product_{best_pid}_prediction_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"最佳产品预测图表已保存到: {plot_path}")
            plt.show()
            
            # 保存所有成功预测产品的R²结果
            r2_df = pd.DataFrame({
                'Product_ID': success_ids,
                'R2_Score': r2_scores
            })
            r2_df.to_csv("product_predictions/successful_products_r2_scores.csv", index=False)
            print(f"成功预测产品的R²分数已保存到: product_predictions/successful_products_r2_scores.csv")
            
            # 保存失败产品列表
            failed_ids = list(set(selected_ids) - set(success_ids))
            if failed_ids:
                failed_df = pd.DataFrame({
                    'Product_ID': failed_ids,
                    'Reason': '数据不足'
                })
                failed_df.to_csv("product_predictions/failed_products.csv", index=False)
                print(f"失败预测产品列表已保存到: product_predictions/failed_products.csv")
    else:
        print("没有产品成功完成预测")

if __name__ == "__main__":
    main()