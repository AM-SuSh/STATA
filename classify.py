import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import logging

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 忽略警告
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# ========== 配置信息 ==========
class Config:
    host = 'rm-uf64aj10c491mu974po.mysql.rds.aliyuncs.com'
    port = 3306
    user = 'Usst'
    password = 'Jyjqxxdzypzxxglxtyj0503'
    database = 'usst-shuju'

# 创建 SQLAlchemy 引擎
encoded_password = quote_plus(Config.password)
engine = create_engine(f"mysql+pymysql://{Config.user}:{encoded_password}@{Config.host}:{Config.port}/{Config.database}?charset=utf8mb4")

# 设置中文字体（强制使用 SimHei 或 YaHei）
zh_fonts = [f.name for f in fm.fontManager.ttflist if any(kw in f.name.lower() for kw in ['sim', 'yahei'])]
if zh_fonts:
    plt.rcParams['font.sans-serif'] = zh_fonts + ['SimHei', 'Microsoft YaHei']
else:
    # 如果未找到系统字体，则尝试手动加载本地字体文件（适用于 Linux/macOS）
    font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows 系统黑体字体路径
    if os.path.exists(font_path):
        my_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        logging.info("手动加载了 SimHei 字体")
    else:
        logging.warning("未找到中文字体，图表将无法显示中文")

plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")


# ========== 核心函数 ==========

def load_data(engine):
    """加载订单数据"""
    sql = "SELECT odoo_company_id, date_added, total, quantity FROM x_order_product"
    df = pd.read_sql(sql, engine)
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year'] = df['date_added'].dt.year
    df = df[df['year'].isin([2023, 2024])]
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    return df


def get_stats(g, prefix):
    """计算统计指标"""
    g = g[g['total'] > 0]

    stats = {
        f'{prefix}_total_sales': g['total'].sum(),
        f'{prefix}_total_quantity': g['quantity'].sum(),
        f'{prefix}_order_count': len(g),
        f'{prefix}_daily_quantity_variance': g.groupby(g['date_added'].dt.date)['quantity'].sum().var(),
        f'{prefix}_daily_total_variance': g.groupby(g['date_added'].dt.date)['total'].sum().var(),
        f'{prefix}_daily_orders_variance': g.groupby(g['date_added'].dt.date).size().var()
    }
    return stats


def compute_stats(df):
    """汇总每个公司的统计信息"""
    stats_list = []
    for company_id, group in df.groupby('odoo_company_id'):
        group_2023 = group[group['year'] == 2023]
        group_2024 = group[group['year'] == 2024]
        stats = {
            'odoo_company_id': company_id,
            **get_stats(group_2023, '2023'),
            **get_stats(group_2024, '2024'),
            **get_stats(group, 'all')
        }
        stats_list.append(stats)
    return pd.DataFrame(stats_list)


def split_into_classes(col_name, data_df):
    """按分位数划分6类"""
    col_data = data_df[[col_name, 'odoo_company_id']].copy()

    quantiles = data_df[col_name].quantile(np.linspace(0, 1, 7)).values[1:-1]
    thresholds = np.round(quantiles, 5).tolist()

    def assign_class(value):
        for i, t in enumerate(thresholds):
            if value <= t:
                return i + 1
        return 6

    col_data.loc[:, 'class'] = col_data[col_name].apply(assign_class)
    grouped = col_data.groupby('class', group_keys=False).apply(
        lambda g: list(zip(g['odoo_company_id'], g[col_name]))
    )
    return grouped.to_dict()


def save_classification_results(classification_results):
    """保存分类结果到 Excel，每个指标一个 sheet"""
    output_excel_file = "company_classification_by_metric.xlsx"
    with pd.ExcelWriter(output_excel_file) as writer:
        for metric, classes in classification_results.items():
            result_list = []
            for cls_num in range(1, 7):
                if cls_num in classes:
                    for cid, val in classes[cls_num]:
                        result_list.append({
                            "中台公司id": cid,
                            "值": round(val, 2) if isinstance(val, float) else val
                        })
            result_df = pd.DataFrame(result_list)
            safe_sheet_name = metric[:31]  # Sheet 名最长不超过 31 字符
            result_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
    logging.info(f"分类结果已写入 Excel 文件：{output_excel_file}")


def perform_clustering(stats_df):
    """执行聚类分析"""
    features = stats_df[[
        'all_total_sales',
        'all_total_quantity',
        'all_order_count',
        'all_daily_total_variance'
    ]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=6, random_state=42)
    stats_df['cluster'] = kmeans.fit_predict(scaled_features)
    return stats_df


def plot_clusters(stats_df):
    """绘制聚类散点图"""
    try:
        from matplotlib.font_manager import FontProperties
        font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows 黑体字体路径
        if os.path.exists(font_path):
            my_font = FontProperties(fname=font_path, size=12)
            title_font = my_font
        else:
            my_font = None
            title_font = None
            logging.warning("未找到 SimHei 字体，图表中文可能显示异常")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='all_total_sales', y='all_total_quantity', hue='cluster', data=stats_df, palette='viridis', alpha=0.8)
        plt.title('中台公司聚类可视化（销售额 vs 销售量）', fontproperties=my_font)
        plt.xlabel('总销售额', fontproperties=my_font)
        plt.ylabel('总销售量', fontproperties=my_font)
        plt.legend(title='聚类编号', prop=my_font if title_font else None)
        plt.grid(True)
        plt.savefig("company_clusters_visualization.png", dpi=300)
        plt.show()
        logging.info("聚类可视化图已保存为 company_clusters_visualization.png")
    except Exception as e:
        logging.error(f"绘图失败：{e}")

def save_cluster_results_to_excel(stats_df):
    """将聚类结果保存到 Excel 的一个单独 sheet 中"""
    cluster_result_file = "company_classification_by_metric.xlsx"

    # 提取聚类相关字段
    cluster_columns = [
        'odoo_company_id',
        'all_total_sales',
        'all_total_quantity',
        'all_order_count',
        'all_daily_total_variance',
        'cluster'
    ]

    cluster_result_df = stats_df[cluster_columns]

    # 写入 Excel 的“聚类分类”sheet
    with pd.ExcelWriter(cluster_result_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        cluster_result_df.to_excel(writer, sheet_name='聚类分类', index=False)

    logging.info(f"聚类结果已追加写入 Excel 文件：{cluster_result_file}")

# ========== 主程序入口 ==========
if __name__ == '__main__':
    try:
        df = load_data(engine)
        stats_df = compute_stats(df)

        metrics = [
            '2023_total_sales', '2023_total_quantity', '2023_order_count',
            '2024_total_sales', '2024_total_quantity', '2024_order_count',
            'all_total_sales', 'all_total_quantity', 'all_order_count',
            '2023_daily_quantity_variance', '2024_daily_quantity_variance', 'all_daily_quantity_variance',
            '2023_daily_total_variance', '2024_daily_total_variance', 'all_daily_total_variance',
            '2023_daily_orders_variance', '2024_daily_orders_variance', 'all_daily_orders_variance'
        ]

        classification_results = {}
        for metric in metrics:
            classification_results[metric] = split_into_classes(metric, stats_df)

        # 保存分类结果到 Excel，每个指标一个 sheet
        save_classification_results(classification_results)

        # 执行聚类并绘图
        stats_df = perform_clustering(stats_df)
        plot_clusters(stats_df)

        # 新增：保存聚类结果到 Excel
        save_cluster_results_to_excel(stats_df)

    except Exception as e:
        logging.error(f"运行过程中发生异常：{e}")
    finally:
        pass