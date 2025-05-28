import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import warnings

# 忽略 SettingWithCopyWarning 和其他 Future/DeprecationWarning
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# 数据库连接配置
host = 'rm-uf64aj10c491mu974po.mysql.rds.aliyuncs.com'
port = 3306
user = 'Usst'
password = 'Jyjqxxdzypzxxglxtyj0503'
database = 'usst-shuju'

# 使用 quote_plus 编码密码中的特殊字符
encoded_password = quote_plus(password)

# 创建 SQLAlchemy 引擎
engine = create_engine(f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{database}?charset=utf8mb4")

try:
    # 查询所有订单数据（使用 SQLAlchemy engine）
    sql = "SELECT odoo_company_id, date_added, total, quantity FROM x_order_product"
    df = pd.read_sql(sql, engine)

    # 转换时间字段，并处理非法日期
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year'] = df['date_added'].dt.year

    # 筛选2023和2024年数据
    df = df[df['year'].isin([2023, 2024])]

    # 强制转换数值列，处理非数字问题
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

    # 准备汇总 DataFrame
    stats_list = []

    for company_id, group in df.groupby('odoo_company_id'):
        # 按年份拆分
        group_2023 = group[group['year'] == 2023]
        group_2024 = group[group['year'] == 2024]

        def get_stats(g, prefix):
            # 排除 total <= 0 的记录（如退款单、测试单等）
            g = g[g['total'] > 0]

            # 【销售额】总销售额：该公司的订单金额总和
            total_sales = g['total'].sum()

            # 【销售量】总商品数量：该公司的商品销售总量
            total_quantity = g['quantity'].sum()

            # 【订单数】订单总数：该公司的订单条数（即 len(g)）
            order_count = len(g)

            # 【日销量波动】按天聚合销售数量后求方差，反映每日销售波动程度
            daily_quantity_variance = g.groupby(g['date_added'].dt.date)['quantity'].sum().var()

            # 【日销售额波动】按天聚合销售额后求方差，反映每日收入稳定性
            daily_total_variance = g.groupby(g['date_added'].dt.date)['total'].sum().var()

            # 【日订单数波动】按天统计订单数后求方差，反映订单频次变化情况
            daily_orders_variance = g.groupby(g['date_added'].dt.date).size().var()

            return {
                f'{prefix}_total_sales': total_sales,
                f'{prefix}_total_quantity': total_quantity,
                f'{prefix}_order_count': order_count,
                f'{prefix}_daily_quantity_variance': daily_quantity_variance,
                f'{prefix}_daily_total_variance': daily_total_variance,
                f'{prefix}_daily_orders_variance': daily_orders_variance
            }

        stats_2023 = get_stats(group_2023, '2023')
        stats_2024 = get_stats(group_2024, '2024')
        stats_all = get_stats(group, 'all')

        full_stats = {
            'odoo_company_id': company_id,
            **stats_2023,
            **stats_2024,
            **stats_all
        }
        stats_list.append(full_stats)

    # 转为 DataFrame
    stats_df = pd.DataFrame(stats_list)

    # 定义函数：将一列划分为6类，并返回 {类别: [(公司ID, 值), ...]} 的字典
    def split_into_classes(col_name, data_df):
        col_data = data_df[[col_name, 'odoo_company_id']].copy()  # 加上 .copy() 避免 SettingWithCopyWarning

        # 获取分位点（六等分）
        quantiles = data_df[col_name].quantile(np.linspace(0, 1, 7)).values[1:-1]
        thresholds = np.round(quantiles, 5).tolist()

        # 分类函数：根据阈值分配类别
        def assign_class(value):
            for i, t in enumerate(thresholds):
                if value <= t:
                    return i + 1
            return 6

        # 添加分类字段
        col_data.loc[:, 'class'] = col_data[col_name].apply(assign_class)

        # 按类别分组并列出公司ID及对应值
        grouped = col_data.groupby('class', group_keys=False).apply(
            lambda g: list(zip(g['odoo_company_id'], g[col_name]))
        )
        return grouped.to_dict()

    # 所有要分类的指标
    metrics = [
        '2023_total_sales', '2023_total_quantity', '2023_order_count',
        '2024_total_sales', '2024_total_quantity', '2024_order_count',
        'all_total_sales', 'all_total_quantity', 'all_order_count',
        '2023_daily_quantity_variance', '2024_daily_quantity_variance', 'all_daily_quantity_variance',
        '2023_daily_total_variance', '2024_daily_total_variance', 'all_daily_total_variance',
        '2023_daily_orders_variance', '2024_daily_orders_variance', 'all_daily_orders_variance'
    ]

    # 构建所有分类结果
    classification_results = {}
    for metric in metrics:
        classification_results[metric] = split_into_classes(metric, stats_df)

    # 输出到 txt 文件
    output_txt_file = "company_classification_by_metric.txt"
    with open(output_txt_file, "w", encoding="utf-8") as f:
        for metric, classes in classification_results.items():
            f.write(f"【{metric} 分类结果】\n\n")
            for cls_num in range(1, 7):
                if cls_num in classes:
                    f.write(f"第{cls_num}类：\n")
                    for cid, val in classes[cls_num]:
                        f.write(f"中台公司id：{cid} | 值：{round(val, 2) if isinstance(val, float) else val}\n")
                    f.write("\n")
            f.write("-" * 80 + "\n\n")

    print(f"分类结果已写入 {output_txt_file}")

    # 输出到 Excel，每个指标一个 sheet
    output_excel_file = "company_classification_by_metric.xlsx"
    with pd.ExcelWriter(output_excel_file) as writer:
        for metric, classes in classification_results.items():
            result_list = []
            for cls_num in range(1, 7):
                if cls_num in classes:
                    for cid, val in classes[cls_num]:
                        result_list.append({
                            "指标名": metric,
                            "类别": f"第{cls_num}类",
                            "中台公司id": cid,
                            "值": val
                        })
            result_df = pd.DataFrame(result_list)
            safe_sheet_name = metric[:31]  # Sheet 名最长不超过 31 字符
            result_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

    print(f"分类结果已写入 Excel 文件：{output_excel_file}")

finally:
    pass