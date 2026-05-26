import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_route_probability(file_path='RouteAndProb.xlsx'):
    """
    读入表格，统计 Route 列除 0 以外的其他值在 'WeightedEnsemble_L2_P_Positive' 
    列的概率分级个数，输出包含占比的表格，并绘制使用自定义颜色的**占比堆叠柱状图**。
    """
    route_col = 'Route'
    prob_col = 'WeightedEnsemble_L2_P_Positive'
    
    # 1. 读取数据
    try:
        # 使用您提供的文件路径
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"错误：文件未找到，请确保文件名为 '{file_path}' 且已上传。")
        return
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return

    # 2. 筛选数据：排除 Route = 0 的记录，并检查列是否存在
    if route_col not in df.columns or prob_col not in df.columns:
        print(f"错误：数据中缺少必需的列 '{route_col}' 或 '{prob_col}'。")
        return
        
    df_filtered = df[df[route_col] != 0].copy()
    
    if df_filtered.empty:
        print("警告：筛选后（Route != 0）的数据集为空，无法进行统计。")
        return

    # 3. 定义概率分级和自定义颜色
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
    # 定义 RGB 颜色 (R, G, B) 列表，顺序与 labels 对应 (Very Low -> Very High)
    rgb_colors_255 = [
        (56, 168, 0),    
        (139, 209, 0),   
        (255, 255, 0),   
        (255, 128, 0),   
        (255, 0, 0)      
    ]
    # 转换为 Matplotlib 要求的 0-1 范围
    mpl_colors = [(r/255, g/255, b/255) for r, g, b in rgb_colors_255]


    # 4. 创建分级列
    df_filtered['Probability_Level'] = pd.cut(
        df_filtered[prob_col].clip(0, 1), 
        bins=bins, 
        labels=labels, 
        right=True, 
        include_lowest=True,
        ordered=True
    )

    # 5. 统计个数并计算总数和占比
    counts_df = df_filtered.groupby([route_col, 'Probability_Level']).size().unstack(fill_value=0)
    counts_df = counts_df.reindex(columns=labels, fill_value=0)
    
    # 计算每条 Route 的总单元数
    counts_df['Total_Count'] = counts_df.sum(axis=1)
    
    # 计算占比 (Percent) 表格：核心绘图数据源
    results_percent = counts_df.iloc[:, :-1].div(counts_df['Total_Count'], axis=0) 
    
    # 6. 整合表格（与上一次提交相同）
    results_table = counts_df.copy()
    ordered_columns = []
    
    for level in labels:
        ordered_columns.append(level)
        results_table[f'{level}_P'] = results_percent[level].round(4)
        ordered_columns.append(f'{level}_P')
        
    ordered_columns.append('Total_Count')
    
    results_table = results_table[ordered_columns].reset_index()
    results_table.columns.name = None 

    # 7. 输出表格到 CSV
    output_file_csv = 'route_probability_counts_and_percentages.csv'
    results_table.to_csv(output_file_csv, index=False)
    print(f"\n统计结果表格（包含占比）已保存到文件：{output_file_csv}")
    print("\n统计结果 (包含个数和占比，部分展示):")
    print(results_table.head())

    # 8. 绘制**占比**堆叠柱状图
    fig, ax = plt.subplots(figsize=(12, 7))

    # >>> 关键修改：使用 results_percent 作为绘图数据源
    results_percent.plot(
        kind='bar', 
        stacked=True, 
        ax=ax,
        color=mpl_colors # 使用自定义颜色列表
    )

    # 图表定制
    ax.set_title('Percentage of Probability Levels by Route (Custom Color Scale)', fontsize=16)
    ax.set_xlabel(route_col, fontsize=14)
    # 更改 Y 轴标签
    ax.set_ylabel('Percentage of Data Points (数据点占比)', fontsize=14)
    
    # 格式化 Y 轴为百分比显示
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    
    # 设置图例标题和位置
    ax.legend(title='Probability Level', loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存图表
    output_file_png = 'route_probability_stacked_percentage.png' # 更改文件名以区分
    plt.tight_layout(rect=[0, 0, 1.0, 1])
    plt.savefig(output_file_png)
    plt.close(fig)
    print(f"\n占比堆叠柱状图已保存到文件：{output_file_png}")

# 运行代码
analyze_route_probability('RouteAndProb.xlsx')