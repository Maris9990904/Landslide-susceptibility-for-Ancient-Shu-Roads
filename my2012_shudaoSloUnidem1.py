import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 读取你的 Excel 数据
data = pd.read_excel('slope_units/SloUnitdem1.xls')  # 调整路径
data = data.drop(columns=['FID'])
# 随机分割数据集：30% 作为测试集，70% 作为训练集，使用随机种子 42
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# 将训练数据和测试数据转换为 AutoGluon 支持的格式
train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)

# 查看数据的一些基本信息
train_data.head()

# 设置标签列
label = 'y'

# 输出标签的描述性统计信息
print(f"Label description:\n{train_data[label].describe()}")

# 创建并训练模型
predictor = TabularPredictor(label=label).fit(train_data)

# 进行预测
y_pred = predictor.predict(test_data.drop(columns=[label]))

# 输出预测结果的前几行
print(f"Predictions:\n{y_pred.head()}")

# 评估模型性能
performance = predictor.evaluate(test_data, silent=True)
print(f"Model performance:\n{performance}")

# 获取并保存 leaderboard（即模型表现）
leaderboard = predictor.leaderboard(test_data)
print(f"Leaderboard:\n{leaderboard}")

# 将 leaderboard 保存到 CSV 文件
leaderboard.to_csv("slope_units/leaderboard_1km.csv", index=False)

# 保存测试集的真实标签和预测值到 CSV 文件
result_df = pd.DataFrame({
    'true_label': test_data[label],  # 测试集的真实标签
    'predicted': y_pred              # 模型预测的结果
})

# 保存到 CSV 文件
result_df.to_csv("slope_units/test_predictions_1km.csv", index=False)

# 获取所有模型的名称
leaderboard = predictor.leaderboard(test_data)

# 创建一个 DataFrame 来保存每个模型的预测结果
all_predictions = pd.DataFrame()

# 遍历每个模型并获取其预测结果
for model_name in leaderboard['model']:
    # 获取该模型的预测结果
    model = predictor._trainer.load_model(model_name)  # 加载每个模型
    y_pred = model.predict(test_data.drop(columns=[label]))  # 使用该模型进行预测
    
    # 将该模型的预测结果存入 DataFrame
    all_predictions[model_name] = y_pred

# 添加真实标签
all_predictions['true_label'] = test_data[label]

# 保存到 CSV 文件
all_predictions.to_csv("slope_units/all_model_predictions_1km.csv", index=False)

