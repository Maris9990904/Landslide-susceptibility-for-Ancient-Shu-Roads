import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import os

# 设置中文字体（服务器环境可能不需要，但保持兼容性）
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建结果目录
result_dir = "xgboost_iris_results"
os.makedirs(result_dir, exist_ok=True)

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# 数据探索
print("数据集形状:", X.shape)
print("类别分布:", np.bincount(y))

# 数据框展示前5行
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [class_names[i] for i in y]
print("\n数据集前5行:")
print(df.head())

# 数据可视化 - 特征分布图
plt.figure(figsize=(12, 8))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    for species in class_names:
        plt.hist(df[df['species'] == species][feature], 
                 label=species, alpha=0.7, bins=15)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
plt.tight_layout()
plt.savefig(f"{result_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# 数据可视化 - 特征相关性热图
plt.figure(figsize=(10, 8))
corr = df.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.savefig(f"{result_dir}/feature_correlation.png", dpi=300, bbox_inches='tight')
plt.close()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 初始化XGBoost分类器
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',  # 多分类问题
    num_class=3,                # 类别数量
    random_state=42
)

# 训练模型
xgb_model.fit(X_train, y_train)

# 预测
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(f"{result_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 特征重要性可视化
feature_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.savefig(f"{result_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# 绘制ROC曲线（多类别的情况）
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_binarized.shape[1]

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {class_names[i]} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multi-class Classification')
plt.legend(loc="lower right")
plt.savefig(f"{result_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# 超参数调优
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n最佳参数:", grid_search.best_params_)
print("最佳交叉验证准确率:", grid_search.best_score_)

# 使用最佳参数的模型
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("\n调优后模型准确率:", accuracy_score(y_test, y_pred_best))

# 保存模型
import joblib
joblib.dump(best_model, f"{result_dir}/xgboost_iris_best_model.pkl")
print(f"\n最佳模型已保存至 {result_dir}/xgboost_iris_best_model.pkl")