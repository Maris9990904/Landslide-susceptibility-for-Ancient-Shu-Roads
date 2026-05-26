import numpy as np
import os
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pandas as pd
from tabpfn import TabPFNClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- 1. 数据加载与预处理 ---
#蜀道第一次大修，用SA,不采样，第二次大修，去掉SA,采样同数非滑坡，按随机7:3划分，ys,2w数据的
# 读取你的 Excel 数据

data_orig = pd.read_excel('dataset/Beizhan2.xls')  # 调整路径至 tmp.xls

# if 'SH' in data_orig.columns:
#     data_orig = data_orig.drop(columns=['SH'])
#     print("Column 'SH' was successfully removed from the dataset.")
# if 'CUR' in data_orig.columns:
#     data_orig = data_orig.drop(columns=['CUR'])
#     print("Column 'CUR' was successfully removed from the dataset.")

# --- 1.2 明确非特征列和标签列 ---
# 根据用户描述，前三列是非特征列：FID, area, grid
non_feature_cols = ['FID', 'area', 'gridcode','is_test']

# 检查 FID 列是否存在并提取 (假设 FID 在 data_orig 中)
if 'FID' in data_orig.columns:
    FID = data_orig['FID'].copy()
else:
    # 假设如果 FID 不存在，则使用索引作为 ID
    FID = pd.Series(data_orig.index.values, name='FID')
    print("Warning: 'FID' column not found. Using DataFrame index as FID.")

# 设置标签列
label = 'Y'
if label not in data_orig.columns:
    print(f"Error: Label column '{label}' not found. Assuming the last column ({data_orig.columns[-1]}) is the label.")
    label = data_orig.columns[-1]
    
# 原始数据（用于最终推理）
# 确定需要从特征集中排除的列
columns_to_drop = [col for col in non_feature_cols + [label] if col in data_orig.columns]
# 确保只删除实际存在的列
X_orig = data_orig.drop(columns=columns_to_drop) 
y_orig = data_orig[label]
print(f"Features (X) used for training: {X_orig.columns.tolist()}")



# --- 2. 按 test 和 y 列分组采样，严格按用户要求构造训练集和测试集 ---
if 'is_test' not in data_orig.columns:
    raise ValueError("数据中缺少 'is_test' 列，请确保有 is_test 列区分训练/测试。")

# 测试集
test_landslide = data_orig[(data_orig['is_test'] == 1) & (data_orig[label] == 1)]
# test_nonlandslide_pool = data_orig[(data_orig['is_test'] == 1) & (data_orig[label] == 0)]
# test_nonlandslide = test_nonlandslide_pool.sample(n=len(test_landslide), random_state=42, replace=len(test_nonlandslide_pool) < len(test_landslide))
test_nonlandslide = data_orig[(data_orig['is_test'] == 1) & (data_orig[label] == 0)]
test_df = pd.concat([test_landslide, test_nonlandslide], axis=0).sample(frac=1, random_state=123).reset_index(drop=True)

# 训练集
train_landslide = data_orig[(data_orig['is_test'] == 0) & (data_orig[label] == 1)]
# train_nonlandslide_pool = data_orig[(data_orig['is_test'] == 0) & (data_orig[label] == 0)]
# train_nonlandslide = train_nonlandslide_pool.sample(n=len(train_landslide), random_state=43, replace=len(train_nonlandslide_pool) < len(train_landslide))
train_nonlandslide = data_orig[(data_orig['is_test'] == 0) & (data_orig[label] == 0)]
train_df = pd.concat([train_landslide, train_nonlandslide], axis=0).sample(frac=1, random_state=124).reset_index(drop=True)

print(f"\n训练集样本数: {len(train_df)}")
print(f"测试集样本数: {len(test_df)}")
print(f"训练集类别分布: {train_df[label].value_counts().to_dict() if not train_df.empty else {}}")
print(f"测试集类别分布: {test_df[label].value_counts().to_dict() if not test_df.empty else {}}")


# 分为X, Y，且FID绝不参与训练
feature_cols = [col for col in train_df.columns if col not in non_feature_cols + [label, 'FID']]
X_train = train_df[feature_cols]
Y_train = train_df[label]
X_test = test_df[feature_cols]
Y_test = test_df[label]

# 转换数据类型，避免torch dtype推断错误
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# Initialize the classifier
clf = TabPFNClassifier(ignore_pretraining_limits=True)




# Prepare the Excel writer to store all results
with pd.ExcelWriter(os.path.join(OUT_DIR, 'preRes_ys.xlsx'), engine='xlsxwriter') as writer:

    X_resampled, y_resampled = X_train, Y_train
    
    # Train the classifier on the resampled training data
    clf.fit(X_resampled, y_resampled)

    # Predict on test set
    predictions = clf.predict(X_test)
    prediction_probabilities = clf.predict_proba(X_test)
    

    # 保存测试集预测概率、真实标签、FID
    df_test_out = pd.DataFrame({
        "FID": X_test.index if 'FID' not in test_df.columns else test_df['FID'].values,
        "Pred_Prob": prediction_probabilities[:, 1],
        "Actual_Label": Y_test.values,
        "Predicted_Label": predictions
    })
    df_test_out.to_excel(os.path.join(OUT_DIR, "test_preRes_ys.xlsx"), index=False)
    print(f"测试集预测结果已保存到 {os.path.join(OUT_DIR, 'test_preRes_ys.xlsx')}")

    # Calculate metrics
    accuracy = accuracy_score(Y_test, predictions)
    try:
        roc_auc = roc_auc_score(Y_test, prediction_probabilities, multi_class="ovr")
    except ValueError:
        roc_auc = roc_auc_score(Y_test, prediction_probabilities[:, 1])  # For binary classification

    # Create a DataFrame to hold the actual and predicted values
    results = pd.DataFrame({
        'Actual': Y_test,
        'Predicted': predictions
    })

    # Add metrics to the results DataFrame
    results['Accuracy'] = accuracy
    results['ROC AUC'] = roc_auc
    print(f"测试集 Accuracy: {accuracy}")
    print(f"测试集 ROC AUC: {roc_auc}")

    # Write the results to the Excel file
    results.to_excel(writer, sheet_name=f'Results', index=False)
    
    # Also add classification report for deeper analysis (optional)
    class_report = classification_report(Y_test, predictions, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_excel(writer, sheet_name=f'Classification_Report')

    print(class_report)
    
# # 对所有原始数据做预测
# X_all = X_orig.astype("float32")
# all_prob = clf.predict_proba(X_all)
# all_pred = clf.predict(X_all)

# # 计算整体 accuracy 和 AUC
# acc_all = accuracy_score(y_orig, all_pred)
# auc_all = roc_auc_score(y_orig, all_prob[:, 1])

# print(f"全量数据 Accuracy: {acc_all}")
# print(f"全量数据 ROC AUC: {auc_all}")

# # 保存 FID、预测概率、真实标签与预测标签到表格
# df_out = pd.DataFrame({
#     "FID": FID.values,
#     "Pred_Prob": all_prob[:, 1],
#     "Actual_Label": y_orig.values,
#     "Predicted_Label": all_pred
# })
# df_out.to_excel(os.path.join(OUT_DIR, "all_data_predictions_ys.xlsx"), index=False)
# print(f"全量数据预测结果已保存到 {os.path.join(OUT_DIR, 'all_data_predictions_ys.xlsx')}")

# # 计算并绘制全量数据ROC曲线
# if len(np.unique(y_orig)) > 1:
#     fpr, tpr, thresholds = roc_curve(y_orig, all_prob[:, 1])
#     auc_score = auc(fpr, tpr)
#     plt.figure(figsize=(6, 6))
#     plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
#     plt.plot([0, 1], [0, 1], 'k--', label='Random')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve (All Data)')
#     plt.legend(loc='lower right')
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUT_DIR, 'roc_curve_ys.png'), dpi=300)
#     plt.close()
#     print(f"全量数据ROC曲线已保存到 {os.path.join(OUT_DIR, 'roc_curve_ys.png')}")
# else:
#     print("全量数据只有一个类别，无法绘制ROC曲线。")
