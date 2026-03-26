import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

df = pd.read_csv("Concrete_Data_Yeh.csv")

df.columns = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
    "Concrete compressive strength"
]



print("\n各列缺失值数量：")
print(df.isnull().sum())

# 如果存在缺失值，则用该列均值填充
if df.isnull().sum().sum() > 0:
    print("\n检测到缺失值，开始用各列均值填充")
    df = df.fillna(df.mean(numeric_only=True))
else:
    print("\n未检测到缺失值，无需填充。")

print(df.isnull().sum())


# 相关性分析并筛选主特征
target_name = "Concrete compressive strength"
all_features = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age"
]

corr_matrix = df.corr(numeric_only=True)
target_corr = corr_matrix[target_name].drop(target_name)
target_corr_abs = target_corr.abs().sort_values(ascending=False)

print("\n各输入特征与目标值的相关系数：")
print(target_corr.sort_values(key=lambda x: x.abs(), ascending=False))

threshold = 0.20
selected_features = target_corr_abs[target_corr_abs >= threshold].index.tolist()

# 如果要用全部特征，下面一行取消注释
# selected_features = all_features

print(f"\n相关性筛选阈值 = {threshold}")
print("最终保留的主特征：")
print(selected_features)

# 保存特征筛选结果
feature_selection_result = pd.DataFrame({
    "Feature": target_corr.index,
    "Correlation": target_corr.values,
    "Absolute Correlation": target_corr.abs().values,
    "Selected": [feature in selected_features for feature in target_corr.index]
}).sort_values(by="Absolute Correlation", ascending=False)

feature_selection_result.to_csv(
    "feature_selection_by_correlation.csv",
    index=False,
    encoding="utf-8-sig"
)

# 保存相关性图
plt.figure(figsize=(9, 5))
target_corr.sort_values(ascending=False).plot(kind="bar")
plt.title("Correlation with Concrete Compressive Strength")
plt.ylabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig("target_correlation_bar.jpg", dpi=300, format="jpg")
plt.close()

plt.figure(figsize=(9, 5))
target_corr_abs.sort_values(ascending=False).plot(kind="bar")
plt.axhline(y=threshold, linestyle="--")
plt.title("Absolute Correlation with Concrete Compressive Strength")
plt.ylabel("Absolute Correlation Coefficient")
plt.tight_layout()
plt.savefig("target_absolute_correlation_bar.jpg", dpi=300, format="jpg")
plt.close()

# 提取输入和输出
X = df[selected_features].copy()
y = df[target_name].copy()

print("\n筛选后的输入特征前5行：")
print(X.head())


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=selected_features)

train_output = X_train_scaled_df.copy()
train_output[target_name] = y_train.reset_index(drop=True)
train_output.to_csv("train_data_selected_features.csv", index=False, encoding="utf-8-sig")

test_output = X_test_scaled_df.copy()
test_output[target_name] = y_test.reset_index(drop=True)
test_output.to_csv("test_data_selected_features.csv", index=False, encoding="utf-8-sig")

print("\n训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)


class ConcreteDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets.values.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ConcreteDataset(X_train_scaled, y_train)
test_dataset = ConcreteDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

input_dim = len(selected_features)
model = MLPRegressor(input_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epochs = 300
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")


plt.figure(figsize=(8, 5)) # 保存训练损失曲线
plt.plot(range(1, num_epochs + 1), train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss (MSE)")
plt.title("Training Loss Curve")
plt.tight_layout()
plt.savefig("training_loss_curve.jpg", dpi=300, format="jpg")
plt.close()


# 在测试集上预测
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        preds = model(batch_X)
        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(batch_y.cpu().numpy().flatten())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# 计算评估指标
mse = mean_squared_error(all_targets, all_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)

print("\n测试集评估结果：")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"R^2  = {r2:.4f}")

metrics_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAE", "R2"],
    "Value": [mse, rmse, mae, r2]
})
metrics_df.to_csv("mlp_regression_metrics.csv", index=False, encoding="utf-8-sig")


plt.figure(figsize=(6, 6))
plt.scatter(all_targets, all_preds, alpha=0.7)
min_val = min(all_targets.min(), all_preds.min())
max_val = max(all_targets.max(), all_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("True Strength")
plt.ylabel("Predicted Strength")
plt.title("True vs Predicted")
plt.tight_layout()
plt.savefig("true_vs_predicted.jpg", dpi=300, format="jpg")
plt.close()

# 保存预测结果
prediction_df = pd.DataFrame({
    "True Strength": all_targets,
    "Predicted Strength": all_preds,
    "Error": all_preds - all_targets
})
prediction_df.to_csv("prediction_results.csv", index=False, encoding="utf-8-sig")

# print("\n文件已保存到当前路径：")
# print("1. feature_selection_by_correlation.csv")
# print("2. target_correlation_bar.jpg")
# print("3. target_absolute_correlation_bar.jpg")
# print("4. train_data_selected_features.csv")
# print("5. test_data_selected_features.csv")
# print("6. training_loss_curve.jpg")
# print("7. true_vs_predicted.jpg")
# print("8. mlp_regression_metrics.csv")
# print("9. prediction_results.csv")