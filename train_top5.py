"""
Breast Cancer Wisconsin veri setinden en onemli 5 ozelligi belirleyip
yeni bir PyTorch modeli egiten script.
"""

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# 1) Veriyi yukle ve en onemli 5 ozelligi bul
# ------------------------------------------------------------------
data = load_breast_cancer()
X_full, y = data.data, data.target
feature_names = data.feature_names

# RandomForest ile feature importance hesapla
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_full, y)
importances = rf.feature_importances_

# En onemli 5 ozelligin indekslerini al
top5_indices = np.argsort(importances)[::-1][:5]
top5_names = feature_names[top5_indices]

print("En onemli 5 ozellik:")
for i, (idx, name) in enumerate(zip(top5_indices, top5_names), 1):
    print(f"  {i}. {name} (index={idx}, importance={importances[idx]:.4f})")

# Sadece bu 5 ozelligi al
X = X_full[:, top5_indices]

# ------------------------------------------------------------------
# 2) Train/test split ve scaling
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 3) PyTorch model tanimla ve egit
# ------------------------------------------------------------------
class BreastCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model = BreastCancerModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

EPOCHS = 300
for epoch in range(1, EPOCHS + 1):
    model.train()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test_t)
            preds = torch.argmax(test_logits, dim=1)
            acc = (preds == y_test_t).float().mean().item()
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")

# ------------------------------------------------------------------
# 4) Sonuclari kaydet
# ------------------------------------------------------------------
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

torch.save(model.state_dict(), models_dir / "breast_cancer_model_top5.pth")
joblib.dump(scaler, models_dir / "cancer_scaler_top5.pkl")
joblib.dump(top5_indices.tolist(), models_dir / "top5_indices.pkl")
joblib.dump(top5_names.tolist(), models_dir / "top5_names.pkl")

print("\nModel, scaler ve ozellik bilgileri kaydedildi.")
print(f"  Model  -> {models_dir / 'breast_cancer_model_top5.pth'}")
print(f"  Scaler -> {models_dir / 'cancer_scaler_top5.pkl'}")
print(f"  Indices-> {models_dir / 'top5_indices.pkl'}")
print(f"  Names  -> {models_dir / 'top5_names.pkl'}")
