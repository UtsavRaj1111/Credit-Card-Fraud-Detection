import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import os

# Optional: fix Joblib warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # set to your CPU cores

# 1. Load dataset
df = pd.read_csv("creditcard.csv")

# 2. Select only features for this simplified model
features = ['V1', 'V2', 'Amount']
X = df[features]
y = df['Class']

# 3. Preprocess Amount
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1,1))

# 4. Balance dataset with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print("After SMOTE, counts of label '1':", sum(y_res==1))

# 5. Use smaller subset for fast testing
X_res_small = X_res.sample(3000, random_state=42)
y_res_small = y_res.loc[X_res_small.index]

# 6. Train XGBoost
X_train, X_test, y_train, y_test = train_test_split(
    X_res_small, y_res_small, test_size=0.2, random_state=42
)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=2)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

cm = confusion_matrix(y_test, y_pred)
roc_score = roc_auc_score(y_test, y_pred_proba)

print("Confusion Matrix:\n", cm)
print("ROC AUC Score:", roc_score)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_score:.2f})')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 8. Save model and scaler for Streamlit
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully!")
