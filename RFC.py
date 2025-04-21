## Random Forest Classifier ##

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load data
df = pd.read_csv('test_data/accepted_loan.csv')

# Create target: 1 for 'Charged Off', 0 for 'Fully Paid' or 'Current'
df['default'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

# Features
features = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'annual_inc',
    'dti', 'revol_bal', 'revol_util', 'total_acc', 'open_acc', 'fico_range_low',
    'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'pub_rec',
    'pub_rec_bankruptcies', 'emp_length', 'home_ownership', 'tot_cur_bal',
    'total_rev_hi_lim', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_util',
    'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mort_acc', 'num_actv_rev_tl',
    'pct_tl_nvr_dlq'
]

# Subset data
data = df[features + ['default']].copy()

# Preprocess
# Encode categorical variables
data['term'] = data['term'].apply(lambda x: 0 if '36' in x else 1)
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
data['grade'] = data['grade'].map(grade_map)
data['sub_grade'] = data['sub_grade'].apply(lambda x: grade_map[x[0]] + int(x[1]) / 10)
emp_length_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
                  '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
data['emp_length'] = data['emp_length'].map(emp_length_map)
data = pd.get_dummies(data, columns=['home_ownership'], drop_first=True)

# Handle missing values
data['mths_since_last_delinq'] = data['mths_since_last_delinq'].fillna(999)
for col in data.columns:
    if data[col].dtype in ['float64', 'int64'] and col != 'default':
        data[col] = data[col].fillna(data[col].median())

# Split features and target
X = data.drop('default', axis=1)
y = data['default']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize one decision tree
plt.figure(figsize=(20, 10))
plot_tree(rf.estimators_[0], feature_names=X.columns, class_names=['Non-Default', 'Default'], filled=True, rounded=True)
plt.title('Decision Tree from Random Forest (Tree 1)')
plt.show()

# Visualize decision boundary with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
rf_pca = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
rf_pca.fit(X_pca, y)
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = rf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.3)
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='blue', label='Non-Default', alpha=0.6)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='red', label='Default', alpha=0.6)
plt.title('Random Forest Decision Boundary')
plt.legend()
plt.show()