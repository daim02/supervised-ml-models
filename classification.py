# Import all libraries like pandas, numpy, matplotlib, seaborn and sci-kit learn.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the Titanic dataset.
file_path = "Titanic_Dataset_Classification.csv"
df = pd.read_csv(file_path)

# Display dataset dimensions and any missing values.
print("Dataset Dimensions:", df.shape)
print("Missing Values:\n", df.isnull().sum())

# Split data into training and testing sets - last 140 rows for testing.
train_set = df.iloc[:-140].copy()
test_set = df.iloc[-140:].copy()

# Drop any irrelevant/un-needed columns.
columns_to_exclude = ['Name', 'Ticket No.']
train_set.drop(columns=[col for col in columns_to_exclude if col in train_set.columns], inplace=True)
test_set.drop(columns=[col for col in columns_to_exclude if col in test_set.columns], inplace=True)

# Encode categorical columns as numeric.
for feature in ['Sex', 'Embarked']:
    encoder = LabelEncoder()
    train_set[feature] = encoder.fit_transform(train_set[feature].astype(str))
    test_set[feature] = encoder.transform(test_set[feature].astype(str))

# Impute missing values using median strategy.
imputer_all = SimpleImputer(strategy='median')
train_clean = pd.DataFrame(imputer_all.fit_transform(train_set), columns=train_set.columns)
test_clean = pd.DataFrame(imputer_all.transform(test_set), columns=test_set.columns)

# Split data into features (X) and target (y).
X_tr = train_clean.drop('Survival', axis=1)
y_tr = train_clean['Survival']
X_te = test_clean.drop('Survival', axis=1)
y_te = test_clean['Survival']

# Scale the features using StandardScaler.
scaler_obj = StandardScaler()
X_tr_scaled = scaler_obj.fit_transform(X_tr)
X_te_scaled = scaler_obj.transform(X_te)

# Model Training for all 3 models.

# Naive Bayes
model_nb = GaussianNB()
model_nb.fit(X_tr_scaled, y_tr)
pred_nb = model_nb.predict(X_te_scaled)

# Logistic Regression.
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=0), param_grid_lr, cv=5)
grid_lr.fit(X_tr_scaled, y_tr)
model_lr = grid_lr.best_estimator_
pred_lr = model_lr.predict(X_te_scaled)

# Support Vector Machine.
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}
grid_svm = GridSearchCV(SVC(random_state=0), param_grid_svm, cv=5)
grid_svm.fit(X_tr_scaled, y_tr)
model_svm = grid_svm.best_estimator_
pred_svm = model_svm.predict(X_te_scaled)

# Function to display evaluation results for each model
def display_results(title, y_true, y_pred):
    print(f"\n--- {title} ---")
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    return acc

# Store results for each model.
results = {}
results['Naive Bayes'] = display_results("Naive Bayes (Main Model)", y_te, pred_nb)
results['Logistic Regression'] = display_results("Logistic Regression (Baseline - Tuned)", y_te, pred_lr)
results['SVM'] = display_results("Support Vector Machine (Baseline - Tuned)", y_te, pred_svm)

# Display best hyperparameters for Logistic Regression and SVM - tuned models.
print("\nBest Logistic Regression Params:", grid_lr.best_params_)
print("Best SVM Params:", grid_svm.best_params_)

# Confusion Matrices for all 3 models.

# Model predictions dictionary/key.
model_preds = {
    "Naive Bayes": pred_nb,
    "Logistic Regression": pred_lr,
    "SVM": pred_svm
}

# Plot the confusion matrices for each model.
for model_name, preds in model_preds.items():
    cm = confusion_matrix(y_te, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.grid(False)
    plt.show()

# Metrics bar chart for all 3 models.

# Get classification report as dictionary for each model.
reports = {
    'Naive Bayes': classification_report(y_te, pred_nb, output_dict=True),
    'Logistic Regression': classification_report(y_te, pred_lr, output_dict=True),
    'SVM': classification_report(y_te, pred_svm, output_dict=True)
}

# Extract weighted average evaluation metrics.
summary_metrics = ['precision', 'recall', 'f1-score']
model_scores = []

for model_name, report in reports.items():
    for metric in summary_metrics:
        model_scores.append({
            'Model': model_name,
            'Metric': metric,
            'Score': report['weighted avg'][metric]
        })

# Convert to Dataframe.
metrics_df = pd.DataFrame(model_scores)

# Plot grouped bar chart.
plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_df, x='Model', y='Score', hue='Metric')
plt.title("Classification Metrics (Weighted Average) by Model")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xlabel("Model")
plt.grid(True, axis='y')
plt.legend(title='Metric')
plt.tight_layout()
plt.show()