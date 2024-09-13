import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, accuracy_score
import sys

# Load and prepare the data
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

data = pd.read_csv(input_file_path)
data[['Mouse_Line', 'Group', 'Week']] = data['INDEX'].str.split('_', expand=True)
data['Week'] = data['Week'].astype(int)
data['Group_Status'] = data['Group'].apply(lambda x: 'Low_Dose' if x == 'L' else ('High_Dose' if x == 'H' else 'Other'))
data = data[data['Group_Status'].isin(['Low_Dose', 'High_Dose'])]

# Uncomment the following line to randomize class labels
# data['Group_Status'] = np.random.permutation(data['Group_Status'].values)

# Set immune parameters
immune_parameters = data.columns.drop(['INDEX', 'Mouse_Line', 'Group', 'Week', 'Group_Status'])

# Prepare the data for the RandomForest analysis
X = data[immune_parameters].values
y = data['Group_Status'].map({'Low_Dose': 0, 'High_Dose': 1}).values

SPLIT = 10
skf = StratifiedKFold(n_splits=SPLIT, shuffle=True, random_state=42)

aggregate_auc = 0
aggregate_cm = np.zeros((2, 2))
feature_importances = np.zeros(len(immune_parameters))
total_y_test = []
total_y_pred = []

# Perform the StratifiedKFold RandomForest analysis
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    feature_importances += clf.feature_importances_

    y_pred = clf.predict(X_test)
    total_y_test.extend(y_test)
    total_y_pred.extend(y_pred)

    cm = confusion_matrix(y_test, y_pred)
    aggregate_cm += cm

    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    aggregate_auc += auc

feature_importances /= SPLIT
average_auc = aggregate_auc / SPLIT

# Calculate overall accuracy and balanced accuracy
overall_accuracy = accuracy_score(total_y_test, total_y_pred)
overall_balanced_accuracy = balanced_accuracy_score(total_y_test, total_y_pred)

# Print the results
print("Aggregate Confusion Matrix:\n", aggregate_cm)
print("\nAverage AUC: {:.2f}".format(average_auc))
print("\nOverall Accuracy: {:.2f}".format(overall_accuracy))
print("\nOverall Balanced Accuracy: {:.2f}".format(overall_balanced_accuracy))

# Save the feature importances to a CSV file
feature_importance_df = pd.DataFrame({
    'Feature': immune_parameters,
    'Average_Importance': feature_importances
})
feature_importance_df.to_csv(output_file_path, index=False)
