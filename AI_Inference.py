#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv("labeled_data.csv")

# Split features and labels
X = data.drop(columns=["label"])
y = data["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Make predictions on the training and testing data
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate accuracy
# Calculate training and testing accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate other classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv("simulation_data.csv")

data.drop(columns=["car_id", "lane"], inplace=True)


# Label encode the "label" column
label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["label"])

# Split features and labels
X = data.drop(columns=["label"])
y = data["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate other classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv("simulation_data.csv")

# Drop unnecessary columns
data.drop(columns=["car_id", "lane"], inplace=True)

# Label encode the "label" column
label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["label"])

# Split features and labels
X = data.drop(columns=["label"])
y = data["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate training and testing accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("simulation_data.csv")

# Drop unnecessary columns
data.drop(columns=["car_id", "lane"], inplace=True)

# Label encode the "label" column
label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["label"])

# Split features and labels
X = data.drop(columns=["label"])
y = data["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

# Train and evaluate each classifier
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    results[name] = {"Train Accuracy": train_accuracy, "Test Accuracy": test_accuracy}

# Create DataFrame from results dictionary
results_df = pd.DataFrame(results).T

# Display results
print("Results:")
print(results_df)


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate synthetic traffic simulation data
# (Replace this section with your data generation code)
# ...

# Load the generated data (replace "traffic_simulation_data.csv" with your file)
data = pd.read_csv("labeled_data.csv")

# Prepare features and labels
X = data.drop(columns=["label"])
y = data["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate other classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualizations
# Distribution plots
sns.displot(data, x="current_velocity", hue="label", kind="kde", fill=True)
plt.title("Distribution of Velocity by Class")
plt.show()

sns.displot(data, x="following_distance", hue="label", kind="kde", fill=True)
plt.title("Distribution of Following Distance by Class")
plt.show()

sns.displot(data, x="ahead_distance", hue="label", kind="kde", fill=True)
plt.title("Distribution of Ahead Distance by Class")
plt.show()

# Scatter plot
sns.scatterplot(data=data, x="current_velocity", y="following_distance", hue="label")
plt.title("Velocity vs. Following Distance")
plt.show()

# Confusion matrix heatmap
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Feature importance plot
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind="bar")
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the generated data (replace "traffic_simulation_data.csv" with your file)
data = pd.read_csv("labeled_data.csv")

# Calculate correlation matrix
correlation_matrix = data.corr()

# Plot correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()




# In[ ]:




