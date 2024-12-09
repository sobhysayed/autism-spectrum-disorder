import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r'autism_data.csv')

# # print(data.head())


# print(data.shape)

# print(data.isna().sum())

# # print(data.duplicated().sum())

# print(df.info())

# print(data.duplicated().sum())


data['age'] = pd.to_numeric(data['age'].str.replace("'", ""), errors='coerce')
data['contry_of_res'] = data['contry_of_res'].str.replace("'", "")

data['age'].fillna(data['age'].median(), inplace=True)

categorical_cols = ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res',
                    'used_app_before', 'age_desc', 'relation', 'Class/ASD']
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    data[col] = label_encoders[col].fit_transform(data[col])

# Define features (X) and target (y)
X = data.drop(columns=['Class/ASD'])
y = data['Class/ASD']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data for better performance
# scaler=LogisticRegression()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Train a RandomForest classifier
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)


# # Predict on the test set and calculate accuracy
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print(f'Accuracy: {accuracy:.2f}')

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Feature scaling
scaler = StandardScaler()

# Fitting the scaler with the training data
scaler.fit(X_train)

# Transforming the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Building the SVM model
model = svm.SVC(kernel='linear')

# Training the model with the training data
model.fit(X_train, Y_train)

# Model Evaluation
# Predicting on the test data
y_pred = model.predict(X_test)

# Calculating the accuracy score
accuracy = accuracy_score(Y_test, y_pred)
print(f"\nAccuracy of the model: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(Y_test, y_pred))


# k = 5
# cross_val_scores = cross_val_score(model, X_train, Y_train, cv=k)

# # Calculate the mean and standard deviation of the cross-validation scores
# cv_mean = cross_val_scores.mean()
# cv_std = cross_val_scores.std()

# # Generate a classification report for detailed metrics
# classification_report_text = classification_report(
#     y_test, y_pred, target_names=label_encoders['Class/ASD'].classes_
# )

# print(f"\nMean Cross-Validation Accuracy (k={k}): {cv_mean * 100:.2f}%")
# print(f"Standard Deviation: {cv_std:.4f}")
# print(f"Classification Report:\n{classification_report_text}")


def plot_learning_curve(estimator, X, y, cv, train_sizes, title):
    """
    Plots the learning curve for a given estimator.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy', n_jobs=-1
    )

    # Calculate mean and standard deviation for training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.grid()

    # Plot training score
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training Score")

    # Plot validation score
    plt.fill_between(
        train_sizes,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, val_scores_mean, "o-", color="g", label="Validation Score")

    plt.legend(loc="best")
    plt.show()


# Plot the learning curve
plot_learning_curve(
    model,
    X,
    y,
    cv=5,  # 5-fold cross-validation
    train_sizes=np.linspace(0.1, 1.0, 10),  # Use 10 points from 10% to 100% of training data
    title="Learning Curve for Random Forest Classifier", )
