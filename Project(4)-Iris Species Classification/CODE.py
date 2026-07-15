                        #STEP - 1
                        #IMPORT THE LIBRARIES
# Import the Pandas library for working with tabular data
import pandas as pd

# Import the built-in Iris dataset from Scikit-Learn
from sklearn.datasets import load_iris

# Load the Iris dataset into a variable
iris = load_iris()

# Convert the feature data into a Pandas DataFrame
# The feature names become the column names
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add the target column (species encoded as 0, 1, or 2)
df["species"] = iris.target

# Replace numeric labels with readable species names
df["species"] = df["species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# Display the first five rows of the dataset
print(df.head())
                        #STEP - 2
                        #UNDERSTAND THE DATASET
print("Shape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nSpecies Count:")
print(df["species"].value_counts())
                            #STEP - 3
                            #EDA
import matplotlib.pyplot as plt
#HISTOGRAM
df.hist(figsize=(10, 8))

plt.tight_layout()

plt.show()
#SCATTER PLOT
plt.figure(figsize=(8,6))

colors = {
    "Setosa": "red",
    "Versicolor": "green",
    "Virginica": "blue"
}

for species in df["species"].unique():
    subset = df[df["species"] == species]

    plt.scatter(
        subset["petal length (cm)"],
        subset["petal width (cm)"],
        label=species
    )

plt.xlabel("Petal Length")

plt.ylabel("Petal Width")

plt.title("Petal Length vs Petal Width")

plt.legend()

plt.show()
#BOX PLOT
plt.figure(figsize=(8,6))

df.boxplot()

plt.title("Feature Distribution")

plt.show()
                      #STEP - 4
                      #DATA PREPROCESSING
from sklearn.model_selection import train_test_split
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Training Features:", X_train.shape)
print("Testing Features :", X_test.shape)

print("Training Labels  :", y_train.shape)
print("Testing Labels   :", y_test.shape)
                    #STEP - 5
                    #BUILD THE MACHINE LEARNING MODEL
# Import the Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier

# Create the model
model = DecisionTreeClassifier(random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

print("Model training completed successfully!")
                      #STEP - 6
                      #EVALUATE THE MODEL
# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Predict the species of the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))   
                       #STEP - 7
                       #PREDICT NEW FLOWERS
# Create a DataFrame for the new flower
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

# Predict the species
prediction = model.predict(new_flower)

print("Predicted Species:", prediction[0])         
