from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


dir_path = os.path.dirname(os.path.realpath(__file__))

data = pd.read_csv(dir_path+"/dataset1/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")




# Select the desired columns
selected_columns = [
    " Destination Port",
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    " Fwd Packet Length Max",
    " Fwd Packet Length Min",
    " Fwd Packet Length Mean",
    " Fwd Packet Length Std",
    " Min Packet Length",
    " Max Packet Length",
    " SYN Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " ACK Flag Count",
    " Label"
]

# Create a new DataFrame with only the selected columns
selected_data = data[selected_columns]

print ("---------------Data head----------------------")
print (data.head())
print ("----------------------------------------------")

print ("---------------Data Info----------------------")
print (data.info())
print ("----------------------------------------------")

# Plot different types of visualizations for each column
'''
for column in selected_data.columns:
    plt.figure(figsize=(12, 8))
    
    # Histogram
    plt.subplot(2, 2, 1)
    plt.hist(selected_data[column], bins=20, color='skyblue', edgecolor='black')
    plt.title("Histogram")
    
    # Box Plot
    plt.subplot(2, 2, 2)
    sns.boxplot(x=selected_data[column], color='lightcoral')
    plt.title("Box Plot")
    
    # KDE Plot
    plt.subplot(2, 2, 3)
    sns.kdeplot(data=selected_data[column], color='orange')
    plt.title("Kernel Density Estimation (KDE) Plot")
    
    # Scatter Plot
    plt.subplot(2, 2, 4)
    plt.scatter(selected_data[column], selected_data[' ACK Flag Count'], alpha=0.5, color='green')
    plt.xlabel(column)
    plt.ylabel(' ACK Flag Count')
    plt.title("Scatter Plot")
    
    
    plt.tight_layout()
    plt.show()
'''


# Create a subset DataFrame with the selected columns
subset_data = data[selected_columns]

# Calculate the correlation matrix
correlation_matrix = subset_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Generate a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)

# Add a title to the plot
plt.title("Correlation Matrix of Selected Columns")

# Display the plot
#plt.show()

# Calculate and display the number of unique values in each column
unique_counts = subset_data.nunique()
print(unique_counts)

#---------------------------------------------------------------------------------------

# Convert the "Label" column to numerical values (e.g., 0 for normal (BENIGN), 1 for malicious (DDOS) )
selected_data[" Label"] = selected_data[" Label"]

# Split the data into features (X) and labels (y)
X = selected_data.drop(" Label", axis=1)
y = selected_data[" Label"]

# Split the data into training and testing sets for the ML model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest classifier --------------------------------------------

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train) #build model

# Make predictions on the test set
y_pred = clf.predict(X_test_scaled)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Total Correct Prediction ",accuracy*len(y_test))
print("Classification Report:\n", classification_rep)
conf_matrix = confusion_matrix(y_test, y_pred)

#--------------------- 

#save model
import pickle
# Save the trained model as a pickle string.
model_pkl_file = (dir_path+"/randomforest_model.pickle")  

with open(model_pkl_file, 'wb') as fid:  
    pickle.dump(clf, fid)
#--------------------------------------------

'''
class_names = ["BENIGN", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris", "Heartbleed"]
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

# Plot the confusion matrix as a heatmap
#plt.figure(figsize=(8, 6))
#sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["BENIGN", "DDoS"], yticklabels=["BENIGN", "DDoS"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
'''
