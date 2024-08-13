import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data.csv")

# Map categorical data to numerical data
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

# Define features and target variable
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']

# Initialize and train the decision tree classifier
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# Plot the decision tree
fig = plt.figure(figsize=(15, 20))
_ = tree.plot_tree(dtree, feature_names=features, filled=True)

# Display the first few rows of the dataframe
df.head()