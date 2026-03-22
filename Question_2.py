# =============================================================================
# CHAPTER 2: THE ARCHITECTURE OF DECISION - THE ENTROPY MODEL
# =============================================================================
# We must first re-introduce our data characters (X and y) to resolve the NameError.

from sklearn.datasets import load_breast_cancer # Added this!
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# --- PRELUDE: LOADING THE DATA ---
# This ensures X and y are defined before we try to split them.
data = load_breast_cancer()
X = data.data
y = data.target

# --- STEP 1: THE STRATIFIED SPLIT ---
# Now that X and y are defined, the $train_test$_split will work perfectly!
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, stratify=y, random_state=42)

# --- STEP 2: INITIALIZING THE CLASSIFIER ---
# Using 'entropy' to measure the "disorder" of our data nodes.
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)

# --- STEP 3: THE TRAINING PHASE ---
dt_entropy.fit(X_train, y_train)

# --- STEP 4: PERFORMANCE EVALUATION ---
train_preds = dt_entropy.predict(X_train)
test_preds = dt_entropy.predict(X_test)

train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print(f"--- Model Performance Report ---")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:     {test_accuracy:.4f}")

# =============================================================================
# ANALYTICAL DISCUSSION (Q2 ANSWERS)
# =============================================================================
'''
1. What does Entropy represent in the context of decision trees?
Entropy is a measure of "impurity" or "uncertainty" in a group of samples. 
The Decision Tree calculates entropy to determine how much information 
is "gained" by splitting the data on a certain feature. A perfect split 
results in zero entropy (total order).

2. Do the results suggest overfitting or good generalization?
If Training Accuracy is significantly higher than Test Accuracy (which 
usually happens with unconstrained trees), it indicates OVERFITTING. 
The model has "memorized" the training data too specifically, making 
it less effective at generalizing to the new test data.
'''