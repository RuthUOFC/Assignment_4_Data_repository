# =============================================================================
# CHAPTER 3: CONTROLLING COMPLEXITY AND INTERPRETABILITY
# =============================================================================
# In this chapter, we impose constraints on our model to prevent overfitting
# and identify the most influential characters (features) in our story.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd # Useful for displaying feature importance

# --- PRELUDE: RE-ESTABLISHING THE CONTEXT ---
data = load_breast_cancer()
X = data.data
y = data.target
feature_names= data.feature_names

# --- STEP 1: THE STRATIFIED SPLIT (Consistent with Q2) ---
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, stratify=y, random_state=42)

# --- STEP 2: INTRODUCING CONSTRAINTS (The "Disciplined" Model) ---
# We use 'max_depth=3' to prevent the tree from growing too deep and complex.
# This forces the model to find the most significant patterns quickly.
dt_constrained = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,          # Constraint: Limit the number of questions the tree can ask
    min_samples_split=10, # Constraint: Minimum samples needed to split a node
    random_state=42
)

# --- STEP 3: TRAINING THE CONSTRAINED MODEL ---
dt_constrained.fit(X_train, y_train)

# --- STEP 4: EVALUATING PERFORMANCE ---
train_acc_c = accuracy_score(y_train, dt_constrained.predict(X_train))
test_acc_c = accuracy_score(y_test, dt_constrained.predict(X_test))

print(f"--- Constrained Model Performance ---")
print(f"Training Accuracy: {train_acc_c:.4f}")
print(f"Test Accuracy:     {test_acc_c:.4f}")

# --- STEP 5: IDENTIFYING THE LEAD CHARACTERS (Feature Importance) ---
# We must first extract the importance values into their own variable...
importances = dt_constrained.feature_importances_

# ...THEN we create the DataFrame to display them nicely!
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 5 Most Important Features ---")
print(feature_importance_df.head(5))
# =============================================================================
# ANALYTICAL DISCUSSION (Q3 ANSWERS)
# =============================================================================
'''
1. How does controlling model complexity affect overfitting?
By introducing constraints like '$max_depth$', we prevent the tree from 
creating "branches" for every minor variation or outlier in the training 
data. This reduces the gap between Training and Test accuracy. While 
Training Accuracy might drop slightly, the model becomes more robust 
and better at generalizing to new, unseen data (reducing overfitting).

2. How does feature importance contribute to interpretability?
Feature importance acts as a "narrative guide," telling us exactly 
which biological measurements (like 'worst perimeter' or 'worst area') 
the model relies on to make a diagnosis. This transparency makes 
Decision Trees "White Box" models, allowing medical professionals 
to trust and verify the logic behind the AI's decision.
'''