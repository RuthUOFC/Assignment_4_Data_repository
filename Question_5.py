# =============================================================================
# CHAPTER 5: MODEL EVALUATION AND COMPARATIVE ANALYSIS
# =============================================================================
# In this final chapter of the tabular data study, we utilize Confusion Matrices
# to perform a rigorous comparison between our two classification architectures.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- PRELUDE: DATA RE-INITIALIZATION ---
breast_cancer_dataset = load_breast_cancer()
feature_matrix = breast_cancer_dataset.data
target_vector = breast_cancer_dataset.target
target_class_names = breast_cancer_dataset.target_names

# --- STEP 1: CONSISTENT DATA PARTITIONING ---
(features_train, features_test,
 target_train, target_test) = train_test_split(
    feature_matrix,
    target_vector,
    test_size=0.20,
    stratify=target_vector,
    random_state=42
)

# --- STEP 2: RE-ESTABLISHING THE CONSTRAINED DECISION TREE ---
constrained_decision_tree_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    random_state =42
)
constrained_decision_tree_model.fit(features_train, target_train)

# --- STEP 3: RE-ESTABLISHING THE NEURAL NETWORK ---
feature_scaler = StandardScaler()
features_train_standardized = feature_scaler.fit_transform(features_train)
features_test_standardized = feature_scaler.transform(features_test)

neural_network_model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='logistic',
    max_iter =1000,
    random_state =42
)
neural_network_model.fit(features_train_standardized, target_train)

# --- STEP 4: GENERATING CONFUSION MATRICES ---
# We generate predictions for both models on the testing dataset.
decision_tree_predictions = constrained_decision_tree_model.predict(features_test)
neural_network_predictions = neural_network_model.predict(features_test_standardized)

# Compute the Confusion Matrices
decision_tree_confusion_matrix = confusion_matrix(target_test, decision_tree_predictions)
neural_network_confusion_matrix = confusion_matrix(target_test, neural_network_predictions)

# --- STEP 5: VISUALIZING THE RESULTS ---
print("--- Constrained Decision Tree Confusion Matrix ---")
print(decision_tree_confusion_matrix)

print("\n--- Neural Network Confusion Matrix ---")
print(neural_network_confusion_matrix)

# Optional: Visualization for clarity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(decision_tree_confusion_matrix, display_labels=target_class_names).plot(ax=ax1)
ax1.set_title("Decision Tree Confusion Matrix")
ConfusionMatrixDisplay(neural_network_confusion_matrix, display_labels = target_class_names).plot(ax=ax2)
ax2.set_title("Neural Network Confusion Matrix")
plt.show()

# =============================================================================
# ANALYTICAL DISCUSSION (Q5 COMPARISON)
# =============================================================================
'''
1. Which model would you prefer for this task?
For a medical diagnostic task, the Neural Network often achieves higher 
accuracy and lower false-negative rates. However, the Decision Tree provides 
unparalleled interpretability. If the goal is high-precision screening, the 
NEURAL NETWORK is preferred. If the priority is medical transparency and 
clinical verification, the DECISION TREE is the superior choice.

2. One advantage and one limitation of each model:

CONSTRAINED DECISION TREE:
- Advantage: INTERPRETABILITY. The model's logic is easily understood 
  by human experts through its rule-based structure.
- Limitation: RIGIDITY. It may struggle to capture highly non-linear, 
  complex relationships compared to neural architectures.

NEURAL NETWORK:
- Advantage: FLEXIBILITY. It is capable of learning extremely complex, 
  high-dimensional patterns with high predictive power.
- Limitation: "BLACK BOX" NATURE. It is difficult to explain exactly 
  why the model made a specific prediction, which can be a risk in 
  sensitive medical applications.
'''