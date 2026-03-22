# =============================================================================
# CHAPTER 4: NEURAL NETWORK ARCHITECTURE FOR BINARY CLASSIFICATION
# =============================================================================
# In this phase, we transition from the rule-based logic of Decision Trees
# to a sophisticated Multi-Layer Perceptron (MLP) architecture.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- PRELUDE: DATA ACQUISITION AND INITIALIZATION ---
breast_cancer_dataset = load_breast_cancer()
feature_matrix = breast_cancer_dataset.data
target_vector = breast_cancer_dataset.target

# --- STEP 1: STRATIFIED DATA PARTITIONING ---
# We partition the data into 80% training and 20% testing subsets.
# Stratification ensures the class distribution remains consistent across sets.
(features_train, features_test, target_train, target_test) = train_test_split(
    feature_matrix,
    target_vector,
    test_size=0.20,
    stratify=target_vector,
    random_state=42
)

# --- STEP 2: FEATURE STANDARDIZATION (SCALING) ---
# Neural networks require features to be on a comparable scale to ensure
# efficient gradient descent optimization.
feature_scaler = StandardScaler()
features_train_standardized = feature_scaler.fit_transform(features_train)
features_test_standardized = feature_scaler.transform(features_test)

# --- STEP 3: NEURAL NETWORK CONFIGURATION ---
# We initialize a Multi-Layer Perceptron with one hidden layer of 100 neurons.
# The 'logistic' activation function provides the required sigmoid output.
neural_network_model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='logistic',   # Sigmoid activation function
    solver='adam',           # Stochastic gradient-based optimizer
    max_iter=1000,           # Maximum number of iterations for convergence
    random_state=42
)

# --- STEP 4: MODEL OPTIMIZATION (TRAINING) ---
# The model iteratively adjusts its internal weights to minimize classification error.
neural_network_model.fit(features_train_standardized, target_train)

# --- STEP 5: QUANTITATIVE PERFORMANCE EVALUATION ---
training_predictions = neural_network_model.predict(features_train_standardized)
testing_predictions = neural_network_model.predict(features_test_standardized)

training_accuracy_score = accuracy_score(target_train, training_predictions)
testing_accuracy_score = accuracy_score(target_test, testing_predictions)

print(f"--- Neural Network Performance Analysis ---")
print(f"Training Accuracy Score: {training_accuracy_score:.4f}")
print(f"Testing Accuracy Score:  {testing_accuracy_score:.4f}")

# =============================================================================
# ANALYTICAL DISCUSSION (Q4 RESPONSES)
# =============================================================================
'''
1. Why is feature scaling necessary for neural networks?
Feature scaling is fundamental because neural networks utilize gradient-based 
optimization algorithms. If input features possess disparate numerical 
ranges, the gradient updates can become biased toward features with larger 
magnitudes, leading to slow convergence or suboptimal local minima. 
Standardization ensures that each feature contributes equally to the 
weight adjustment process, facilitating a more stable and efficient 
learning trajectory.

2. What does an epoch represent during neural network training?
An epoch is defined as one complete iteration through the entire training 
dataset. During a single epoch, every training sample is passed through 
the network (forward propagation), the loss is calculated, and the 
internal weights are updated (backward propagation). Multiple epochs 
are typically required for the network to refine its parameters and 
accurately map the underlying patterns within the data.
'''