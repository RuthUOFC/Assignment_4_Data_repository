#The Gathering of Data
# Our story begins by summoning the necessary tools and the dataset itself.
import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd

# We load the 'Breast Cancer Wisconsin' dataset, our protagonist for the first half of the tale.
data = load_breast_cancer()

# --- STEP 1: Constructing the Feature Matrix and Target Vector ---
# X represents the "Features" (the 30 numeric measurements of the cells).
# y represents the "Target" (the diagnosis: Malignant or Benign).
X = data.data
y = data.target

# --- STEP 2: Reporting the Shape of our World ---
# We need to know how many samples (rows) and features (columns) we are working with.
print(f"Shape of Feature Matrix X: {X.shape}") # Should be (569, 30)
print(f"Shape of Target Vector y: {y.shape}")   # Should be (569,)

# --- STEP 3: Counting the Classes ---
# Let's see how many samples fall into each category.
# In this dataset: 0 = Malignant, 1 = Benign
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(data.target_names, counts))

print(f"Number of samples per class: {class_counts}")

'''
--- DISCUSSION & ANALYSIS ---

1. Is the dataset balanced or imbalanced?
Looking at the counts (212 Malignant vs. 357 Benign), the dataset is "moderately imbalanced." 
While it's not a severe split (like 1% vs 99%), there are significantly more benign 
cases than malignant ones.

2. Why is class balance important for classification models?
Class balance is a crucial "plot point" because if a model sees mostly one class 
during training, it might become "lazy" and simply learn to predict the majority 
class every time to get a high accuracy score. This is dangerous in medical 
scenarios! We want the model to be equally good at identifying the rarer, 
more critical class (Malignant) as it is at identifying the common one (Benign).
'''