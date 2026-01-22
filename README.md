### Heart Disease Prediction Using Logistic Regression

## 📌 Project Overview

- This project demonstrates the complete Machine Learning pipeline to predict the presence of heart disease using Logistic Regression. The model is built using Scikit-learn and evaluated using standard classification metrics such as accuracy, precision, recall, and confusion matrix.

## Tools Used

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

## Dataset Description

The dataset contains patient health-related attributes:

- Feature	Description
- Age	 of the patient
- Sex	Gender (1 = Male, 0 = Female)
- Chol	Serum cholesterol level
- Target	Output (1 = Heart Disease, 0 = No Heart Disease)
  
## Step-by-Step Implementation

- 1: Load Dataset
- The heart disease dataset is loaded into a Pandas DataFrame for analysis and preprocessing.

- 2: Split Dataset into Training and Testing Sets
- The dataset is divided using train_test_split():
- Training Data → 70%
- Testing Data → 30%
- This prevents overfitting and ensures fair evaluation.

- 3: Train Logistic Regression Model
- A Logistic Regression classifier is trained using Scikit-learn.
- Suitable for binary classification problems
- Efficient for medical prediction tasks
- The max_iter parameter is increased to ensure model convergence.

- 4: Predict on Test Data
- The trained model is used to make predictions on test data.
- To evaluate model accuracy

- 5: Model Evaluation Metrics
- The following performance metrics are calculated:
- Accuracy
- Precision
- Recall

- 6: Confusion Matrix
- A confusion matrix is generated to visualize prediction results:
- Actual / Predicted	No Disease	Disease
- True Negative	Correct No Disease Prediction	
- False Positive	Incorrect Disease Prediction	
- False Negative	Missed Disease Case	
- True Positive	Correct Disease Prediction	

- 7: Save Trained Model
- The trained Logistic Regression model is saved.

- 8: Generate Evaluation Report
- Model performance metrics and confusion matrix results are saved.

## Conclusion

- This project demonstrates the implementation of a Heart Disease Prediction system using Logistic Regression. The trained model achieves reliable performance and can be reused for future predictions. This project highlights the importance of data splitting, proper evaluation, and model persistence in real-world Machine Learning applications.
