| Concept           | Description                                              | Example                               |
| ----------------- | -------------------------------------------------------- | ------------------------------------- |
| **AI**            | Broad field simulating human intelligence                | Netflix Recommendations               |
| **ML**            | Subset of AI focused on data-driven prediction           | Predicting house prices               |
| **DL**            | ML using neural networks for complex pattern recognition | Image classification using CNN        |
| **DS**            | Extracting insights from data using stats + ML           | Analyzing customer churn using ML     |
| **Supervised**    | Learning from labeled data                               | Spam detection, salary prediction     |
| **Unsupervised**  | Finding hidden patterns in unlabeled data                | Customer clustering, PCA for features |
| **Reinforcement** | Learning through trial-and-error using rewards           | Game AI, Self-driving cars            |
| Technique               | Type                      | Output Type               | Key Strengths                                   | Limitation                          |
| ----------------------- | ------------------------- | ------------------------- | ----------------------------------------------- | ----------------------------------- |
| **Linear Regression**   | Regression                | Continuous                | Simple, interpretable                           | Sensitive to outliers               |
| **Ridge/Lasso**         | Regression                | Continuous                | Controls overfitting, feature selection (Lasso) | Needs regularization tuning         |
| **Logistic Regression** | Classification            | Categorical               | Probabilistic interpretation, fast training     | Limited to linear decision boundary |
| **Decision Tree**       | Classification/Regression | Categorical or Continuous | Interpretability, handles mixed types           | Overfitting without pruning         |
| Model               | Library                  | Key Function              | Concept Learned                            |
| ------------------- | ------------------------ | ------------------------- | ------------------------------------------ |
| Linear Regression   | `LinearRegression`       | `.fit()`, `.predict()`    | Fits a straight line to data               |
| Ridge & Lasso       | `Ridge`, `Lasso`         | `.fit()` + Regularization | Controls overfitting                       |
| Logistic Regression | `LogisticRegression`     | Binary classifier         | Probabilistic output via sigmoid           |
| Decision Tree       | `DecisionTreeClassifier` | Splits data into regions  | Rule-based classification using thresholds |
Summary

| No. | Algorithm                                | Type           | Core Idea                                                                                      | Key Use Cases                           |
| --- | ---------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------- |
| 1️⃣ | **Linear Regression**                    | Regression     | Models linear relationship between input and continuous output                                 | Predicting house prices, salaries       |
| 2️⃣ | **Logistic Regression**                  | Classification | Models probability of a class using a logistic (sigmoid) function                              | Binary classification (spam, churn)     |
| 3️⃣ | **Decision Tree**                        | Both           | Splits data based on feature thresholds recursively using rules                                | Credit approval, risk modeling          |
| 4️⃣ | **AdaBoost (Adaptive Boosting)**         | Both           | Combines weak learners (e.g., trees) iteratively, giving more weight to misclassified examples | Face detection, fraud detection         |
| 5️⃣ | **Logistic Regression** *(again listed)* | Classification | Already described above                                                                        |                                         |
| 6️⃣ | **Random Forest**                        | Both           | Ensemble of decision trees using bagging; reduces overfitting                                  | Diagnosis, recommendation systems       |
| 7️⃣ | **Gradient Boosting**                    | Both           | Sequential tree boosting that reduces errors by correcting previous learners                   | Ranking, search engines, forecasting    |
| 8️⃣ | **XGBoost**                              | Both           | Optimized version of Gradient Boosting with regularization and parallelism                     | Kaggle competitions, large-scale ML     |
| 9️⃣ | **Naive Bayes**                          | Classification | Probabilistic model using Bayes’ theorem assuming feature independence                         | Text classification, sentiment analysis |
| Algorithm           | Best For              | Output Type    | Key Feature                       |
| ------------------- | --------------------- | -------------- | --------------------------------- |
| Linear Regression   | Continuous prediction | Regression     | Line of best fit                  |
| Logistic Regression | Binary classification | Classification | Sigmoid function for probability  |
| Decision Tree       | Simple rules          | Both           | If-else splitting                 |
| AdaBoost            | Improving weak models | Both           | Boosts weak classifiers           |
| Random Forest       | Accuracy + stability  | Both           | Ensemble of decision trees        |
| Gradient Boosting   | High accuracy         | Both           | Sequential error correction       |
| XGBoost             | Speed + accuracy      | Both           | Optimized GBM                     |
| Naive Bayes         | Text, fast inference  | Classification | Probabilistic, simple assumptions |

Algorithm Quick Reference:
1. Linear Regression
Output: Continuous



Evaluation: RMSE, MAE, R²

2. Logistic Regression
Output: Probability for class membership

Binary: Sigmoid function → 0/1

Multi-class: Softmax

Evaluation: Accuracy, F1-score, ROC-AUC

3. Decision Tree
Output: Class or regression value

Splitting Criteria: Gini impurity or entropy

Evaluation: Accuracy for classification, RMSE for regression

4. AdaBoost
Boosts simple models sequentially

Gives higher weight to misclassified samples

Works well with shallow decision trees (stumps)

5. Random Forest
Bagging ensemble of many decision trees

Averages or majority votes predictions

Robust to overfitting and noise

6. Gradient Boosting
Trees built sequentially

Each model corrects the residuals of the previous

Slower than Random Forest but often more accurate

7. XGBoost
Advanced Gradient Boosting

Regularization + Parallelization

Highly optimized for speed and accuracy

8. Naive Bayes
Based on Bayes Theorem

Assumes independence among features

Fast and works well with text data
