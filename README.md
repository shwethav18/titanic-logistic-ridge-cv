# Logistic Regression on Titanic Dataset with Ridge Penalty

Can we accurately predict which passengers survived the Titanic disaster using demographic and travel-related data? This project investigates how the performance of a logistic regression model changes with varying levels of cross-validation and L2 regularization (Ridge) on the Titanic dataset.

## Dataset

- **Source:** Titanic Dataset (Kaggle / class-provided notebook)
- **Target Variable:** Survived (0 or 1)
- **Features:** Age, Fare, Sex, Pclass, Embarked, etc.

## Problem Statement

Can we predict passenger survival based on demographic and travel details, and how does model accuracy change with different values of K in cross-validation?

## Approach

1. **Preprocessing**
   - Imputed missing values and encoded categorical variables
   - Applied `StandardScaler` for numerical features

2. **Modeling**
   - Used `LogisticRegression` with `penalty='l2'`
   - Performed cross-validation for K = 2, 3, 4, and 5
   - Recorded classification accuracy across folds

3. **Evaluation**
   - Compared model accuracy across different K values
   - Observed impact of cross-validation size on generalization

## Results

The model showed expected variability in performance across different K values, reinforcing the importance of robust validation. Ridge regularization helped control overfitting.

## Transferable Skills Demonstrated

- Built and evaluated classification models
- Implemented cross-validation and regularization techniques
- Improved existing Jupyter notebook code into a reproducible experiment
- Analyzed model stability using performance metrics

## Future Improvements

- Tune hyperparameter `C` using `GridSearchCV`
- Add evaluation metrics such as confusion matrix, precision, and recall
- Engineer new features from text and ordinal data
- Apply workflow to larger or more complex classification problems

## Tools Used

- Python
- scikit-learn
- NumPy, Pandas
- Jupyter Notebook
