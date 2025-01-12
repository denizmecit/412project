## CS412 Project - Elif (28808)

### Classification Models
I worked on implementing and evaluating classical machine learning models for the influencer classification task. My main contributions include:

#### Models Implemented
- Naive Bayes Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

#### Key Features
- Implemented text preprocessing for Turkish language
- Used NLTK for stopword removal
- Feature engineering based on post captions and metadata
- Cross-validation for model evaluation
- Hyperparameter tuning for Random Forest

#### Performance Metrics
- Achieved accuracy of XX% with Random Forest
- Implemented weighted F1 score calculation
- Evaluated models using classification reports

#### Code Structure
- Data preprocessing functions
- Model training and evaluation
- Prediction pipeline for test data
- JSON output formatting for submissions

1. Data Loading and Preprocessing
```python
train_classification_df = pd.read_csv("train-classification.csv")
train_classification_df = train_classification_df.rename(columns={
'Unnamed: 0': 'user_id',
'label': 'category'
})
```

2. Feature Processing
```python
nltk.download('stopwords')
turkish_stopwords = stopwords.words('turkish')
```

3. Model Evaluation
```python
def log_mse_like_counts(y_true, y_pred):
y_true = np.array(y_true)
y_pred = np.array(y_pred)
log_y_true = np.log1p(y_true)
log_y_pred = np.log1p(y_pred)
return np.mean((log_y_true - log_y_pred) 2)
```

### Project Files
- `project_explore_Ali_copy.ipynb`: Main notebook containing the classification models
- Turkish stopwords implementation
- Feature extraction functions
- Model evaluation metrics

### Results
- Achieved consistent performance across different categories
- Successfully handled Turkish language content
- Implemented robust evaluation metrics
- Created scalable prediction pipeline
