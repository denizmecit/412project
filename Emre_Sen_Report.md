Emre Şen 31227

### 1. **Importing Libraries**
   I started by importing the libraries I needed:
   - `pandas` and `numpy` for working with the data.
   - `scikit-learn` for splitting the data, preprocessing, and evaluating the models.
   - `nltk` for text preprocessing, especially for handling stopwords.

### 2. **Installing Packages**
   Since I was working in Google Colab, I had to install `nltk` using `!pip install` to make sure everything worked smoothly.

### 3. **Text Preprocessing**
   I downloaded Turkish stopwords from `nltk` and created a list of them (`turkish_stopwords`) to help clean the text data by removing common words that don’t add much meaning (like "ve," "bir," etc.).

### 4. **Loading the Data**
   I connected my Google Drive to access the datasets. I used:
   - `train-classification.csv`: A CSV file containing `user_id` and `category` (label).
   - `training-dataset.jsonl.gz`: A compressed file with user behavior data that I used to create the features.

### 5. **Cleaning and Organizing the Data**
   I renamed some columns to make the dataset easier to work with (e.g., changing `Unnamed: 0` to `user_id`).
   I also converted all category labels to lowercase so there wouldn't be any inconsistencies when comparing them.
   I then created a dictionary that mapped each `user_id` to its corresponding `category`.

### 6. **Feature Engineering**
   I used `TfidfVectorizer` to convert the text data into numerical form so that the machine learning models could work with it. TF-IDF helps give more weight to important words and less to common ones.
   I made sure to remove Turkish stopwords during this process to improve the overall feature quality.

### 7. **Model Training**
   I split the dataset into training and testing sets using `train_test_split`.
   After that, I trained the models and ran them to see how well they could classify the data.

### 8. **Evaluation**
   I used `accuracy_score` and `classification_report` to check how well my models performed.
   The `classification_report` gave me metrics like precision, recall, and F1-score, which helped me understand how well the model performed for each category.

---
