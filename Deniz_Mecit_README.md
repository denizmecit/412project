Author: Deniz Mecit
Date: 12 January 2025

#Project Overview

I have developed a regression workflow on Instagram post/profile data with the goal of predicting a numeric engagement metric (such as popularity) on a transformed (log) scale. This solution integrates numerical and textual features:
	•	BERT embeddings of textual data (biography, category name, entities, captions).
	•	XGBoost for the regression model.
	•	A test set that produces predictions in JSON format, mapping post IDs to predicted engagement scores.

Data Description
	1.	Training Data
	•	I use the file training-dataset.jsonl.gz, which has multiple lines of JSON records.
	•	Each record includes:
	•	A profile object (username, biography, category_name, entities, follower_count, post_count, etc.).
	•	A posts list, each with like_count, comments_count, etc.
	•	I parse these into numeric summaries (like aggregated sums) and text columns for embedding.
	2.	Test Data
	•	The file, for example test-regression-round3.jsonl, provides new post data with caption, comments_count, and an id.
	•	I process this file similarly (using placeholders for absent features) to produce a final prediction JSON.

Methodology
	1.	Data Loading & Preprocessing
	•	I implement a function to read .jsonl (or gzipped) lines into a list of dicts.
	•	I convert that list into a Pandas DataFrame (df). Numeric fields are aggregated or summed as needed; text fields remain for embedding.
	2.	Feature Engineering
	•	Text Embeddings: I utilize a Turkish BERT model (dbmdz/bert-base-turkish-cased) to encode text fields. I extract the [CLS] token from the model’s output as the embedding.
	•	Numeric Transformations: Because certain features (like follower_count, post_count, comments_count) may be highly skewed, I apply log transforms.
	•	I concatenate the numeric vector with the BERT embedding to form the final feature set.
	3.	Modeling
	•	I use XGBoost to handle the regression on these final features:
	•	The numeric columns (often log-transformed).
	•	The BERT embeddings for text.
	•	The target variable is the log-transformed like-count sum, e.g., log10(sum_like_count + 1) to reduce skew.
	•	I perform a standard train/validation split, train the model, and measure error on the validation subset.
	4.	Evaluation
	•	I measure Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) in log space.
	•	I also compute the R² score in log space to see how much variation is explained by the model.
	•	For interpretability, I invert the log predictions via 10^(prediction) - 1.
	5.	Prediction & JSON Output
	•	For testing, I load a file like test-regression-round3.jsonl containing posts for which I want predictions.
	•	I embed each post’s text with BERT, combine it with placeholders for numeric features, and run it through the trained XGBoost model.
	•	I invert the log transform and produce a JSON object: { "post_id": predicted_value }, saving it to a .json file.

Code Flow
	1.	Cells [2] & [3]
	•	Install or import necessary libraries (Transformers, XGBoost).
	•	Define file paths and reading utilities.
	•	Load and parse the training data (training-dataset.jsonl.gz).
	2.	Cell [4]
	•	Construct a DataFrame from the JSON: numeric and text columns.
	3.	Cell [5]
	•	Load dbmdz/bert-base-turkish-cased model/tokenizer.
	•	Define a function returning the [CLS] vector from BERT.
	4.	Cell [6]
	•	Generate embeddings for each row’s text, combine them into a NumPy array.
	5.	Cell [7]
	•	Apply log-transforms to numeric columns.
	•	Prepare the final training input by concatenating numeric arrays with the BERT embeddings.
	•	Train the XGBoost regressor on these features.
	6.	Cell [8]
	•	Calculate MSE, RMSE, and R² in log space. Show a sample of predicted vs. actual (in both log and inverted scales).
	7.	Cell [10]
	•	Load the test .jsonl file.
	•	Embed the new data with BERT, set placeholders for missing fields.
	•	Predict, invert the transform, and save the results in a JSON structure of { post_id : predicted_like_count }.

Key Takeaways
	•	BERT provides high-quality semantic embeddings for textual data, improving performance.
	•	Log transformations help manage large numeric ranges in social media metrics.
	•	XGBoost handles combined tabular and embedding features effectively.
	•	This pipeline is flexible enough for additional features or advanced tuning.

Possible Extensions
	•	Hyperparameter optimization (e.g., learning_rate, max_depth, etc.).
	•	More refined text preprocessing, potentially domain-specific expansions.
	•	Additional numeric or temporal features (posting frequency, average engagement rates, etc.).
	•	An ensemble approach for even higher accuracy.


