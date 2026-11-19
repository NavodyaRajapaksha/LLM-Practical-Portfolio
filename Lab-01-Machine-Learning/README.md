
# Lab 01 – Sentiment Analysis Using Traditional Machine Learning

## 1. Introduction

Sentiment analysis is a Natural Language Processing (NLP) technique used to identify the emotional tone or opinion expressed in textual data. It is widely used in applications such as customer feedback analysis, product review classification and social media monitoring.

This laboratory session focuses on implementing a sentiment classification model using traditional machine learning techniques. The implementation involves text preprocessing, exploratory data analysis, Bag of Words feature extraction, Logistic Regression model training and performance evaluation.

The model performs binary sentiment classification by categorizing textual reviews into positive and negative sentiments.

## 2. Objectives

The main objective of this laboratory session is to understand and practically implement fundamental Natural Language Processing techniques using traditional machine learning.

The specific objectives are:

- Understand the fundamentals of Natural Language Processing.
- Load and explore a labelled sentiment analysis dataset.
- Apply basic text cleaning and preprocessing techniques.
- Identify frequently occurring words in positive and negative reviews.
- Convert textual data into numerical features using Bag of Words (BoW).
- Train a machine learning model for sentiment classification.
- Evaluate model performance using appropriate classification metrics.
- Predict the sentiment of previously unseen sentences.
- Understand the limitations of traditional NLP approaches.

## 3. Tools and Technologies

| Tool / Library | Purpose |
|---|---|
| Python | Programming language used for implementation |
| Jupyter Notebook | Code development and experimentation |
| Pandas | Data loading, manipulation and analysis |
| Matplotlib | Data visualization |
| Regular Expressions (re) | Text cleaning and preprocessing |
| Collections (Counter) | Word frequency analysis |
| Scikit-learn | Feature extraction, model training and evaluation |

## 4. Dataset Description

The dataset used in this laboratory session is the Sentiment Labelled Sentences Dataset obtained from the UCI Machine Learning Repository.

**Dataset Source:**  
[Sentiment Labelled Sentences Dataset](https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences)

The dataset contains textual reviews collected from three different sources:

- Amazon – Product reviews
- IMDb – Movie reviews
- Yelp – Restaurant reviews

Each sentence is associated with a binary sentiment label:

- **0 – Negative sentiment**
- **1 – Positive sentiment**

The combined dataset loaded during the implementation contains 2,748 reviews.

### Dataset Distribution

| Sentiment | Label | Number of Reviews | Percentage |
|---|---:|---:|---:|
| Positive | 1 | 1,386 | 50.44% |
| Negative | 0 | 1,362 | 49.56% |
| Total | - | 2,748 | 100% |

The loaded dataset is approximately balanced, with a similar number of positive and negative reviews.

### Dataset Attributes

| Column | Description |
|---|---|
| sentence | Original review text |
| label | Binary sentiment label |
| source | Original source of the review |
| clean_text | Preprocessed review text created during implementation |


## 5. Tasks and Implementation

### Task 01 – Download and Load the Dataset

The Sentiment Labelled Sentences Dataset was downloaded from the UCI Machine Learning Repository.

The Amazon, IMDb and Yelp text files were loaded separately using the Pandas library. Each dataset was assigned a source identifier before combining all three datasets into a single DataFrame.

The dataset shape, initial records and sentiment label distribution were examined to understand the structure and composition of the data.

The combined dataset contained 2,748 records and three initial columns.

**Relevant functions:**

- `pd.read_csv()` – Load the individual dataset files.
- `pd.concat()` – Combine the datasets.
- `df.head()` – Display the initial records.
- `value_counts()` – Identify sentiment class distribution.

**Output:**

| Description | Result |
|---|---|
| Dataset shape | (2748, 3) |
| Positive reviews | 1,386 |
| Negative reviews | 1,362 |

The approximately balanced class distribution provides a suitable starting point for binary sentiment classification.

### Task 02 – Basic Text Preprocessing

Text preprocessing was performed to transform raw textual data into a more consistent format before feature extraction.

A custom text-cleaning function was implemented using Python regular expressions.

The following preprocessing operations were performed:

1. Convert all text into lowercase.
2. Remove punctuation and unnecessary special characters.
3. Remove extra whitespace.
4. Store the processed sentences in a new column named `clean_text`.
5. Compare the original and preprocessed sentences.

**Text preprocessing function:**

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["sentence"].apply(clean_text)
```

Lowercasing reduces inconsistencies caused by capitalization, while punctuation removal and whitespace normalization help standardize the textual data.

#### Original and Preprocessed Text Comparison

| Original Text | Preprocessed Text |
|---|---|
| Good case, Excellent value. | good case excellent value |
| The mic is great. | the mic is great |
| Great for the jawbone. | great for the jawbone |
| Needless to say, I wasted my money. | needless to say i wasted my money |
| What a waste of money and time!. | what a waste of money and time |

A random sample of ten records was also examined to compare the original sentences, cleaned sentences and sentiment labels.

This process demonstrated how raw text can be transformed into a consistent representation suitable for further NLP processing.

### Task 03 – Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to understand the characteristics of positive and negative reviews before training the machine learning model.

The following activities were performed:

- Display positive review examples.
- Display negative review examples.
- Visualize sentiment class distribution using a bar chart.
- Identify frequently occurring words in each sentiment category.
- Observe differences between positive and negative reviews.

#### Positive Review Examples

- Good case, Excellent value.
- Great for the jawbone.
- The mic is great.
- Very good quality though.

#### Negative Review Examples

- Needless to say, I wasted my money.
- What a waste of money and time!.
- I advise EVERYONE DO NOT BE FOOLED!

#### Sentiment Distribution

A bar chart was generated using Matplotlib to visualize the number of positive and negative reviews.

The visualization demonstrated that the dataset contains approximately equal numbers of positive and negative samples.

<!-- Insert the sentiment distribution screenshot here. -->

#### Word Frequency Analysis

The Python `Counter` class was used to identify the 15 most frequently occurring words in each sentiment category.

```python
from collections import Counter

def most_common_words(text_series, n=15):
    words = " ".join(text_series).split()
    return Counter(words).most_common(n)
```

Selected results obtained from the word frequency analysis are shown below.

| Positive Reviews – Word | Frequency | Negative Reviews – Word | Frequency |
|---|---:|---|---:|
| the | 960 | the | 993 |
| and | 665 | i | 562 |
| i | 475 | and | 473 |
| a | 449 | a | 440 |
| great | 193 | not | 252 |
| good | 173 | t | 209 |

Positive reviews frequently contain words such as "great" and "good", whereas negative reviews frequently contain negation words such as "not".

Common words such as "the", "and" and "i" appear frequently in both categories because stop-word removal was not applied during the initial word frequency analysis.

These observations demonstrate that word occurrence patterns can provide useful information for sentiment classification.

### Task 04 – Split the Dataset

The preprocessed dataset was divided into training and testing subsets using the `train_test_split()` function from Scikit-learn.

The cleaned text was selected as the input feature, while the sentiment label was selected as the target variable.

An 80:20 splitting ratio was used.

```python
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

| Dataset | Percentage | Number of Samples |
|---|---:|---:|
| Training dataset | Approximately 80% | 2,198 |
| Testing dataset | Approximately 20% | 550 |
| Total | 100% | 2,748 |

The training dataset was used to train the machine learning model, while the testing dataset was used to evaluate its performance on unseen data.

The `random_state=42` parameter was used to ensure reproducibility, while `stratify=y` was used to maintain approximately equivalent sentiment class proportions across both datasets.

### Task 05 – Create Bag of Words Representation

Machine learning algorithms require numerical input rather than raw textual data. Therefore, the Bag of Words (BoW) technique was used to convert the preprocessed sentences into numerical feature vectors.

Bag of Words represents a document using the frequency of words appearing in that document. It creates a vocabulary of unique terms and represents each sentence according to the occurrence of those terms.

The `CountVectorizer` class from Scikit-learn was used to implement Bag of Words.

```python
vectorizer = CountVectorizer(
    stop_words="english",
    min_df=2
)

X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)
```

#### Vectorizer Configuration

| Parameter | Description |
|---|---|
| stop_words="english" | Removes predefined English stop words |
| min_df=2 | Retains terms appearing in at least two training documents |
| fit_transform() | Learns the vocabulary and transforms training data |
| transform() | Transforms testing data using the existing vocabulary |

#### Feature Extraction Results

| Feature Matrix | Dimensions |
|---|---|
| Training Bag of Words | (2198, 1643) |
| Testing Bag of Words | (550, 1643) |
| Vocabulary size | 1,643 |

The training and testing datasets were successfully converted into numerical representations containing 1,643 features.

The vectorizer was fitted only on the training dataset to prevent vocabulary information from the testing dataset influencing the training process.

Although Bag of Words is a simple and efficient text representation technique, it does not preserve word order, contextual relationships or semantic meaning.

### Task 06 – Train a Machine Learning Model

Logistic Regression was selected as the machine learning algorithm for sentiment classification.

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification problems. It estimates the probability of an input belonging to a particular class based on its numerical features.

The model was trained using the Bag of Words representation of the training dataset.

```python
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_bow, y_train)
```

#### Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Maximum iterations | 1,000 |
| Random state | 42 |
| Feature representation | Bag of Words |
| Number of output classes | 2 |

After training, predictions were generated for the testing dataset.

```python
y_pred = model.predict(X_test_bow)
```

Logistic Regression provides a computationally efficient baseline for traditional text classification.

### Task 07 – Evaluate Model Performance

The trained Logistic Regression model was evaluated using the testing dataset.

The following classification metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

Accuracy measures the overall proportion of correctly classified reviews.

Precision measures the proportion of predicted positive reviews that are actually positive.

Recall measures the proportion of actual positive reviews correctly identified by the model.

The F1-score represents the harmonic mean of precision and recall, providing a balanced measurement of classification performance.

The confusion matrix was generated to identify correct predictions and classification errors for both sentiment categories.

### Task 08 – Test the Model with New Sentences

The trained model was tested using new sentences that were not included in the training dataset.

Each new sentence was cleaned using the same preprocessing function and transformed into a numerical representation using the previously fitted Bag of Words vectorizer.

The transformed sentence was then passed to the trained Logistic Regression model to predict its sentiment.

```python
new_sentence = "The product is excellent and I really enjoyed using it."

new_clean = clean_text(new_sentence)
new_bow = vectorizer.transform([new_clean])

prediction = model.predict(new_bow)[0]

sentiment = "Positive" if prediction == 1 else "Negative"
```

Two sentences expressing different sentiments were tested to observe the model's prediction capability.

## 6. Results and Evidence

### 6.1 Model Evaluation Results

The following results were obtained after evaluating the Logistic Regression model on 550 testing samples.

| Evaluation Metric | Score | Percentage |
|---|---:|---:|
| Accuracy | 0.787273 | 78.73% |
| Precision | 0.807692 | 80.77% |
| Recall | 0.758123 | 75.81% |
| F1-score | 0.782123 | 78.21% |

Precision, recall and F1-score in the above table are calculated for the positive sentiment class.

The model achieved an overall accuracy of 78.73%, demonstrating that traditional machine learning techniques can identify sentiment patterns from textual data with reasonable effectiveness.

Out of 550 testing samples, 433 reviews were classified correctly, while 117 reviews were misclassified.

### 6.2 Classification Report

| Sentiment | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.77 | 0.82 | 0.79 | 273 |
| Positive | 0.81 | 0.76 | 0.78 | 277 |
| Macro Average | 0.79 | 0.79 | 0.79 | 550 |
| Weighted Average | 0.79 | 0.79 | 0.79 | 550 |

The classification report indicates that the model achieved similar F1-scores for positive and negative reviews.

Negative reviews achieved a higher recall, indicating that the model correctly identified a greater proportion of actual negative reviews.

Positive reviews achieved higher precision, indicating that a greater proportion of reviews predicted as positive were correctly classified.

### 6.3 Confusion Matrix

The confusion matrix provides a detailed representation of the model's correct and incorrect classifications.

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 223 | 50 |
| Positive | 67 | 210 |

| Classification | Number of Samples |
|---|---:|
| True Negative (TN) | 223 |
| True Positive (TP) | 210 |
| False Positive (FP) | 50 |
| False Negative (FN) | 67 |

The confusion matrix demonstrates that the model correctly classified 223 negative reviews and 210 positive reviews.

However, 50 negative reviews were incorrectly classified as positive, while 67 positive reviews were incorrectly classified as negative.

<!-- Insert the confusion matrix screenshot here. -->

### 6.4 Sentiment Prediction Results

The trained model generated the following predictions for previously unseen sentences.

| Input Sentence | Expected Sentiment | Predicted Sentiment |
|---|---|---|
| The product is excellent and I really enjoyed using it. | Positive | Positive |
| This product is terrible and I hate it. | Negative | Negative |

Both sentences were classified according to their intended sentiment.

These results provide examples of the model's ability to classify unseen sentences. However, additional testing with more diverse and complex sentences would be necessary to establish its generalization capability.

<!-- Insert the sentiment prediction output screenshot here. -->

## 7. Observations and Discussion

Several important observations can be identified from the implementation and experimental results.

### Dataset Distribution

The dataset contains approximately equal numbers of positive and negative reviews. This reduces the risk of the model becoming strongly biased towards a majority sentiment class.

### Effect of Text Preprocessing

Text preprocessing transformed inconsistent raw sentences into a standardized format. Lowercasing and punctuation removal helped reduce variations in textual features.

However, the preprocessing method may also remove information that contributes to sentiment, such as punctuation used for emphasis.

### Word Frequency Patterns

Positive reviews frequently contain words such as "great" and "good", while negative reviews frequently contain negation words such as "not".

Common words appear in both sentiment categories, demonstrating that word frequency alone may not always provide sufficient information for accurate sentiment classification.

### Model Performance

The Logistic Regression model achieved an accuracy of 78.73%, providing a reasonable baseline for sentiment analysis using traditional machine learning.

The similar F1-scores obtained for both sentiment categories indicate that the model performs relatively consistently across the two classes.

However, the 117 incorrectly classified reviews demonstrate that the model has limitations when identifying more complex sentiment patterns.

### Bag of Words Limitations

Bag of Words provides a straightforward method for converting textual data into numerical features. Nevertheless, it ignores word order and contextual relationships.

For example, sentences containing "good" and "not good" may be represented using similar features, particularly when negation words are removed during stop-word filtering.

This can negatively affect sentiment classification.

## 8. Challenges and Limitations

The implemented sentiment analysis approach has several technical limitations.

- Bag of Words does not capture contextual or semantic relationships between words.
- English stop-word removal may eliminate important negation words required for sentiment interpretation.
- The preprocessing function separates contractions, which may reduce the clarity of their original meaning.
- Previously unseen words may not contribute to predictions because they are absent from the training vocabulary.
- Traditional machine learning models may struggle to identify sarcasm, implicit sentiment and ambiguous expressions.
- The implementation supports only positive and negative sentiment classifications and does not include neutral sentiment.
- The model was evaluated using a single training and testing split, which may not fully represent its performance across different data subsets.
- Only two additional sentences were used for manual prediction testing.

The loaded dataset contained 2,748 records. Further investigation of the original dataset files and loading procedure would be useful to verify whether any records were omitted during data loading.

These limitations highlight the importance of developing more advanced NLP techniques capable of capturing contextual information.

## 9. Learning Outcomes

This laboratory session provided practical exposure to fundamental NLP and machine learning concepts.

The key learning outcomes include:

- Understanding the role of sentiment analysis in Natural Language Processing.
- Loading, combining and exploring datasets using Pandas.
- Implementing text preprocessing using regular expressions.
- Performing exploratory data analysis on textual data.
- Identifying word frequency patterns across sentiment categories.
- Understanding how textual data can be represented numerically.
- Implementing Bag of Words using CountVectorizer.
- Preparing training and testing datasets for supervised machine learning.
- Training a Logistic Regression model for binary classification.
- Evaluating classification performance using accuracy, precision, recall and F1-score.
- Interpreting classification reports and confusion matrices.
- Performing sentiment prediction on previously unseen sentences.
- Understanding the limitations of traditional machine learning approaches to NLP.

The laboratory session establishes foundational knowledge for understanding more advanced NLP approaches, including word embeddings, neural networks, transformers and Large Language Models.

## 10. Execution Instructions

Install the required Python libraries using the following command:

```bash
pip install pandas matplotlib scikit-learn notebook
```

Download the Sentiment Labelled Sentences Dataset from the UCI Machine Learning Repository.

Update the dataset file paths in the Jupyter Notebook according to the local dataset location.

Open the uploaded Jupyter Notebook and execute all cells sequentially to reproduce the implementation and evaluation results.


