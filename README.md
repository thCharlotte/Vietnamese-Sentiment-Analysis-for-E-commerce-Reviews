# Vietnamese Sentiment Analysis for E-commerce Reviews

This project is a mid-term report for an NLP course, focusing on the classification of user sentiments from Vietnamese e-commerce platforms (Tiki, Lazada, Shopee, TikTok).

##  Project Objective

The goal is to collect raw review data, preprocess it, and then train, evaluate, and compare various machine learning and deep learning models to find an effective solution for Vietnamese sentiment classification, paying special attention to imbalanced data.

## Dataset

The dataset consists of user reviews collected from multiple online shopping platforms. It has been manually labeled into three categories: positive, negative, and neutral.

**Dataset Link:** [Download Dataset from Google Drive](https://drive.google.com/file/d/1FVpNV9n7avFRUdB-v288-dbY3K54KnAI/view?usp=sharing)

##  Methodology

### 1. Data Preprocessing

* Data collection from various sources.
* Raw data cleaning and normalization.
* Labeling based on sentiment (positive, negative, neutral) combined with review scores.

### 2. Feature Engineering (Text Representation)

* Bag-of-Words (BoW)
* BoW + N-grams
* TF-IDF
* TF-IDF + N-grams
* Doc2Vec

### 3. Model Training & Evaluation

We implemented and compared a wide range of models:

**Traditional Machine Learning:**

* Naive Bayes
* Logistic Regression
* Decision Tree

**Deep Learning:**

* Convolutional Neural Networks (CNN)
* Long Short-Term Memory (LSTM)

**Pre-trained Transformer Model:**

* PhoBERT

Models were evaluated based on their F1-score to effectively handle the imbalanced nature of the dataset.

##  Technologies Used

* **Language:** Python
* **Core Libraries:** Pandas, NumPy, Scikit-learn
* **Deep Learning:** TensorFlow / Keras (or PyTorch)
* **Transformers:** transformers (for PhoBERT)
* **Tools:** Jupyter Notebook, Google Colab

##  How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/thCharlotte/sentiment-feedback-classification-model.git](https://github.com/thCharlotte/sentiment-feedback-classification-model.git)
    ```
2.  Download the dataset from the link above and place it in the appropriate data folder (or as specified in the notebook).

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  Open and run the Jupyter Notebook (.ipynb) to see the preprocessing, training, and evaluation steps.
