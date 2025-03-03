# Offensive-Language-Identification-and-Word-Embedding-Analysis-using-NLP
Project Summary:
This project focuses on analyzing offensive language in social media text data using Natural Language Processing (NLP) techniques. The dataset, OLID (Offensive Language Identification Dataset), consists of cleaned tweets labeled as offensive (OFF) or not offensive (NOT). The project leverages topic modeling, word embeddings, and machine learning to uncover patterns in offensive language.


Key Components of the Project:
1. Data Preprocessing
Load the dataset (OLID.csv) and ensure data integrity.
Tokenize tweets using NLTK (word_tokenize) and convert text to lowercase.
Remove missing values and handle necessary data cleaning steps.

2. Topic Modeling with LDA
Apply Latent Dirichlet Allocation (LDA) to uncover hidden topics in tweets.
Identify the most frequently occurring words within each topic.
Helps in understanding key themes present in offensive vs. non-offensive tweets.

3. Word Embedding Training (Word2Vec)
Train a Word2Vec model to learn word representations based on tweet context.
Uses Gensim's Word2Vec algorithm with adjustable parameters (vector_size, window, etc.).
Provides a way to visualize semantic relationships between words.

4. Dimensionality Reduction & Visualization
Use PCA (Principal Component Analysis) to reduce the word embedding dimensions.
Generate a 2D plot of words to visualize their relationships.
Helps in understanding how offensive words are clustered.

5. Insights & Future Improvements
Explore more models: Experiment with BERT, FastText, or TF-IDF for text representation.
Improve preprocessing: Implement stopword removal, stemming, and lemmatization.
Enhance visualization: Use t-SNE instead of PCA for better cluster representation.
Build a classifier: Train a machine learning model (XGBoost, SVM, or deep learning) to classify tweets as offensive or not.
