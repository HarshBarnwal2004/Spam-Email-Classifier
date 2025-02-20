# Spam Email Classifier 🚀
# 🔍 Overview
Spam Email Classifier is a machine learning-based application that detects and classifies emails as spam or not spam. Using Natural Language Processing (NLP) techniques, the model analyzes the content of an email and predicts whether it is a legitimate message or spam.

# 🛠️ Technologies Used
Python 🐍
Scikit-Learn (Machine Learning)
NLTK / SpaCy (Natural Language Processing)
Pandas & NumPy (Data Processing)
Pickle (Model Serialization)
Flask / Streamlit (For deployment, if applicable)
Git & GitHub (Version Control)

# 📌 Features
✅ Preprocesses email text (removes stopwords, tokenization, vectorization)
✅ Trained using multiple classifiers (Random Forest, Decision Tree, Naïve Bayes, etc.)
✅ Predicts spam vs. non-spam emails with high accuracy
✅ Serialized model using Pickle for easy deployment

# 📂 Project Structure
📁 Mail Classifier
│── 📄 app.py             # Main script to test the model  
│── 📄 train.py           # Model training script  
│── 📄 preprocess.py      # Text preprocessing module  
│── 📄 df.pkl             # Saved trained model  
│── 📄 cv.pkl             # CountVectorizer model  
│── 📄 dataset.csv        # Dataset used for training  
│── 📄 README.md          # Project documentation  

# 📊 Machine Learning Approach
1️⃣ Data Collection: Used a dataset containing spam and non-spam emails.
2️⃣ Text Preprocessing: Removed stopwords, tokenized, and converted text to numerical format using TF-IDF / CountVectorizer.
3️⃣ Model Training: Trained multiple classifiers (Random Forest, Decision Tree, Naïve Bayes) to identify the best-performing model.
4️⃣ Model Evaluation: Evaluated using accuracy, precision, recall, and F1-score.
5️⃣ Deployment: Saved the trained model using Pickle for quick predictions.

# 🚀 How to Run the Project
git clone https://github.com/HarshBarnwal2004/Spam-Email-Classifier.git
cd Spam-Email-Classifier
pip install -r requirements.txt
python app.py
