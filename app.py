import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

## nltk.download('stopwords')
## stopwords.words('english')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower() ## Lower case
    text = nltk.word_tokenize(text)  ## Tokenization
    
    y = []
    for i in text:     ## Removing special characters
        if i.isalnum(): 
            y.append(i)
        
    text = y[:] 
    y.clear()
    
    for i in text:    ## Removing stop words and punctuation
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:    ## Stemming
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
 
    # STEPS:
    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

