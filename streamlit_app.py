import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

# Download NLTK stop words
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Initialize stop words
stop_words = set(stopwords.words('english'))

# Function to generate word cloud
def generate_wordcloud(data, number_of_words, color_scheme, text_case, additional_stop_words):
    # Tokenization
    tokens = word_tokenize(data)
    
    # Remove stop words and apply text case
    if text_case == 'Upper case':
        tokens = [word.upper() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    else:
        tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    
    # Additional stop words
    if additional_stop_words:
        additional_stop_words_list = additional_stop_words.split(',')
        tokens = [word for word in tokens if word.lower() not in additional_stop_words_list]
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, max_words=number_of_words, 
                          color_func=lambda *args, **kwargs: color_scheme, 
                          background_color='white').generate(' '.join(tokens))
    
    # Display the generated word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Function to convert fig to PNG and display in Streamlit
def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    st.image(buf)

# Streamlit app
def main():
    st.title('Word Cloud Generator')

    # Text input
    text_input = st.text_area("Paste text here...")

    # File upload
    file = st.file_uploader("Or upload your data (.csv or .txt)", type=['csv', 'txt'])
    if file:
        if file.type == "text/csv":
            df = pd.read_csv(file)
            text_input = ' '.join(df[df.columns[0]])
        elif file.type == "text/plain":
            text_input = file.getvalue().decode("utf-8")

    # Sidebar options
    number_of_words = st.sidebar.slider('Number of words', 5, 100, 50, 5)
    text_color = st.sidebar.selectbox('Text color', ('Black', 'Colorful'))
    text_case = st.sidebar.selectbox('Text case', ('Upper case', 'Lower case'))
    additional_stop_words = st.sidebar.text_input('Additional stop words', '')

    # Generate word cloud
    if st.button('Generate Word Cloud'):
        if text_color == 'Colorful':
            color_scheme = plt.cm.inferno
        else:
            color_scheme = 'black'

        generate_wordcloud(text_input, number_of_words, color_scheme, text_case, additional_stop_words)

    # Download PNG button
    if st.button('Download PNG'):
        wordcloud = generate_wordcloud(text_input, number_of_words, color_scheme, text_case, additional_stop_words)
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        fig_to_png(fig)

if __name__ == "__main__":
    main()
