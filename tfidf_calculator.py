import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Download the stopwords from NLTK
nltk.download('punkt')
nltk.download('stopwords')

def read_pdf(filepath):
    # Open and read the PDF
    pdf_content = ""
    with open(filepath, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        num_pages = pdf_reader.numPages
        for page in range(num_pages):
            page_content = pdf_reader.getPage(page).extractText()
            pdf_content += page_content
    return pdf_content

def calculate_tfidf(pdf_content):
    # Calculate TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([pdf_content]) # Note: 'fit_transform' expects a list of documents

    # We can convert this matrix to a Pandas dataframe to make it easier to work with
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Return the dataframe as a string
    return df.to_string()
