# Langchain dependencies
from langchain_core.documents import Document
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, UnstructuredExcelLoader # Importing Text loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
#from langchain.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
from langchain_openai import OpenAIEmbeddings # Importing OpenAIEmbeddings
import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from os.path import isfile


def load_documents(DATA_PATH: str):
    """
    Load text documents from the specified directory using TextLoader.
    Returns:
    List of Document objects: Loaded Text documents represented as Langchain
                                                        Document objects.
    """

    _, doc_type = os.path.splitext(DATA_PATH)

    # Initialize loaders with specified directory
    if doc_type == '.docx':    
        document_loader = Docx2txtLoader(DATA_PATH)
    elif doc_type == '.xlsx': 
        document_loader = UnstructuredExcelLoader(DATA_PATH, mode="elements")
    else:
        document_loader = TextLoader(DATA_PATH) 

    # Load documents and return them as a list of Document objects
    return document_loader#.load() 
    #return document_loader

def directory_load_documents(DATA_PATH: str):

    documents = []
    #list_of_files = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]

    list_of_files = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if isfile(os.path.join(root, file)):
                list_of_files.append(os.path.join(root, file))
    
    for file in list_of_files:
        FILE_PATH = file
        documents.append(load_documents(FILE_PATH))

    merged_documents = MergedDataLoader(documents)

    all_docs = merged_documents.load()
    
    return all_docs

def clean_text(documents: list[Document]):
    # download stop words
    nltk.download('stopwords')
    ## portuguese stopwords
    STOPWORDS = nltk.corpus.stopwords.words('portuguese')
    STOPWORDS.append(
                ['último','é','acerca','agora','algmas','alguns','ali','ambos','antes','apontar','aquela',
                'aquelas','aquele','aqueles','aqui','atrás','bem','bom','cada','caminho','cima','com','como',
                'comprido','conhecido','corrente','das','debaixo','dentro','desde','desligado','deve','devem',
                'deverá','direita','diz','dizer','dois','dos','e','ela','ele','eles','em','enquanto','então',
                'está','estão','estado','estar','estará','este','estes','esteve','estive','estivemos','estiveram',
                'eu','fará','faz','fazer','fazia','fez','fim','foi','fora','horas','iniciar','inicio','ir','irá',
                'ista','iste','isto','ligado','maioria','maiorias','mais','mas','mesmo','meu','muito','muitos','nós',
                'não','nome','nosso','novo','o','onde','os','ou','outro','para','parte','pegar','pelo','pessoas','pode',
                'poderá	podia','por','porque','povo','promeiro','quê','qual','qualquer','quando','quem','quieto','são',
                'saber','sem','ser','seu','somente','têm','tal','também','tem','tempo','tenho','tentar','tentaram','tente',
                'tentei','teu','teve','tipo','tive','todos','trabalhar','trabalho','tu','um','uma','umas','uns','usa','usar',
                'valor','veja','ver','verdade','verdadeiro','você']
            )

    for document in documents:
        
        document.page_content = document.page_content.lower()  # Convert all characters in text to lowercase
        # Example after this step: "i won't go there! this is a testing @username https://example.com <p>paragraphs!</p> #happy :)"

        document.page_content = re.sub(r'https?://\S+|www\.\S+', '', document.page_content)  # Remove URLs
        # Example after this step: "i won't go there! this is a testing @username  <p>paragraphs!</p> #happy :)"

        document.page_content = re.sub(r'<.*?>', '', document.page_content)  # Remove HTML tags
        # Example after this step: "i won't go there! this is a testing @username  paragraphs! #happy :)"

        document.page_content = re.sub(r'@\w+', '', document.page_content)  # Remove mentions
        # Example after this step: "i won't go there! this is a testing   paragraphs! #happy :)"

        document.page_content = re.sub(r'#\w+', '', document.page_content)  # Remove hashtags
        # Example after this step: "i won't go there! this is a testing   paragraphs!  :)"

        # Translate emoticons to their word equivalents
        emoticons = {':)': 'smile', ':-)': 'smile', ':(': 'sad', ':-(': 'sad'}
        words = document.page_content.split()
        words = [emoticons.get(word, word) for word in words]
        text = " ".join(words)
        # Example after this step: "i won't go there! this is a testing paragraphs! smile"

        document.page_content = re.sub(r'[^\w\s]', '', document.page_content)  # Remove punctuations
        # Example after this step: "i won't go there this is a testing paragraphs smile"

        document.page_content = re.sub(r'\s+[a-zA-Z]\s+', ' ', document.page_content)  # Remove standalone single alphabetical characters
        # Example after this step: "won't go there this is testing paragraphs smile"

        document.page_content = re.sub(r'\s+', ' ', document.page_content, flags=re.I)  # Substitute multiple consecutive spaces with a single space
        # Example after this step: "won't go there this is testing paragraphs smile"

        # Remove stopwords
        document.page_content = ' '.join(word for word in document.page_content.split() if word not in STOPWORDS)
        # Example after this step: "won't go there testing paragraphs smile"

        # Stemming
        stemmer = PorterStemmer()
        document.page_content = ' '.join(stemmer.stem(word) for word in document.page_content.split())
        # Example after this step: "won't go there test paragraph smile"

        # Lemmatization. (flies --> fly, went --> go)
        lemmatizer = WordNetLemmatizer()
        document.page_content = ' '.join(lemmatizer.lemmatize(word) for word in document.page_content.split())

    return documents    


def split_text(documents: list[Document]):
    """
    Split the text content of the given list of Document objects into smaller chunks.
    Args:
    documents (list[Document]): List of Document objects containing text content to split.
    Returns:
    list[Document]: List of Document objects representing the split text chunks.
    """

    # Clean text by removing stopwords
    cleaned_documents = clean_text(documents)

    # Initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, # Size of each chunk in characters
        chunk_overlap=50, # Overlap between consecutive chunks
        length_function=len, # Function to compute the length of the text
        add_start_index=True, # Flag to add start index to each chunk
    )

    #clean_text(documents)

    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(cleaned_documents)
    #print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks # Return the list of split text chunks

def save_to_chroma(CHROMA_PATH: str, chunks: list[Document]):
    """
    Save the given list of Document objects to a Chroma database.
    Args:
    chunks (list[Document]): List of Document objects representing text chunks to save.
    Returns:
    None
    """

    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new Chroma database from the documents using OpenAI embeddings
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(
            model='text-embedding-3-large'
        ),
        persist_directory=CHROMA_PATH
    )

    #print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store(DATA_PATH: str, CHROMA_PATH: str):
    """
    Function to generate vector database in chroma from documents.
    """
    documents = directory_load_documents(DATA_PATH) # Load documents from a source
    chunks = split_text(documents) # Split documents into manageable chunks
    #save_to_chroma(CHROMA_PATH, chunks) # Save the processed data to a data store    
    return chunks