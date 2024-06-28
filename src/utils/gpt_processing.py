from langchain.chat_models import ChatOpenAI # Import OpenAI LLM
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai # Google's LLM
from utils.text_processing import generate_data_store
from docx import Document
from markdowntodocx.markdownconverter import markdownToWordFromString


def collect_text_context(DATA_PATH: str):
    chuncks = generate_data_store(DATA_PATH,'')
    context_text = [doc.page_content for doc in chuncks]

    return context_text

def get_llm_response(prompt: str, DATA_PATH: str, doc_name: str, model='openai'):

    context_text = collect_text_context(DATA_PATH)

    prompt_template = ChatPromptTemplate.from_template(prompt)
    prompt = prompt_template.format(context=context_text)

    if model == 'openai':
      response_text = chat_openai(prompt)
    elif model == 'google':
       response_text = chat_googleai(prompt)
    else:
       response_text = 'LLM Model Invalid' 

    save_to_docx(response_text, doc_name)

    return response_text   

def chat_openai(prompt: str):
    # Initialize OpenAI chat model
    model = ChatOpenAI(
       model = 'gpt-3.5-turbo',
       temperature=0.3
    )
    # Generate response text based on the prompt
    response_text = model.predict(prompt[:16385]) # Maximum tokens

    return response_text

def chat_googleai(prompt: str):    
    
    # Initialize Google chat model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash"
    )    
    chat = model.start_chat(enable_automatic_function_calling=True)
    # Generate response text based on the prompt
    response = chat.send_message(prompt)
    response_text = response.text

    return response_text    

def save_to_docx(llm_response:str, doc_name: str):
    document = Document()
    document.add_paragraph(llm_response)
    document.save('../../data/generated_docs/'+doc_name)
    res , msg = markdownToWordFromString(llm_response, '../../data/generated_docs/formated/'+doc_name)
    print(res, msg)