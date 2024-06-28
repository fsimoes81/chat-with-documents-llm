import sys
sys.path.append("../")
from utils.gpt_processing import get_llm_response
from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
file_path = '../../data/upload'

def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError as e:
        print("Error occurred while deleting files:", e)

def save_uploadedfile(uploadedfile):
    delete_files_in_directory(file_path)
    with open(os.path.join(file_path, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"Arquivo: {uploadedfile.name} salvo em {file_path}")

def gera_scope(model: str):
    PROMPT_TEMPLATE = """ 
    Você é um Analista de Dados que irá atuar em um projeto de business intelligence. Baseado na transcrição da reunião abaixo, monte um documento baseado na estrutura abaixo. Este documento precisa ser inteiramente baseado na {context}.
    Segue a estrutura abaixo:
    1. Introdução
    2. Participantes da Reunião
    3. Necessidades de Técnicas:
    a. Indicadores
    i. Sistemas de origem
    ii. Banco de dados do sistema de origem
    iii. Nome da tabela
    iv. Regras para calcular o indicador
    b. Modelagem
    i. Relacionamento entre as tabelas mencionadas 
    4. Itens Pendentes de Definição:
    5. Próximos Passos:
    """
   
    escopo_file_name = f'teste_doc_escopo_{model}.docx'
    resposta = get_llm_response(PROMPT_TEMPLATE, file_path, escopo_file_name, model=model)
    arquivo_escopo = os.path.join('../../data/generated_docs/formated', escopo_file_name)

    return arquivo_escopo, resposta

uploaded_files = st.file_uploader("Escolha seus arquivos", accept_multiple_files=True)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        save_uploadedfile(uploaded_file)

llm_model = st.radio(
    "Defina qual modelo será usado:",
    ["ChatGPT", "Google Gemini"],
    index=0,
)    

if llm_model == "ChatGPT":
    model = 'openai'
else:
    model = 'google'

st.write("Você selecionou:", llm_model)

if st.button("Gera Documento"):
    caminho, response = gera_scope(model)
    st.write(response)
    st.divider()
    st.markdown(f"Arquivo gerado em: {caminho}")
    st.divider()
    
    # Adiciona o botão para download do arquivo gerado
    with open(caminho, "rb") as file:
        st.download_button(
            label="Baixar arquivo .docx",
            data=file,
            file_name=os.path.basename(caminho),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )