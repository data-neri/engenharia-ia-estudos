from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os


load_dotenv()
api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
   model = "local_model",
   base_url=api_base,
   api_key=api_key,
   temperature=0.8,
   max_tokens=250
)


embedings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} 
)

arquivos = [
    "documentos/GTB_gold_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf",
    "documentos/GTB_standard_Nov23.pdf"
]

documento = sum (
    [
    PyPDFLoader(arquivo).load() for arquivo in arquivos
    ],[]
)


pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,chunk_overlap=200
).split_documents(documento)

dados_recuperados = FAISS.from_documents(
    pedacos, embedings
).as_retriever(search_kwargs={"k":2})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "responda somente com base somente no que que esta no documento, faça um interpretação sobre o que aconteceu com a pessoa e auxilie oque ela pode fazer e se pode ajudar com o que aconteceu e se é possivel ajudar ela com algo com base no documento"),
        ("human", "{query}\n\ncontexto: \n{contexto}\n\nResposta:")
    ]
)

cadeia = prompt | modelo | StrOutputParser()

def responder(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    return cadeia.invoke({
        "query": pergunta,
        "contexto": contexto
    }) 

pergunta_usuario = input("O que você deseja saber sobre o documento? ")
print("\nBuscando resposta...\n")
print(responder(pergunta_usuario))