# loader de documentos PDF
from langchain_community.document_loaders import PyPDFLoader
# Divisão de texto em blocos
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings
from langchain_openai import OpenAIEmbeddings
# Banco vetorial
from langchain_community.vectorstores import Chroma
# Cadeia RAG
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")


caminho_pdf=("regras_futebol.pdf")
loader = PyPDFLoader(caminho_pdf)
documentos = loader.load()
len(documentos)