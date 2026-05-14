import os # Biblioteca para interagir com o sistema operacional
from dotenv import load_dotenv # Carrega as variáveis de ambiente do arquivo .env
from langchain_openai import ChatOpenAI # Conecta com o modelo de IA (LM Studio)
from langchain_core.prompts import ChatPromptTemplate # Estrutura o modelo de mensagens
from langchain_core.output_parsers import StrOutputParser # Limpa a resposta para vir apenas o texto
from langchain_core.chat_history import InMemoryChatMessageHistory # Cria um "balde" na memória RAM para guardar as mensagens
from langchain_core.runnables.history import RunnableWithMessageHistory # O "gerenciador" que une a IA com a memória

# Carrega as chaves e URLs de conexão
load_dotenv()
api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")

# Configura o modelo de IA local
modelo = ChatOpenAI(
    model = "local_model",
    base_url = api_base,
    api_key = api_key,
    temperature = 0.3, # Temperatura baixa para respostas mais precisas e menos "inventadas"
    max_completion_tokens = 4096 # Limite máximo de palavras na resposta
)

# Define o roteiro da conversa
prompt_sugestão = ChatPromptTemplate.from_messages(
 [
  ("system", "Você em um agente de viagens famoso, se apresente como senhor passeios "),
  # O placeholder é um espaço reservado onde o LangChain vai injetar o histórico da conversa automaticamente
  ("placeholder","{historico}"), 
  ("human", "{pergunta}") 
 ]
)

# Cria a sequência básica: Pergunta -> IA -> Texto Limpo
cadeia = prompt_sugestão | modelo | StrOutputParser()

# Dicionário simples para guardar diferentes conversas (uma para cada usuário ou sessão)
memoria = {}
sessao = "aula_langchain" # Nome da "sala" de conversa atual

# Função que verifica se já existe um histórico para a sessão; se não existir, cria um novo
def historico_por_sessao(sessao :str):
 if sessao not in memoria: 
  memoria[sessao] = InMemoryChatMessageHistory()
 return memoria[sessao] 

# Lista de perguntas para testar se a IA lembra do contexto anterior
lista_perguntas = [
   "quero visitar o Brasil, qual é o melhor lugar para eu ir",
   "qual a melhor epoca do ano para ir?" # Note que aqui a IA precisa saber que ainda estamos falando de Brasil
]

# A "Mágica": envolve a cadeia comum com o gerenciador de memória
cadeia_memoria = RunnableWithMessageHistory(
 runnable=cadeia, # A lógica de IA que criamos acima
 get_session_history=historico_por_sessao, # A função que busca/salva o histórico
 input_messages_key="pergunta", # Nome da variável da pergunta atual
 history_messages_key="historico" # Nome da variável onde o histórico será injetado no prompt
)

# Loop para percorrer a lista de perguntas e simular um chat real
for um_pergunta in lista_perguntas:
 # O invoke agora recebe também a configuração da 'session_id' para saber qual histórico usar
 resposta = cadeia_memoria.invoke(
  {
   "pergunta": um_pergunta
  },
  config={"session_id": sessao}
 )
 # Exibe o diálogo no terminal
 print("usuario:", um_pergunta)
 print("IA:", resposta, "\n")