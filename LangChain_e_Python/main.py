# Importações para gerenciar o sistema e variáveis de ambiente
import os 
from dotenv import load_dotenv
# Importação da conexão com o modelo local (simulando OpenAI)
from langchain_openai import ChatOpenAI
# Importação de templates de perguntas e formatadores de respostas
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
# Pydantic é usado para garantir que a IA responda exatamente no formato que você quer (campos específicos)
from pydantic import Field, BaseModel 
# Ferramenta para ver "debaixo do capô" e entender o que a IA está pensando (debug)
from langchain_core.globals import set_debug

# Ativa o modo detalhado para você ver todo o log de mensagens no terminal
set_debug(True)
# Carrega as configurações (URLs e Chaves) do arquivo .env
load_dotenv()
api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")

# Configura o modelo para conversar com o LM Studio localmente
modelo = ChatOpenAI(
    model="local-model",         
    base_url=api_base,           
    api_key=api_key,             
    temperature=0.3, # Temperatura baixa para a IA ser mais objetiva e seguir o formato JSON
    max_completion_tokens=500
)

# --- DEFINIÇÃO DE ESTRUTURAS (SCHEMA) ---
# Aqui você define para o Python como devem ser os "objetos" de saída
class Destino(BaseModel):
    cidade: str = Field(description="a cidade que voce recomenda visitar")
    motivo: str = Field(description="o motivo pelo qual é interessante visitar essa cidade")

class Restaurante(BaseModel):
    cidade: str = Field(description="a cidade recomendada")
    restaurantes: str = Field(description="lista de restaurantes recomendados")

# --- PARSEADORES ---
# Eles servem para "pegar" o texto da IA e transformar em um dicionário/objeto do Python
parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurante = JsonOutputParser(pydantic_object=Restaurante)

# --- TEMPLATES DE PROMPT ---
# 1. Sugestão de cidade baseada em interesse
modelo_de_cidade = PromptTemplate(
    template="Sugira uma cidade dado ao meu interesse por {interesse}. {formato_de_saida}",
    input_variables=["interesse"],
    # Injeta as instruções de como o JSON deve ser montado automaticamente
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()}
)

# 2. Sugestão de restaurantes para a cidade que a IA acabou de sugerir
modelo_de_restaurantes = PromptTemplate(
    template="Sugira restaurantes locais em {cidade}. {formato_de_saida}",
    input_variables=["cidade"], 
    partial_variables={"formato_de_saida": parseador_restaurante.get_format_instructions()}
)

# 3. Sugestão de cultura (saída final em texto simples)
modelo_cultura = PromptTemplate(
    template="Sugira 3 atividades culturais imperdíveis em {cidade}",
    input_variables=["cidade"]
)

# --- CRIAÇÃO DAS CADEIAS (CHAINS) ---
# Cada cadeia executa: Prompt -> IA -> Formatador de Saída
cadeia1 = modelo_de_cidade | modelo | parseador_destino
cadeia2 = modelo_de_restaurantes | modelo | parseador_restaurante
cadeia3 = modelo_cultura | modelo | StrOutputParser()

# --- SEQUÊNCIA FINAL (O PIPELINE) ---
# O resultado da cadeia1 (Cidade) entra na cadeia2, que entra na cadeia3
# OBS: Para isso funcionar perfeitamente, os nomes das variáveis (como 'cidade') devem coincidir
cadeia = (cadeia1 | cadeia2 | cadeia3)

# Executa o processo completo passando o interesse inicial
resultado = cadeia.invoke({"interesse": "porto-rico"})

# Imprime o resultado final (Atividades culturais sugeridas no último passo)
print(resultado)