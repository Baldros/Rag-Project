#from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
#from utils import get_env_variables

def get_chat_agent():
    #api_key, model_name = get_env_variables()
    
    # Inicializar o LLM corretamente com o model_name e api_key reais
    #llm = ChatOpenAI(model=model_name, api_key=api_key)
    llm = ChatOllama(model="qwen3.5:0.8b")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente útil e responde de forma clara e direta."),
        ("human", "{input}"),
    ])
    
    # Para o vanilla, apenas conectamos o prompt direto no LLM (LCEL)
    chain = prompt | llm
    
    return chain