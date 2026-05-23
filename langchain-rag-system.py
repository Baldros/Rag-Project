from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

def get_chat_agent():    
    # Inicializar o LLM corretamente com o model_name e api_key reais
    llm = ChatOllama(model="qwen3.5:0.8b")
    
    # O LangChain conecta e mapeia seu SQLite automaticamente
    db = SQLDatabase.from_uri("sqlite:///chroma.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # O kit já te dá as ferramentas prontas!
    tools = toolkit.get_tools()
    
    # Agora é só passar para o agente
    agent = create_agent(
        llm,
        tools,
        system_prompt="Você é um assistente útil especializado em interagir com o banco de dados."
    )
    
    return agent