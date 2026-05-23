import sys
import time
import importlib

def format_elapsed(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"

def print_slow(text, delay=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def main():
    print_slow("="*50)
    print_slow("🤖 BEM-VINDO AO RAG AGENT CHAT 🤖")
    print_slow("="*50)
    print("\nEscolha o framework que deseja utilizar:")
    print("1. LlamaIndex (RAG)")
    print("2. LangChain (RAG)")
    print("3. Vanilla LLM (Sem RAG)")
    
    choice = input("\nOpção (1, 2 ou 3): ").strip()
    
    if choice == "1":
        print("\n🔄 Carregando agente LlamaIndex...")
        try:
            llama_rag = importlib.import_module("llama-rag-system")
            agent = llama_rag.get_chat_engine()
            system = "LlamaIndex"
        except Exception as e:
            print(f"❌ Erro ao carregar o LlamaIndex: {e}")
            return
            
    elif choice == "2":
        print("\n🔄 Carregando agente LangChain...")
        try:
            langchain_rag = importlib.import_module("langchain-rag-system")
            agent = langchain_rag.get_chat_agent()
            system = "LangChain"
        except Exception as e:
            print(f"❌ Erro ao carregar o LangChain: {e}")
            return
            
    elif choice == "3":
        print("\n🔄 Carregando agente Vanilla LLM...")
        try:
            vanilla_sys = importlib.import_module("vanilla-system")
            agent = vanilla_sys.get_chat_agent()
            system = "Vanilla LLM"
        except Exception as e:
            print(f"❌ Erro ao carregar o Vanilla LLM: {e}")
            return
    else:
        print("❌ Opção inválida. Saindo...")
        return

    print_slow(f"\n✅ Agente {system} pronto! Digite 'sair' para encerrar a conversa.")
    print("-" * 50)
    
    while True:
        try:
            msg = input("\nVocê: ")
            if msg.lower() in ["sair", "exit", "quit", "q"]:
                print_slow("\nEncerrando o chat. Até logo! 👋")
                break
                
            start = time.perf_counter()
            
            if choice == "1":
                # LlamaIndex
                response = agent.chat(msg)
                elapsed = time.perf_counter() - start
                print(f"\n🤖 Agente: ", end="")
                print_slow(str(response), delay=0.005)
                
            elif choice == "2":
                # LangChain (via LangGraph)
                response = agent.invoke({"messages": [{"role": "user", "content": msg}]})
                elapsed = time.perf_counter() - start
                ai_msg = response["messages"][-1]
                print(f"\n🤖 Agente: ", end="")
                print_slow(ai_msg.content, delay=0.005)
                
            elif choice == "3":
                # Vanilla LLM
                response = agent.invoke({"input": msg})
                elapsed = time.perf_counter() - start
                print(f"\n🤖 Agente: ", end="")
                print_slow(getattr(response, "content", str(response)), delay=0.005)
            
            print(f"\n   ⏱  {format_elapsed(elapsed)}")
                
        except (KeyboardInterrupt, EOFError):
            print_slow("\n\nEncerrando o chat. Até logo! 👋")
            break
        except Exception as e:
            print(f"\n❌ Erro durante a conversa: {e}\n")

if __name__ == "__main__":
    main()
