import os

from langchain.agents import create_agent

# imports do Agente
from langchain.tools import tool  # O decorator que você achou!

# imports de loader e splitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# imports de vetor e embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# imports do LLM
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prompts import template

#  -- Initial Config --
FAQ_PATH = "data/FAQ - PerguntasFrequentes - MovimentacaoPessoal_v5.1.pdf"

# --- LOADING MODELS
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
llm = ChatOllama(model="llama3.1:8b", temperature=0.5)

# --- PREPARING FAQ ---

DB_FAISS_PATH = "faq_db_faiss"

if not os.path.exists(DB_FAISS_PATH):
    loader = PyPDFLoader(FAQ_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    chunks = text_splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text extracted, verify pdf")
        exit(1)

    vector_store = FAISS.from_documents(chunks, embeddings_model)
    vector_store.save_local(DB_FAISS_PATH)
    print(f"Database saved in {DB_FAISS_PATH}")
else:
    print("loading vetorial database...")
    vector_store = FAISS.load_local(
        DB_FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True
    )


@tool
def retrieve_context(query: str) -> str:
    """
    Busca no FAQ do governo informações para responder a pergunta
    """
    retrieved_docs = vector_store.similarity_search(query, k=3)

    context = "\n\n".join(
        f"Fonte {i + 1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)
    )
    print(f"--- Contexto Encontrado para a Pergunta: '{query}' ---")
    print(context)
    return context


# busca os 'k' chunks mais relevantes.
print("building RAG chain")
prompt = ChatPromptTemplate.from_template(template)
prompt_text = prompt.format(context="{context}", input="{input}")
tools = [retrieve_context]
# document_chain = create_stuff_documents_chain(llm, prompt)

agent = create_agent(llm, tools, system_prompt=prompt_text)

first_question = "Como solicitar a movimentação?"
print(f"\nQuestion: {first_question}")
answer1 = agent.invoke({"messages": [{"role": "user", "content": first_question}]})
for msg in answer1["messages"]:
    if msg.__class__.__name__ == "AIMessage" and msg.content.strip():
        final_answer = msg.content
print(f"Final answer model: {final_answer}")

# second_question = "Quais as modalidades da movimentação?"
# print(f"\nQuestion: {second_question}")
# answer2 = agent.invoke({"input": second_question})
# print(f"answer: {answer2['answer']}")

# third_question = "Há  limite  de  solicitação  de  movimentação  de  servidores  ou  empregados  públicos por órgão ou entidade e por período?"
# print(f"\nQuestion: {third_question}")
# answer3 = agent.invoke({"input": third_question})
# print(f"answer: {answer3['answer']}")
