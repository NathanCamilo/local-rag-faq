import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools import tool
from prompts import template
#  -- Initial Config --
FAQ_PATH = "data/FAQ - PerguntasFrequentes - MovimentacaoPessoal_v5.1.pdf"

# --- LOADING MODELS
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
llm = ChatOllama(model="llama3.1:8b", temperature=0.9)

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

      db = FAISS.from_documents(chunks, embeddings_model)
      db.save_local(DB_FAISS_PATH)
      print(f'Database saved in {DB_FAISS_PATH}')
else:
      print("loading vetorial database...")
      db = FAISS.load_local(DB_FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True)

print("building RAG chain")


prompt = ChatPromptTemplate.from_template(template)

# busca os 'k' chunks mais relevantes.
retriever = db.as_retriever(search_kwargs={"k": 5})

# document_chain = create_stuff_documents_chain(llm, prompt)

agent = create_agent(llm, retriever, system_prompt=prompt)

first_question = "Como solicitar a movimentação?"
print(f"\nQuestion: {first_question}")
answer1 = agent.invoke({"input": first_question})
print(f"answer: {answer1['answer']}")

second_question = "Quais as modalidades da movimentação?"
print(f"\nQuestion: {second_question}")
answer2 = agent.invoke({"input": second_question})
print(f"answer: {answer2['answer']}")

third_question = "Há  limite  de  solicitação  de  movimentação  de  servidores  ou  empregados  públicos por órgão ou entidade e por período?"
print(f"\nQuestion: {third_question}")
answer3 = agent.invoke({"input": third_question})
print(f"answer: {answer3['answer']}")