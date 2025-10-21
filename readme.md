# 🤖 Local RAG - FAQ de Movimentação de Pessoal

Este projeto implementa um **sistema RAG (Retrieval-Augmented Generation)** local, que responde perguntas sobre o documento oficial de **FAQ de Movimentação de Pessoal** do Governo Federal.
A aplicação usa **LangChain**, **FAISS**, **Ollama** e **HuggingFace Embeddings** para buscar informações relevantes no PDF e gerar respostas contextuais com LLM.

---

## 🧩 Tecnologias utilizadas

- [LangChain](https://www.langchain.com/) — Framework de orquestração de LLMs
- [FAISS](https://faiss.ai/) — Banco vetorial para busca semântica
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — Modelo de embeddings multilíngue
- [Ollama](https://ollama.ai/) — Execução local do modelo `llama3.1:8b`
- [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) — Leitura e parsing de PDFs

---

## 📁 Estrutura do projeto

```bash
local-rag-faq/
│
├── data/
│ └── FAQ - PerguntasFrequentes - MovimentacaoPessoal_v5.1.pdf
│
├── faq_db_faiss/ # Gerado automaticamente após o primeiro run
│
├── main.py # Script principal (monta o RAG e executa perguntas)
├── prompts.py # Template de prompt usado pelo agente
├── requirements.txt # Dependências do ambiente
└── README.md
```

---

## ⚙️ Instalação

### 1. Criar ambiente virtual (recomendado com Conda)

```bash
conda create -n LangChainEnv python=3.10
conda activate LangChainEnv
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt

```

### 3. Instalar e iniciar o Ollama

Baixe e instale o Ollama: https://ollama.ai

Depois, baixe o modelo usado:

ollama pull llama3.1:8b

E deixe o servidor Ollama rodando:

ollama serve

### 🧠 Como funciona

Carregamento do documento PDF
→ PyPDFLoader extrai o texto.

Divisão em blocos menores (chunks)
→ RecursiveCharacterTextSplitter quebra o texto em segmentos de ~1000 caracteres.

Criação do banco vetorial FAISS
→ Cada chunk é convertido em vetor com o modelo paraphrase-multilingual-MiniLM-L12-v2.

Busca semântica (Retriever)
→ Dado um prompt do usuário, são buscados os k trechos mais relevantes.

Geração da resposta (RAG)
→ O contexto recuperado é passado ao LLM llama3.1:8b via Ollama, que gera uma resposta natural e contextualizada.

### 🧩 Prompt do agente

O prompt base é definido em prompts.py e controlado por:

```bash
prompt = ChatPromptTemplate.from_template(template)
```

Você pode ajustar o tom ou formato da resposta editando esse template.
