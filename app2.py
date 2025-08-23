<<<<<<< HEAD
# app.py

import streamlit as st
import os
import boto3
from botocore.exceptions import NoCredentialsError
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import time

import firebase_admin
from firebase_admin import credentials, firestore
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


# --- Configuração da Página ---
st.set_page_config(page_title="Assistente de Pesquisa RAG", layout="wide")

# --- CSS Customizado para um Visual Profissional ---
st.markdown("""
<style>
    /* Esconde o menu hamburguer e o footer do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Estilo do container principal */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    /* Estilo da barra lateral */
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }

    /* Estilo dos botões de chat na sidebar */
    .st-emotion-cache-1f1G2gn {
        width: 100%;
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        background-color: transparent;
        margin-bottom: 0.5rem;
        text-align: left;
        padding: 0.5rem 1rem;
    }
    .st-emotion-cache-1f1G2gn:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# --- Carregamento de Segredos e Configurações ---
try:
    GOOGLE_API_KEY = st.secrets["google"]["api_key"]
    PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
    INDEX_NAME = st.secrets["pinecone"]["index_name"]
    AWS_ACCESS_KEY_ID = st.secrets["aws"]["access_key_id"]
    AWS_SECRET_ACCESS_KEY = st.secrets["aws"]["secret_access_key"]
    S3_BUCKET_NAME = st.secrets["aws"]["s3_bucket_name"]
    S3_REGION = st.secrets["aws"]["s3_region"]

    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

except (KeyError, FileNotFoundError):
    st.error("⚠️ Arquivo de segredos (secrets.toml) não encontrado ou mal configurado. Por favor, siga as instruções para criar o seu.")
    st.stop()

try:
    firestore_creds = dict(st.secrets["firestore_credentials"])
    cred = credentials.Certificate(firestore_creds)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    st.error(f"Falha ao conectar ao Firestore: {e}. O histórico de chat não será salvo.")
    db = None

    # --- Gerenciamento do Histórico de Conversa ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
    # Carrega as sessões existentes do Firestore se a conexão for válida
    if db:
        chat_histories_ref = db.collection("chat_histories")
        try:
            sessions = [doc.id for doc in chat_histories_ref.stream()]
            st.session_state.chat_sessions = sessions
        except Exception as e:
            st.error(f"Erro ao carregar sessões do Firestore: {e}")

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

class FirestoreChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        if not db:
            raise ConnectionError("Cliente Firestore não inicializado.")
        self.session_id = session_id
        self.collection = db.collection("chat_histories")
        self.doc_ref = self.collection.document(self.session_id)

    @property
    def messages(self) -> list[BaseMessage]:
        doc = self.doc_ref.get()
        if doc.exists:
            return messages_from_dict(doc.to_dict().get("messages", []))
        return []

    def add_message(self, message: BaseMessage) -> None:
        current_messages = self.messages
        current_messages.append(message)
        # Serialize each message individually before saving the list
        serialized_messages = [message_to_dict(msg) for msg in current_messages]
        self.doc_ref.set({"messages": serialized_messages})

    def clear(self) -> None:
        self.doc_ref.delete()


# --- Funções em Cache para Inicialização de Serviços ---
@st.cache_resource
def init_services():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_output_tokens=None, google_api_key=GOOGLE_API_KEY)
        vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        s3_client = boto3.client('s3', region_name=S3_REGION)
        return llm, retriever, s3_client
    except Exception as e:
        st.error(f"Erro ao inicializar os serviços: {e}")
        st.stop()

# --- Gerenciamento do Histórico de Conversa ---


if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if db:
        return FirestoreChatMessageHistory(session_id)
    else:
        # Fallback para memória se o Firestore falhar
        # Isso não é ideal para produção, mas evita que o app quebre
        st.warning("Usando histórico em memória. As conversas não serão salvas.")
        if f"fallback_{session_id}" not in st.session_state:
            st.session_state[f"fallback_{session_id}"] = InMemoryChatMessageHistory()
        return st.session_state[f"fallback_{session_id}"]

# --- Criação da Cadeia RAG ---
def create_rag_chain_with_history(llm, retriever):
    history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Dada a conversa acima, gere uma consulta de pesquisa para recuperar informações relevantes para a última pergunta."),
    ])
    history_aware_retriever_chain = create_history_aware_retriever(llm, retriever, history_aware_retriever_prompt)
    
    system_prompt = (
      "Você é um assistente para tarefas de resposta a perguntas. "
      "Use as partes do contexto recuperado para responder à pergunta. "
      "Interprete e reformule as informações com suas próprias palavras. "
      "Se a resposta não estiver presente no contexto, diga claramente que não sabe a resposta. "
      "Não cite ou utilize imagens/figuras. Descreva tudo de forma textual e detalhada. "
      "Seja sempre cordial e prestativo."
      "\n\n"
      "Contexto: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever_chain, Youtube_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

# --- Função para Links S3 ---
def generate_presigned_url(s3_client, object_key: str):
    clean_key = os.path.basename(object_key).replace('.md', '.pdf')
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': clean_key},
            ExpiresIn=3600
        )
        return url, clean_key
    except Exception:
        return None, None

# --- Inicialização ---
llm, retriever, s3_client = init_services()

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        # Substitua pela URL da sua logo
        st.image("logolat.png", width=120)
    with col2:
        # Substitua pela URL da sua outra logo
        st.image("UFCG_logo.png", width=120)

    st.title("Minhas Conversas")

    if st.button("💬 + Novo Chat", use_container_width=True):
        chat_id = f"chat_{int(time.time())}"
        st.session_state.chat_sessions.append(chat_id)
        st.session_state.active_chat_id = chat_id
        # Inicializa com uma mensagem de boas-vindas
        get_session_history(chat_id).add_ai_message("Olá! Como posso te ajudar hoje?")

    st.markdown("---")

    # Exibe os chats existentes
    for chat_id in reversed(st.session_state.chat_sessions):
        # Pega a primeira mensagem do usuário para usar como título do chat
        history = get_session_history(chat_id)
        chat_title = "Nova Conversa"
        if len(history.messages) > 1:
             chat_title = history.messages[1].content[:30] + "..." # Pega os 30 primeiros caracteres da primeira pergunta
        
        if st.button(chat_title, key=chat_id, use_container_width=True):
            st.session_state.active_chat_id = chat_id

# --- ÁREA DE CHAT PRINCIPAL ---
if not st.session_state.active_chat_id:
    st.info("Selecione uma conversa ou inicie um 'Novo Chat' na barra lateral.")
else:
    active_id = st.session_state.active_chat_id
    chat_history = get_session_history(active_id)

    # Exibe as mensagens do chat ativo
    for msg in chat_history.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    # Campo de input
    if query := st.chat_input("Pergunte algo..."):
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("ai"):
            with st.spinner("Analisando documentos e gerando resposta..."):
                conversational_rag_chain = create_rag_chain_with_history(llm, retriever)
                config = {"configurable": {"session_id": active_id}}
                
                response = conversational_rag_chain.invoke({"input": query}, config=config)
                
                answer = response.get("answer", "Desculpe, não consegui processar sua pergunta.")
                st.markdown(answer)

                # Exibe as fontes
                if "context" in response and response["context"]:
                    with st.expander("📚 Fontes Consultadas"):
                        unique_sources = {doc.metadata.get('source') for doc in response["context"]}
                        links_gerados = 0
                        for source_path in unique_sources:
                            if source_path:
                                url, filename = generate_presigned_url(s3_client, source_path)
                                if url:
                                    st.markdown(f"- [{filename}]({url})")
                                    links_gerados += 1
                        if links_gerados == 0:
                            st.write("Nenhuma fonte válida encontrada no S3 para os documentos recuperados.")
=======
# app.py

import streamlit as st
import os
import boto3
from botocore.exceptions import NoCredentialsError
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import time
import asyncio


import firebase_admin
from firebase_admin import credentials, firestore
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


# --- Configuração da Página ---
st.set_page_config(page_title="Assistente de Pesquisa RAG", layout="wide")

# --- CSS Customizado para um Visual Profissional ---
st.markdown("""
<style>
    /* Esconde o menu hamburguer e o footer do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Estilo do container principal */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    /* Estilo da barra lateral */
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }

    /* Estilo dos botões de chat na sidebar */
    .st-emotion-cache-1f1G2gn {
        width: 100%;
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        background-color: transparent;
        margin-bottom: 0.5rem;
        text-align: left;
        padding: 0.5rem 1rem;
    }
    .st-emotion-cache-1f1G2gn:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# --- Carregamento de Segredos e Configurações ---
try:
    GOOGLE_API_KEY = st.secrets["google"]["api_key"]
    PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
    INDEX_NAME = st.secrets["pinecone"]["index_name"]
    AWS_ACCESS_KEY_ID = st.secrets["aws"]["access_key_id"]
    AWS_SECRET_ACCESS_KEY = st.secrets["aws"]["secret_access_key"]
    S3_BUCKET_NAME = st.secrets["aws"]["s3_bucket_name"]
    S3_REGION = st.secrets["aws"]["s3_region"]

    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

except (KeyError, FileNotFoundError):
    st.error("⚠️ Arquivo de segredos (secrets.toml) não encontrado ou mal configurado. Por favor, siga as instruções para criar o seu.")
    st.stop()

try:
    firestore_creds = dict(st.secrets["firestore_credentials"])
    cred = credentials.Certificate(firestore_creds)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    st.error(f"Falha ao conectar ao Firestore: {e}. O histórico de chat não será salvo.")
    db = None

    # --- Gerenciamento do Histórico de Conversa ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
    # Carrega as sessões existentes do Firestore se a conexão for válida
    if db:
        chat_histories_ref = db.collection("chat_histories")
        try:
            sessions = [doc.id for doc in chat_histories_ref.stream()]
            st.session_state.chat_sessions = sessions
        except Exception as e:
            st.error(f"Erro ao carregar sessões do Firestore: {e}")

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

class FirestoreChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        if not db:
            raise ConnectionError("Cliente Firestore não inicializado.")
        self.session_id = session_id
        self.collection = db.collection("chat_histories")
        self.doc_ref = self.collection.document(self.session_id)

    @property
    def messages(self) -> list[BaseMessage]:
        doc = self.doc_ref.get()
        if doc.exists:
            return messages_from_dict(doc.to_dict().get("messages", []))
        return []

    def add_message(self, message: BaseMessage) -> None:
        current_messages = self.messages
        current_messages.append(message)
        # Serialize each message individually before saving the list
        serialized_messages = [message_to_dict(msg) for msg in current_messages]
        self.doc_ref.set({"messages": serialized_messages})

    def clear(self) -> None:
        self.doc_ref.delete()


# --- Funções em Cache para Inicialização de Serviços ---
@st.cache_resource
def init_services():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    # --- FIM DA CORREÇÃO ---

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_output_tokens=None, google_api_key=GOOGLE_API_KEY)
        vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        s3_client = boto3.client('s3', region_name=S3_REGION)
        return llm, retriever, s3_client
    except Exception as e:
        st.error(f"Erro ao inicializar os serviços: {e}")
        st.stop()

# --- Gerenciamento do Histórico de Conversa ---


if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if db:
        return FirestoreChatMessageHistory(session_id)
    else:
        # Fallback para memória se o Firestore falhar
        # Isso não é ideal para produção, mas evita que o app quebre
        st.warning("Usando histórico em memória. As conversas não serão salvas.")
        if f"fallback_{session_id}" not in st.session_state:
            st.session_state[f"fallback_{session_id}"] = InMemoryChatMessageHistory()
        return st.session_state[f"fallback_{session_id}"]

# --- Criação da Cadeia RAG ---
def create_rag_chain_with_history(llm, retriever):
    history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Dada a conversa acima, gere uma consulta de pesquisa para recuperar informações relevantes para a última pergunta."),
    ])
    history_aware_retriever_chain = create_history_aware_retriever(llm, retriever, history_aware_retriever_prompt)
    
    system_prompt = (
      "Você é um assistente para tarefas de resposta a perguntas. "
      "Use as partes do contexto recuperado para responder à pergunta. "
      "Interprete e reformule as informações com suas próprias palavras. "
      "Se a resposta não estiver presente no contexto, diga claramente que não sabe a resposta. "
      "Não cite ou utilize imagens/figuras. Descreva tudo de forma textual e detalhada. "
      "Seja sempre cordial e prestativo."
      "\n\n"
      "Contexto: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever_chain, Youtube_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

# --- Função para Links S3 ---
def generate_presigned_url(s3_client, object_key: str):
    clean_key = os.path.basename(object_key).replace('.md', '.pdf')
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': clean_key},
            ExpiresIn=3600
        )
        return url, clean_key
    except Exception:
        return None, None

# --- Inicialização ---
llm, retriever, s3_client = init_services()

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        # Substitua pela URL da sua logo
        st.image("logolat.png", width=120)
    with col2:
        # Substitua pela URL da sua outra logo
        st.image("UFCG_logo.png", width=120)

    st.title("Minhas Conversas")

    if st.button("💬 + Novo Chat", use_container_width=True):
        chat_id = f"chat_{int(time.time())}"
        st.session_state.chat_sessions.append(chat_id)
        st.session_state.active_chat_id = chat_id
        # Inicializa com uma mensagem de boas-vindas
        get_session_history(chat_id).add_ai_message("Olá! Como posso te ajudar hoje?")

    st.markdown("---")

    # Exibe os chats existentes
    for chat_id in reversed(st.session_state.chat_sessions):
        # Pega a primeira mensagem do usuário para usar como título do chat
        history = get_session_history(chat_id)
        chat_title = "Nova Conversa"
        if len(history.messages) > 1:
             chat_title = history.messages[1].content[:30] + "..." # Pega os 30 primeiros caracteres da primeira pergunta
        
        if st.button(chat_title, key=chat_id, use_container_width=True):
            st.session_state.active_chat_id = chat_id

# --- ÁREA DE CHAT PRINCIPAL ---
if not st.session_state.active_chat_id:
    st.info("Selecione uma conversa ou inicie um 'Novo Chat' na barra lateral.")
else:
    active_id = st.session_state.active_chat_id
    chat_history = get_session_history(active_id)

    # Exibe as mensagens do chat ativo
    for msg in chat_history.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    # Campo de input
    if query := st.chat_input("Pergunte algo..."):
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("ai"):
            with st.spinner("Analisando documentos e gerando resposta..."):
                conversational_rag_chain = create_rag_chain_with_history(llm, retriever)
                config = {"configurable": {"session_id": active_id}}
                
                response = conversational_rag_chain.invoke({"input": query}, config=config)
                
                answer = response.get("answer", "Desculpe, não consegui processar sua pergunta.")
                st.markdown(answer)

                # Exibe as fontes
                if "context" in response and response["context"]:
                    with st.expander("📚 Fontes Consultadas"):
                        unique_sources = {doc.metadata.get('source') for doc in response["context"]}
                        links_gerados = 0
                        for source_path in unique_sources:
                            if source_path:
                                url, filename = generate_presigned_url(s3_client, source_path)
                                if url:
                                    st.markdown(f"- [{filename}]({url})")
                                    links_gerados += 1
                        if links_gerados == 0:
                            st.write("Nenhuma fonte válida encontrada no S3 para os documentos recuperados.")
>>>>>>> d63bed1 (Novo início sem segredos)
