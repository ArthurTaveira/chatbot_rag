# app3.py (substitua o seu atual por esse)
import os
import asyncio
import chainlit as cl
import firebase_admin
from firebase_admin import credentials, firestore
import boto3
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict, AIMessage

# --- CONFIGS (via ENV) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION")

# --- FIRESTORE (tenta inicializar, mas não quebra se falhar) ---
db = None
try:
    cred_path = os.getenv("FIRESTORE_CREDENTIALS", "firestore_credentials.json")
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firestore inicializado com sucesso.")
    else:
        print(f"Arquivo de credenciais Firestore não encontrado em: {cred_path}")
except Exception as e:
    print("⚠️ Erro ao conectar ao Firestore:", e)
    db = None

# --- Classe de histórico com fallback para memória ---
class FirestoreChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        if not db:
            print("Firestore não inicializado — usando fallback InMemoryChatMessageHistory.")
            self._use_fallback = True
            self._fallback = InMemoryChatMessageHistory()
            return
        self._use_fallback = False
        self.collection = db.collection("chat_histories")
        self.doc_ref = self.collection.document(self.session_id)

    @property
    def messages(self) -> list[BaseMessage]:
        if self._use_fallback:
            return self._fallback.messages
        doc = self.doc_ref.get()
        if doc.exists:
            return messages_from_dict(doc.to_dict().get("messages", []))
        return []

    def add_message(self, message: BaseMessage) -> None:
        if self._use_fallback:
            return self._fallback.add_message(message)
        current_messages = self.messages
        current_messages.append(message)
        serialized = [message_to_dict(msg) for msg in current_messages]
        self.doc_ref.set({"messages": serialized})

    def clear(self) -> None:
        if self._use_fallback:
            return self._fallback.clear()
        self.doc_ref.delete()

# --- Inicialização segura dos serviços (retorna None em falha) ---
def init_services():
    try:
        if not GOOGLE_API_KEY or not PINECONE_API_KEY or not INDEX_NAME:
            print("Aviso: Algumas variáveis de ambiente (GOOGLE_API_KEY/PINECONE_API_KEY/INDEX_NAME) não estão definidas.")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
        vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        s3_client = boto3.client('s3', region_name=S3_REGION) if S3_REGION else None
        print("Serviços inicializados: llm, retriever, s3_client (s3_client pode ser None).")
        return llm, retriever, s3_client
    except Exception as e:
        print("Erro ao inicializar serviços (llm/retriever/s3):", e)
        return None, None, None

llm, retriever, s3_client = init_services()

# --- Cria a cadeia RAG com histórico (segura) ---
def create_rag_chain_with_history():
    if llm is None or retriever is None:
        print("llm ou retriever não inicializados — não é possível criar rag_chain.")
        return None

    history_aware_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Dada a conversa acima, gere uma consulta de pesquisa relevante."),
    ])
    history_aware_chain = create_history_aware_retriever(llm, retriever, history_aware_prompt)

    system_prompt = (
        "Você é um assistente de pesquisa. "
        "Responda usando apenas o contexto recuperado. "
        "Se não souber, diga claramente. "
        "Explique de forma textual e detalhada."
        "\n\nContexto: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_chain, qa_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: FirestoreChatMessageHistory(session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

rag_chain = create_rag_chain_with_history()

# --- Função para gerar links S3 (verifica s3_client) ---
def generate_presigned_url(object_key: str):
    if not s3_client:
        return None, None
    clean_key = os.path.basename(object_key).replace('.md', '.pdf')
    try:
        url = s3_client.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET_NAME, 'Key': clean_key}, ExpiresIn=3600)
        return url, clean_key
    except Exception as e:
        print("Erro ao gerar presigned URL:", e)
        return None, None

# --- Events Chainlit ---
@cl.on_chat_start
async def start_chat():
    await cl.Message(content="👋 Olá! Como posso te ajudar hoje?").send()

@cl.on_message
async def main(message):
    # Se o Chainlit passar um objeto Message, pega o texto; se passar string, usa direto
    if hasattr(message, "content"):
        query = message.content
    elif isinstance(message, str):
        query = message
    else:
        # fallback seguro
        query = str(message)

    # sanity-check de inicialização
    if rag_chain is None:
        await cl.Message(content="O sistema RAG não foi inicializado corretamente. Verifique os logs no terminal.").send()
        return

    session_id = cl.user_session.get("id", "default")
    print(f"Recebida pergunta (session_id={session_id}): {query}")

    try:
        response = await asyncio.to_thread(
            lambda: rag_chain.invoke({"input": query}, config={"configurable": {"session_id": session_id}})
        )
        print("Response recebido do rag_chain:", type(response), response.keys() if isinstance(response, dict) else "no dict")
    except Exception as e:
        print("Erro durante invocation do rag_chain:", e, flush=True)
        await cl.Message(content="Ocorreu um erro interno ao gerar a resposta (veja logs no terminal).").send()
        return

    if not response:
        print("Resposta vazia do rag_chain.")
        await cl.Message(content="Desculpe — não recebi resposta do mecanismo RAG.").send()
        return

    answer = response.get("answer", "Desculpe, não consegui processar sua pergunta.")
    msg = cl.Message(content=answer)

    # Se houver fontes, adiciona
    source_docs = []
    if isinstance(response, dict) and "context" in response and response["context"]:
        unique_sources = {getattr(doc.metadata, 'get', lambda k: None)('source') if hasattr(doc, "metadata") else doc.metadata.get('source') for doc in response["context"]}
        # fallback mais simples se linha acima falhar:
        try:
            unique_sources = {doc.metadata.get('source') for doc in response["context"]}
        except Exception:
            unique_sources = set()

        for source_path in unique_sources:
            if source_path:
                url, filename = generate_presigned_url(source_path)
                if url:
                    source_docs.append(f"[{filename}]({url})")

    if source_docs:
        msg.elements = [cl.Text(name="📚 Fontes Consultadas", content="\n".join(source_docs))]

    # envia a mensagem
    try:
        await msg.send()
    except Exception as e:
        print("Erro ao enviar mensagem via Chainlit:", e)
        # Tentar enviar fallback
        try:
            await cl.Message(content=answer).send()
        except Exception as e2:
            print("Erro fallback envio:", e2)

