import os
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from chromadb.config import Settings
from chromadb import Client
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from openai import OpenAI
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

# Get base directory
BASE_DIR = Path(__file__).resolve().parent

# Configuration
EMBEDDING_MODEL = "dangvantuan/vietnamese-embedding"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
PERSIST_DIRECTORY = str(BASE_DIR / "chroma_db")

QB_TEXT_FILE_PATH = str(BASE_DIR / "data" / "Quang_Binh_Dialect_Rules.txt")
QB_SYSTEM_MESSAGE = """
You are an expert in the Quảng Bình dialect. Given the appropriate context for dialect rules, 
your task is to rephrase standard Vietnamese sentences into natural, idiomatic Quảng Bình dialect. 
Preserve the meaning and tone. Reply only with the transformed sentence unless asked otherwise.
"""
QB_COLLECTION_NAME = "qb_dialect_rules"

ANCIENT_TEXT_FILE_PATH = str(BASE_DIR / "data" / "ancient_vietnamese_context.txt")
ANCIENT_SYSTEM_MESSAGE = """
You are an expert in the Ancient Vietnamese language. Given the relevant data of the Ancient Vietnamese language, 
your task is to rephrase standard Vietnamese sentences into natural Ancient Vietnamese language. 
Preserve the meaning and tone. Reply only with the transformed sentence unless asked otherwise.
"""
ANCIENT_COLLECTION_NAME = "ancient_vietnamese_data"

def get_api_model(model_id: str):
    if model_id == "qb_deepseek_api":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        model_name = os.getenv("DEEPSEEK_MODEL")
        sys_message = QB_SYSTEM_MESSAGE
        data_path = QB_TEXT_FILE_PATH
        collection_name = QB_COLLECTION_NAME
        return QuangBinhDialectTranslator(api_key, base_url, sys_message,
                                         data_path, collection_name, model_name)
    elif model_id == "qb_openai_api":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = None
        model_name = os.getenv("OPENAI_MODEL")
        sys_message = QB_SYSTEM_MESSAGE
        data_path = QB_TEXT_FILE_PATH
        collection_name = QB_COLLECTION_NAME
        return QuangBinhDialectTranslator(api_key, base_url, sys_message,
                                         data_path, collection_name, model_name)
    elif model_id == "ancient_deepseek_api":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        model_name = os.getenv("DEEPSEEK_MODEL")
        sys_message = ANCIENT_SYSTEM_MESSAGE
        data_path = ANCIENT_TEXT_FILE_PATH
        collection_name = ANCIENT_COLLECTION_NAME
        return QuangBinhDialectTranslator(api_key, base_url, sys_message,
                                         data_path, collection_name, model_name)
    elif model_id == "ancient_openai_api":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = None
        model_name = os.getenv("OPENAI_MODEL")
        sys_message = ANCIENT_SYSTEM_MESSAGE
        data_path = ANCIENT_TEXT_FILE_PATH
        collection_name = ANCIENT_COLLECTION_NAME
        return QuangBinhDialectTranslator(api_key, base_url, sys_message,
                                         data_path, collection_name, model_name)

class QuangBinhDialectTranslator:
    def __init__(self, api_key: str, base_url: str, sys_message: str, 
                 data_path: str, collection_name: str, model_name: str):
        if base_url is not None:
            self.model_client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            self.model_client = OpenAI(
                api_key=api_key,
            )
        self.data_path = data_path
        self.collection_name = collection_name
        self.sys_message = sys_message
        self.model_name = model_name
        
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = Client(Settings(persist_directory=PERSIST_DIRECTORY))
        self.vector_store = self._initialize_vector_store()
        self.retriever = self.vector_store.as_retriever()
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize Chroma vector store with embeddings"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load and split document
        loader = TextLoader(self.data_path, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings with tokenization
        tokenized_texts = [tokenize(text) for text in texts]
        embeddings = self.embedding_model.encode(tokenized_texts)
        
        # Create/reuse Chroma collection
        try:
            collection = self.client.get_collection(self.collection_name)
        except:
            collection = self.client.create_collection(self.collection_name)
        
        # Add documents to collection
        # for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        #     collection.add(
        #         documents=[chunk.page_content],
        #         metadatas=[{'source': self.data_path}],
        #         embeddings=[embedding.tolist()],
        #         ids=[str(idx)]
        #     )
        
        # Add documents to collection in batch style
        batch_size = 32
        total_items = len(chunks)
        
        # Convert embeddings to list of lists if they're numpy arrays
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        # Process in batches
        for batch_start in range(0, total_items, batch_size):
            batch_end = min(batch_start + batch_size, total_items)
            
            batch_ids = [str(i) for i in range(batch_start, batch_end)]
            batch_documents = texts[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]
            batch_metadatas = [{'id': i} for i in range(batch_start, batch_end)]
            
            # Add batch to Chroma
            collection.add(
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        return Chroma(
            collection_name=self.collection_name,
            client=self.client,
            embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )

    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context for a query"""
        results = self.retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in results])
    
    def translate_sentence(self, input_sentence: str) -> tuple:
        """
        Translate a standard Vietnamese sentence to Quang Binh dialect
        Returns tuple: (context, translated_sentence)
        """
        context = self.retrieve_context(input_sentence)
        prompt = f"### Input:\n{input_sentence}\n\nContext:\n{context}\n### Output:"
        
        response = self.model_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.sys_message},
                {"role": "user", "content": prompt}
            ],
            temperature=1.3,
            stream=False
        )
        
        return context, response.choices[0].message.content.strip()