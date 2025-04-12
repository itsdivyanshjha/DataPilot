import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import json
import os
import io
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
import torch
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGManager:
    def __init__(self):
        self.vector_store = None
        self.current_dataset_info = None
        self.embeddings = None
        self.cached_embeddings = {}
        self.current_dataset_hash = None
        self.vector_store_path = "vector_store"
        
        # Initialize sentence transformer for embeddings
        logger.info("Initializing Sentence Transformer for embeddings...")
        try:
            self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence Transformer initialized successfully")
            
            # Try to load existing vector store
            if os.path.exists(self.vector_store_path):
                try:
                    self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
                    logger.info("Loaded existing vector store")
                except Exception as e:
                    logger.warning(f"Failed to load existing vector store: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to initialize Sentence Transformer: {str(e)}")
            raise

    def _get_dataset_hash(self, df: pd.DataFrame) -> str:
        """Generate a hash for the dataset to track changes"""
        return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using sentence transformer with caching"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.cached_embeddings:
            return self.cached_embeddings[cache_key]
            
        try:
            logger.info(f"Generating embedding for text: {text[:50]}...")
            embedding = self.embeddings.encode(text, convert_to_numpy=True).tolist()
            self.cached_embeddings[cache_key] = embedding
            logger.info("Successfully generated embedding")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise

    def create_knowledge_base(self, df: pd.DataFrame) -> None:
        """
        Create a knowledge base from the dataset using FAISS vector store.
        Only recreate if the dataset has changed.
        """
        try:
            dataset_hash = self._get_dataset_hash(df)
            
            # Check if we need to recreate the knowledge base
            if (self.vector_store is not None and 
                self.current_dataset_hash == dataset_hash):
                logger.info("Using existing knowledge base")
                return
                
            logger.info("Starting knowledge base creation...")
            self.current_dataset_hash = dataset_hash
            
            # Store dataset information
            logger.info("Storing dataset information...")
            self.current_dataset_info = {
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'shape': df.shape,
                'summary_stats': df.describe().to_dict() if not df.empty else {}
            }
            
            # Create documents from the dataset
            logger.info("Creating documents from dataset...")
            documents = []
            
            # Add dataset overview
            overview_doc = f"Dataset Overview:\n- Shape: {df.shape}\n- Columns: {', '.join(df.columns)}\n"
            overview_doc += f"- Total Records: {len(df)}\n- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            documents.append(Document(page_content=overview_doc, metadata={'type': 'dataset_overview'}))
            
            # Add detailed column information
            for col in df.columns:
                try:
                    col_info = f"Column '{col}' Analysis:\n"
                    col_info += f"- Data Type: {df[col].dtype}\n"
                    col_info += f"- Unique Values: {df[col].nunique()}\n"
                    
                    if df[col].dtype in ['int64', 'float64']:
                        col_info += f"- Range: {df[col].min()} to {df[col].max()}\n"
                        col_info += f"- Mean: {df[col].mean():.2f}\n"
                        col_info += f"- Median: {df[col].median():.2f}\n"
                        col_info += f"- Standard Deviation: {df[col].std():.2f}\n"
                    elif df[col].dtype == 'object':
                        col_info += f"- Most Common Values:\n"
                        top_values = df[col].value_counts().head(5)
                        for val, count in top_values.items():
                            col_info += f"  * {val}: {count} occurrences\n"
                    
                    col_info += f"- Missing Values: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.2f}%)"
                    documents.append(Document(page_content=col_info, metadata={'type': 'column_analysis', 'column': col}))
                except Exception as e:
                    logger.warning(f"Error processing column {col}: {str(e)}")
            
            # Add correlation information for numerical columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = df[numeric_cols].corr()
                    corr_doc = "Numerical Column Correlations:\n"
                    for i, col1 in enumerate(numeric_cols):
                        for col2 in numeric_cols[i+1:]:
                            corr = corr_matrix.loc[col1, col2]
                            if abs(corr) > 0.5:  # Only include significant correlations
                                corr_doc += f"- {col1} and {col2}: {corr:.2f}\n"
                    documents.append(Document(page_content=corr_doc, metadata={'type': 'correlations'}))
                except Exception as e:
                    logger.warning(f"Error calculating correlations: {str(e)}")
            
            # Add data samples with reduced size but more context
            sample_size = min(20, len(df))
            logger.info(f"Adding {sample_size} data samples...")
            for idx, row in df.sample(n=sample_size).iterrows():
                try:
                    row_doc = f"Sample Row {idx}:\n"
                    for col in df.columns:
                        row_doc += f"- {col}: {row[col]}\n"
                    documents.append(Document(page_content=row_doc, metadata={'type': 'data_sample', 'row_index': idx}))
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {str(e)}")
            
            # Create text splitter with smaller chunks
            logger.info("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=20,
                length_function=len
            )
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Create vector store with error handling
            logger.info("Creating vector store...")
            try:
                # Create a custom embedding function that uses our embed_text method
                def embed_docs(texts: List[str]) -> List[List[float]]:
                    return [self.embed_text(text) for text in texts]
                
                # Create FAISS index manually
                embeddings = embed_docs([doc.page_content for doc in chunks])
                dimension = len(embeddings[0])
                index = faiss.IndexFlatL2(dimension)
                index.add(np.array(embeddings, dtype=np.float32))
                
                # Store the documents and create the vector store
                self.vector_store = FAISS(
                    self.embed_text,
                    index,
                    chunks,
                    {},
                    normalize_L2=True
                )
                
                # Save the vector store
                os.makedirs(self.vector_store_path, exist_ok=True)
                self.vector_store.save_local(self.vector_store_path)
                logger.info("Vector store created and saved successfully")
            except Exception as e:
                logger.error(f"Error creating vector store: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error in create_knowledge_base: {str(e)}")
            # Clear any partial state
            self.vector_store = None
            self.current_dataset_info = None
            self.current_dataset_hash = None
            raise ValueError(f"Error creating knowledge base: {str(e)}")
        
    def get_relevant_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a given query.
        """
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return []
            
            logger.info(f"Searching for context with query: {query}")
            # Search for relevant documents
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Format context
            context = []
            for doc in docs:
                context.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
            
            logger.info(f"Found {len(context)} relevant documents")
            return context
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def get_dataset_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current dataset.
        """
        return self.current_dataset_info
    
    def is_initialized(self) -> bool:
        """
        Check if the RAG system is initialized with a dataset.
        """
        return self.vector_store is not None and self.current_dataset_info is not None
    
    def clear(self) -> None:
        """
        Clear the current knowledge base.
        """
        try:
            logger.info("Clearing knowledge base...")
            self.vector_store = None
            self.current_dataset_info = None
            self.current_dataset_hash = None
            self.cached_embeddings.clear()
            
            # Remove saved vector store
            if os.path.exists(self.vector_store_path):
                import shutil
                shutil.rmtree(self.vector_store_path)
                
            logger.info("Knowledge base cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            raise 