"""
EHR Embeddings using Google Gemini API
"""
import google.generativeai as genai
import pandas as pd
import numpy as np
import time
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import pickle
import hashlib
from tqdm import tqdm

from config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """
    EHR data embedder using Google's Gemini API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini embedder
        
        Args:
            api_key: Google API key (if not provided, will use from config)
        """
        self.api_key = api_key or Config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        self.model = Config.EMBEDDING_MODEL
        self.batch_size = Config.BATCH_SIZE
        self.max_retries = Config.MAX_RETRIES
        self.cache_dir = Config.EMBEDDINGS_CACHE_DIR
        
        logger.info(f"Initialized GeminiEmbedder with model: {self.model}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache if exists"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            use_cache: Whether to use caching
            
        Returns:
            Embedding vector as numpy array
        """
        if use_cache:
            cache_key = self._get_cache_key(text)
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding
        
        for attempt in range(self.max_retries):
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embedding = np.array(result['embedding'])
                
                if use_cache:
                    self._save_to_cache(cache_key, embedding)
                
                return embedding
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e
    
    def embed_batch(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        embeddings = []
        
        for text in tqdm(texts, desc="Generating embeddings"):
            embedding = self.embed_text(text, use_cache=use_cache)
            embeddings.append(embedding)
            
            # Rate limiting
            time.sleep(0.1)
        
        return np.array(embeddings)
    
    def embed_ehr_data(self, 
                      data: pd.DataFrame, 
                      text_columns: List[str],
                      combine_columns: bool = True,
                      separator: str = " | ") -> pd.DataFrame:
        """
        Generate embeddings for EHR data
        
        Args:
            data: DataFrame containing EHR data
            text_columns: List of column names containing text data
            combine_columns: Whether to combine text columns into single text
            separator: Separator for combining columns
            
        Returns:
            DataFrame with embeddings added
        """
        logger.info(f"Processing {len(data)} EHR records")
        
        if combine_columns:
            # Combine text columns
            combined_texts = []
            for _, row in data.iterrows():
                text_parts = [str(row[col]) for col in text_columns if pd.notna(row[col])]
                combined_text = separator.join(text_parts)
                combined_texts.append(combined_text)
            
            # Generate embeddings
            embeddings = self.embed_batch(combined_texts)
            
            # Add embeddings to dataframe
            result_df = data.copy()
            for i, embedding in enumerate(embeddings):
                result_df.loc[result_df.index[i], 'embedding'] = embedding
                
        else:
            # Generate separate embeddings for each column
            result_df = data.copy()
            for col in text_columns:
                texts = data[col].fillna("").astype(str).tolist()
                embeddings = self.embed_batch(texts)
                
                for i, embedding in enumerate(embeddings):
                    result_df.loc[result_df.index[i], f'{col}_embedding'] = embedding
        
        logger.info("Embedding generation completed")
        return result_df


def preprocess_ehr_text(text: str) -> str:
    """
    Preprocess EHR text for better embeddings
    
    Args:
        text: Raw EHR text
        
    Returns:
        Preprocessed text
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and clean
    text = str(text).strip()
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # You can add more preprocessing steps here:
    # - Remove PHI (Personal Health Information) patterns
    # - Standardize medical terminology
    # - Handle special characters
    
    return text


def create_embedding_features(embeddings: np.ndarray, 
                            prefix: str = "emb_") -> pd.DataFrame:
    """
    Convert embeddings array to feature DataFrame
    
    Args:
        embeddings: Array of embeddings
        prefix: Prefix for column names
        
    Returns:
        DataFrame with embedding features
    """
    n_samples, n_features = embeddings.shape
    feature_names = [f"{prefix}{i}" for i in range(n_features)]
    
    return pd.DataFrame(embeddings, columns=feature_names) 