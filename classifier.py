import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Tuple
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditCardIntentClassifier:
    """
    Credit card query intent classifier using sentence transformers and cosine similarity.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the classifier with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.labeled_queries = None
        self.query_embeddings = None
        self.intent_labels = None
        self.intent_descriptions = {}
        
    def load_data(self, intent_labels_path: str, queries_path: str) -> None:
        """
        Load intent labels and labeled queries from Excel files.
        
        Args:
            intent_labels_path: Path to intent labels Excel file
            queries_path: Path to labeled queries Excel file
        """
        try:
            # Load intent labels and descriptions from Excel
            intent_df = pd.read_excel(intent_labels_path)
            self.intent_descriptions = dict(zip(intent_df['label'], intent_df['intent_description']))
            logger.info(f"Loaded {len(self.intent_descriptions)} intent categories")
            
            # Load labeled queries from Excel
            queries_df = pd.read_excel(queries_path)
            self.labeled_queries = queries_df['query'].tolist()
            self.intent_labels = queries_df['intent_label'].tolist()
            
            logger.info(f"Loaded {len(self.labeled_queries)} labeled queries")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def initialize_model(self) -> None:
        """
        Initialize the sentence transformer model.
        """
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def create_embeddings(self) -> None:
        """
        Create embeddings for all labeled queries.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        if self.labeled_queries is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        try:
            logger.info("Creating embeddings for labeled queries...")
            self.query_embeddings = self.model.encode(
                self.labeled_queries, 
                convert_to_tensor=False,
                show_progress_bar=True
            )
            logger.info(f"Created embeddings with shape: {self.query_embeddings.shape}")
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def classify_query(self, query: str, return_confidence: bool = False) -> Dict:
        """
        Classify a single query and return the predicted intent label.
        
        Args:
            query: Input query string to classify
            return_confidence: Whether to return confidence score
            
        Returns:
            Dictionary containing intent_label and optionally confidence score
        """
        if self.model is None or self.query_embeddings is None:
            raise ValueError("Classifier not properly initialized")
        
        try:
            # Create embedding for input query
            query_embedding = self.model.encode([query], convert_to_tensor=False)
            
            # Calculate cosine similarities with all labeled queries
            similarities = cosine_similarity(query_embedding, self.query_embeddings)[0]
            
            # Find the most similar query
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            predicted_label = self.intent_labels[best_match_idx]
            
            # Get intent description
            intent_description = self.intent_descriptions.get(predicted_label, "Unknown")
            
            logger.info(f"Query classified as intent {predicted_label} ({intent_description}) with confidence {best_similarity:.3f}")
            
            result = {
                "intent_label": int(predicted_label),
                "intent_description": intent_description
            }
            
            if return_confidence:
                result["confidence"] = float(best_similarity)
                result["matched_query"] = self.labeled_queries[best_match_idx]
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying query: {str(e)}")
            raise
    
    def get_intent_info(self, label: int) -> str:
        """
        Get intent description for a given label.
        
        Args:
            label: Intent label number
            
        Returns:
            Intent description string
        """
        return self.intent_descriptions.get(label, "Unknown intent")
    
    def setup(self, intent_labels_path: str = "data_files/credit_card_intent_labels.xlsx", 
              queries_path: str = "data_files/credit_card_queries_labeled.xlsx") -> None:
        """
        Complete setup of the classifier - load data, initialize model, create embeddings.
        
        Args:
            intent_labels_path: Path to intent labels CSV file
            queries_path: Path to labeled queries CSV file
        """
        logger.info("Setting up Credit Card Intent Classifier...")
        
        # Check if files exist
        if not os.path.exists(intent_labels_path):
            raise FileNotFoundError(f"Intent labels file not found: {intent_labels_path}")
        if not os.path.exists(queries_path):
            raise FileNotFoundError(f"Queries file not found: {queries_path}")
        
        self.load_data(intent_labels_path, queries_path)
        self.initialize_model()
        self.create_embeddings()
        
        logger.info("Classifier setup complete and ready for inference!")


# Create a global classifier instance
classifier = CreditCardIntentClassifier()

def initialize_classifier():
    """
    Initialize the global classifier instance.
    This function will be called when the FastAPI app starts.
    """
    classifier.setup()
    return classifier

def classify_user_query(query: str) -> Dict:
    """
    Classify a user query and return the result.
    
    Args:
        query: User input query
        
    Returns:
        Classification result dictionary
    """
    if classifier.model is None:
        raise ValueError("Classifier not initialized")
    
    return classifier.classify_query(query, return_confidence=True)