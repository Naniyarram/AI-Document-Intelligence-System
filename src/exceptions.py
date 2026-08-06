# src/exceptions.py
"""
Custom exception hierarchy for the AI Document Intelligence System.
"""

class DocumentIntelligenceError(Exception):
    """Base exception for all system-related errors."""
    pass

class DocumentParseError(DocumentIntelligenceError):
    """Raised when parsing or loading a document page fails."""
    pass

class EmbeddingError(DocumentIntelligenceError):
    """Raised when generating embeddings or interacting with the vector database fails."""
    pass

class RetrievalError(DocumentIntelligenceError):
    """Raised when retrieval (dense, sparse, reranking) fails."""
    pass

class LLMGenerationError(DocumentIntelligenceError):
    """Raised when communicating with the LLM API fails."""
    pass

class VLMProcessingError(DocumentIntelligenceError):
    """Raised when visual page extraction via the VLM fails."""
    pass

class ConfigurationError(DocumentIntelligenceError):
    """Raised when invalid configurations are encountered."""
    pass
