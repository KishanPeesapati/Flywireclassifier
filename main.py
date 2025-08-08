from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, List, Union
import logging
import time
import uvicorn
from contextlib import asynccontextmanager

# Import our classifier
from classifier import initialize_classifier, classify_user_query, classifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for query classification"""
    query: str = Field(
        ..., 
        min_length=3,  # Minimum 3 characters
        max_length=500,
        description="Credit card related query to classify (text only)"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query_content(cls, v: str) -> str:
        """Validate that query contains meaningful text"""
        v = v.strip()
        
        # Check if empty after stripping
        if not v:
            raise ValueError("Query cannot be empty or just whitespace")
        
        # Check if query is only numbers
        if v.isdigit():
            raise ValueError("Query cannot be only numbers")
        
        # Check if query has at least some alphabetic characters
        if not any(c.isalpha() for c in v):
            raise ValueError("Query must contain at least some letters")
        
        # Check for minimum word count (at least 1 real word)
        words = v.split()
        meaningful_words = [word for word in words if len(word) > 1 and any(c.isalpha() for c in word)]
        if len(meaningful_words) < 1:
            raise ValueError("Query must contain at least one meaningful word")
        
        return v

class ClassificationResponse(BaseModel):
    """Response model for classification results"""
    intent_label: int = Field(..., description="Predicted intent label")
    intent_description: str = Field(..., description="Human readable intent description")
    confidence: float = Field(..., description="Classification confidence score (0.0-1.0)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    is_confident: bool = Field(..., description="Whether prediction meets confidence threshold")
    
class LowConfidenceResponse(BaseModel):
    """Response model for low confidence classifications"""
    message: str = Field(..., description="Explanation for low confidence")
    intent_label: Optional[int] = Field(None, description="Best guess intent label")
    intent_description: Optional[str] = Field(None, description="Best guess intent description")
    confidence: float = Field(..., description="Classification confidence score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for user")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    model_loaded: bool

# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown events.
    Initialize the classifier during startup.
    """
    logger.info("Starting up Credit Card Intent Classification Service...")
    try:
        # Initialize classifier during startup
        initialize_classifier()
        logger.info("Classifier initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {str(e)}")
        raise
    
    yield
    
    # Cleanup code can go here if needed
    logger.info("Shutting down Credit Card Intent Classification Service...")

# Create FastAPI application
app = FastAPI(
    title="Credit Card Intent Classification API",
    description="Microservice for classifying credit card related customer queries into intent categories",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic service information"""
    return {
        "service": "Credit Card Intent Classification API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = classifier.model is not None
    
    if model_loaded:
        return HealthResponse(
            status="healthy",
            message="Service is running and model is loaded",
            model_loaded=True
        )
    else:
        return HealthResponse(
            status="unhealthy",
            message="Service is running but model is not loaded",
            model_loaded=False
        )

# Configuration
CONFIDENCE_THRESHOLD = 0.6  # 60% confidence threshold

@app.post("/classify", response_model=Union[ClassificationResponse, LowConfidenceResponse])
async def classify_query(request: QueryRequest):
    """
    Classify a credit card related query into intent categories.
    
    Args:
        request: QueryRequest containing the query to classify
        
    Returns:
        ClassificationResponse with intent label and confidence, or
        LowConfidenceResponse if confidence is below threshold
        
    Raises:
        HTTPException: If classification fails
    """
    start_time = time.time()
    
    try:
        # Query is already validated by Pydantic
        query = request.query.strip()
        
        logger.info(f"Classifying query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        # Perform classification
        result = classify_user_query(query)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        confidence = result.get("confidence", 0.0)
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(
                f"Low confidence prediction: confidence={confidence:.3f}, "
                f"threshold={CONFIDENCE_THRESHOLD}, query='{query[:50]}'"
            )
            
            return LowConfidenceResponse(
                message=f"I'm not confident about this classification (confidence: {confidence:.1%}). Please rephrase your question or contact support.",
                intent_label=result["intent_label"],
                intent_description=result["intent_description"],
                confidence=confidence,
                processing_time_ms=round(processing_time, 2),
                suggestions=[
                    "Try rephrasing your question with more details",
                    "Use specific credit card related terms",
                    "Contact customer support for assistance"
                ]
            )
        
        response = ClassificationResponse(
            intent_label=result["intent_label"],
            intent_description=result["intent_description"],
            confidence=confidence,
            processing_time_ms=round(processing_time, 2),
            is_confident=True
        )
        
        logger.info(
            f"High confidence classification: intent={result['intent_label']}, "
            f"confidence={confidence:.3f}, time={processing_time:.2f}ms"
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during classification"
        )

@app.get("/intents", response_model=Dict[int, str])
async def get_available_intents():
    """
    Get all available intent labels and their descriptions.
    
    Returns:
        Dictionary mapping intent labels to descriptions
    """
    try:
        if classifier.intent_descriptions:
            return classifier.intent_descriptions
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Intent descriptions not available - classifier not properly initialized"
            )
    except Exception as e:
        logger.error(f"Error retrieving intents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving available intents"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "The requested endpoint does not exist"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return {"error": "Internal server error", "message": "An unexpected error occurred"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )