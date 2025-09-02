"""Shared test fixtures and configuration for the RAG system test suite."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Test data for courses
SAMPLE_COURSE_CONTENT = """Course Title: Introduction to Python Programming
Course Link: https://example.com/python-course
Course Instructor: Dr. Jane Smith

Lesson 1: Python Basics
Lesson Link: https://example.com/python-course/lesson-1
This lesson covers the fundamentals of Python programming including variables, data types, and basic operations. You'll learn how to write your first Python program and understand the syntax.

Lesson 2: Control Structures
Lesson Link: https://example.com/python-course/lesson-2
Learn about conditional statements, loops, and how to control the flow of your Python programs. This includes if statements, for loops, and while loops.

Lesson 3: Functions and Modules
Functions are reusable blocks of code that perform specific tasks. This lesson covers function definition, parameters, return values, and importing modules.
"""

SAMPLE_COURSE_CONTENT_2 = """Course Title: Advanced Machine Learning
Course Link: https://example.com/ml-course
Course Instructor: Prof. John Doe

Lesson 1: Neural Networks
Lesson Link: https://example.com/ml-course/lesson-1
Deep dive into neural network architectures, backpropagation, and gradient descent algorithms.

Lesson 2: Convolutional Networks
Learn about CNN architectures, pooling layers, and applications in computer vision.
"""


@pytest.fixture(scope="session")
def temp_docs_directory() -> Generator[str, None, None]:
    """Create a temporary directory with sample course documents for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        docs_path = Path(temp_dir)
        
        # Create sample course files
        course1_path = docs_path / "python_programming.txt"
        course1_path.write_text(SAMPLE_COURSE_CONTENT)
        
        course2_path = docs_path / "machine_learning.txt"
        course2_path.write_text(SAMPLE_COURSE_CONTENT_2)
        
        yield str(docs_path)


@pytest.fixture(scope="session")
def temp_chroma_directory() -> Generator[str, None, None]:
    """Create a temporary ChromaDB directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_config(temp_chroma_directory: str):
    """Mock configuration for testing."""
    config = Mock()
    config.anthropic_api_key = "test_api_key"
    config.anthropic_base_url = "https://api.anthropic.com"
    config.chroma_persist_directory = temp_chroma_directory
    config.embedding_model_name = "all-MiniLM-L6-v2"
    config.chunk_size = 800
    config.chunk_overlap = 100
    config.max_search_results = 5
    config.session_history_limit = 20
    return config


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "This is a test response from the AI."
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_query_data():
    """Sample query data for API testing."""
    return {
        "valid_query": {
            "query": "What are Python variables?",
            "session_id": "test_session_123"
        },
        "query_without_session": {
            "query": "How do neural networks work?"
        },
        "invalid_query": {
            "invalid_field": "this should fail"
        }
    }


@pytest.fixture
def sample_response_data():
    """Sample response data for testing."""
    return {
        "answer": "Python variables are containers that store data values.",
        "sources": [
            {"text": "Python Course - Lesson 1", "link": "https://example.com/python-course/lesson-1"},
            {"text": "Variables and data types section", "link": None}
        ],
        "session_id": "test_session_123"
    }


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for testing."""
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "This is a test answer",
        [
            {"text": "Source 1", "link": "https://example.com/source1"},
            {"text": "Source 2", "link": None}
        ]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to Python Programming", "Advanced Machine Learning"]
    }
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session.return_value = "new_session_123"
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI application with mocked dependencies."""
    # Import here to avoid circular imports and ensure mocks are in place
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Recreate the app structure without static file mounting
    app = FastAPI(title="Course Materials RAG System Test", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # API Endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        from fastapi import HTTPException
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            source_items = []
            for source in sources:
                if isinstance(source, dict) and 'text' in source:
                    source_items.append(SourceItem(
                        text=source['text'],
                        link=source.get('link')
                    ))
                elif isinstance(source, str):
                    source_items.append(SourceItem(text=source, link=None))
                else:
                    source_items.append(SourceItem(text=str(source), link=None))
            
            return QueryResponse(
                answer=answer,
                sources=source_items,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        from fastapi import HTTPException
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI application."""
    return TestClient(test_app)


@pytest.fixture
def course_analytics_data():
    """Sample course analytics data."""
    return {
        "total_courses": 3,
        "course_titles": [
            "Introduction to Python Programming",
            "Advanced Machine Learning", 
            "Web Development with FastAPI"
        ]
    }


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_vs = Mock()
    mock_vs.search_content.return_value = [
        {
            "course_title": "Introduction to Python Programming",
            "lesson_number": 1,
            "content": "Python variables are containers for storing data values.",
            "metadata": {"source": "python_course_lesson_1"}
        }
    ]
    mock_vs.get_all_courses.return_value = [
        {"title": "Introduction to Python Programming", "instructor": "Dr. Jane Smith"},
        {"title": "Advanced Machine Learning", "instructor": "Prof. John Doe"}
    ]
    return mock_vs


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress warnings during tests."""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")
    warnings.filterwarnings("ignore", category=UserWarning)