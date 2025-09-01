import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import tempfile
import shutil

# Add the backend directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk


class TestRAGSystemIntegration:
    """Integration tests for the complete RAG system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration"""
        config = Config()
        config.CHROMA_PATH = os.path.join(temp_dir, "chroma_test")
        config.MAX_RESULTS = 5  # Fix the critical issue for testing
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "test-model"
        config.ANTHROPIC_BASE_URL = "https://test-api.com"
        return config
    
    @pytest.fixture
    def test_config_with_zero_results(self, temp_dir):
        """Create test configuration with MAX_RESULTS=0 to reproduce the bug"""
        config = Config()
        config.CHROMA_PATH = os.path.join(temp_dir, "chroma_test")
        config.MAX_RESULTS = 0  # This reproduces the critical bug
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "test-model"
        config.ANTHROPIC_BASE_URL = "https://test-api.com"
        return config
    
    @pytest.fixture
    def mock_anthropic_response(self):
        """Create mock Anthropic response"""
        mock_content = Mock()
        mock_content.text = "Based on the course materials, machine learning is..."
        
        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"
        
        return mock_response
    
    @pytest.fixture
    def mock_tool_use_response(self):
        """Create mock Anthropic response with tool use"""
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "machine learning"}
        
        mock_response = Mock()
        mock_response.content = [mock_tool_content]
        mock_response.stop_reason = "tool_use"
        
        return mock_response
    
    @pytest.fixture
    def sample_course_data(self):
        """Create sample course data for testing"""
        course = Course(
            title="Introduction to Machine Learning",
            course_link="https://example.com/ml-course",
            instructor="Dr. Jane Smith",
            lessons=[
                Lesson(lesson_number=1, title="What is ML?", lesson_link="https://example.com/lesson-1"),
                Lesson(lesson_number=2, title="Linear Regression", lesson_link="https://example.com/lesson-2")
            ]
        )
        
        chunks = [
            CourseChunk(
                content="Machine learning is a subset of artificial intelligence.",
                course_title="Introduction to Machine Learning",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Linear regression is used for predictive modeling.",
                course_title="Introduction to Machine Learning",
                lesson_number=2,
                chunk_index=1
            )
        ]
        
        return course, chunks
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('ai_generator.anthropic.Anthropic')
    def test_rag_system_initialization(self, mock_anthropic, mock_embedding, mock_chroma, test_config):
        """Test RAG system initialization"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        rag_system = RAGSystem(test_config)
        
        # Check all components are initialized
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None
        
        # Check tools are registered
        assert "search_course_content" in rag_system.tool_manager.tools
        assert "get_course_outline" in rag_system.tool_manager.tools
    
    @patch('rag_system.chromadb.PersistentClient')
    @patch('rag_system.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') 
    @patch('rag_system.anthropic.Anthropic')
    def test_query_with_successful_search(self, mock_anthropic, mock_embedding, mock_chroma, 
                                         test_config, mock_tool_use_response, mock_anthropic_response):
        """Test successful query processing with search"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        # Mock search results
        mock_collection.query.return_value = {
            'documents': [['Machine learning is a subset of AI']],
            'metadatas': [[{'course_title': 'ML Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        # Mock Anthropic responses
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_anthropic_response
        ]
        
        rag_system = RAGSystem(test_config)
        
        # Execute query
        response, sources = rag_system.query("What is machine learning?")
        
        # Check response
        assert response == "Based on the course materials, machine learning is..."
        assert len(sources) > 0
        
        # Check that Anthropic was called twice (tool use + final response)
        assert mock_anthropic_client.messages.create.call_count == 2
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_with_max_results_zero_bug(self, mock_anthropic, mock_embedding, mock_chroma,
                                            test_config_with_zero_results, mock_tool_use_response):
        """Test that MAX_RESULTS=0 causes search to fail (reproducing the bug)"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        # With MAX_RESULTS=0, ChromaDB returns empty results
        mock_collection.query.return_value = {
            'documents': [[]],  # Empty due to n_results=0
            'metadatas': [[]],
            'distances': [[]]
        }
        
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = mock_tool_use_response
        
        rag_system = RAGSystem(test_config_with_zero_results)
        
        # Execute query - this should demonstrate the bug
        response, sources = rag_system.query("What is machine learning?")
        
        # With MAX_RESULTS=0, the search tool should return "No relevant content found"
        # Check that ChromaDB was called with n_results=0
        search_call = mock_collection.query.call_args
        if search_call:
            assert search_call[1]["n_results"] == 0  # This is the bug!
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_without_tools(self, mock_anthropic, mock_embedding, mock_chroma, 
                                test_config, mock_anthropic_response):
        """Test query that doesn't trigger tool use"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = mock_anthropic_response
        
        rag_system = RAGSystem(test_config)
        
        # Execute a general query that shouldn't trigger search
        response, sources = rag_system.query("Hello, how are you?")
        
        # Should get direct response without tool use
        assert response == "Based on the course materials, machine learning is..."
        assert len(sources) == 0  # No sources from search
        
        # Only one API call should be made
        assert mock_anthropic_client.messages.create.call_count == 1
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_with_session_management(self, mock_anthropic, mock_embedding, mock_chroma,
                                          test_config, mock_anthropic_response):
        """Test query processing with session management"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = mock_anthropic_response
        
        rag_system = RAGSystem(test_config)
        
        # First query creates session
        response1, sources1 = rag_system.query("First question", session_id="test-session")
        
        # Second query should use existing session with history
        response2, sources2 = rag_system.query("Follow-up question", session_id="test-session")
        
        # Both responses should be received
        assert response1 == "Based on the course materials, machine learning is..."
        assert response2 == "Based on the course materials, machine learning is..."
        
        # Check that history was included in second call
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        system_prompt = second_call_args[1]["system"]
        assert "Previous conversation:" in system_prompt
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('ai_generator.anthropic.Anthropic')
    def test_search_error_handling(self, mock_anthropic, mock_embedding, mock_chroma,
                                  test_config, mock_tool_use_response):
        """Test error handling when search fails"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        # Make ChromaDB query raise an exception
        mock_collection.query.side_effect = Exception("Database connection failed")
        
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = mock_tool_use_response
        
        rag_system = RAGSystem(test_config)
        
        # This should not crash but handle the error gracefully
        try:
            response, sources = rag_system.query("What is machine learning?")
            # The system should handle the error without crashing
        except Exception as e:
            pytest.fail(f"RAG system should handle search errors gracefully: {e}")
    
    @patch('rag_system.chromadb.PersistentClient') 
    @patch('rag_system.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('rag_system.anthropic.Anthropic')
    def test_add_course_document_integration(self, mock_anthropic, mock_embedding, mock_chroma,
                                           test_config, sample_course_data):
        """Test adding course document with full integration"""
        course, chunks = sample_course_data
        
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        
        # Create a mock document file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""Course Title: Introduction to Machine Learning
Course Link: https://example.com/ml-course
Course Instructor: Dr. Jane Smith

Lesson 1: What is Machine Learning?
Machine learning is a subset of artificial intelligence.

Lesson 2: Linear Regression
Linear regression is used for predictive modeling.""")
            temp_file = f.name
        
        try:
            rag_system = RAGSystem(test_config)
            
            # Mock the document processor to return our sample data
            with patch.object(rag_system.document_processor, 'process_course_document') as mock_processor:
                mock_processor.return_value = (course, chunks)
                
                result_course, chunk_count = rag_system.add_course_document(temp_file)
                
                # Check results
                assert result_course.title == "Introduction to Machine Learning"
                assert chunk_count == 2
                
                # Check that vector store methods were called
                assert mock_collection.add.call_count == 2  # Once for metadata, once for content
        finally:
            os.unlink(temp_file)
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('ai_generator.anthropic.Anthropic')
    def test_get_course_analytics(self, mock_anthropic, mock_embedding, mock_chroma, test_config):
        """Test course analytics functionality"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        # Mock course data
        mock_collection.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        
        rag_system = RAGSystem(test_config)
        
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 3
        assert analytics["course_titles"] == ['Course 1', 'Course 2', 'Course 3']
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('ai_generator.anthropic.Anthropic')
    def test_anthropic_api_integration(self, mock_anthropic, mock_embedding, mock_chroma, test_config):
        """Test that Anthropic client is properly initialized with config values"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        rag_system = RAGSystem(test_config)
        
        # Check that Anthropic client was initialized with correct parameters
        mock_anthropic.assert_called_once_with(
            api_key="test-key",
            base_url="https://test-api.com"
        )
        
        # Check that AI generator has correct model
        assert rag_system.ai_generator.model == "test-model"
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_manager_integration(self, mock_anthropic, mock_embedding, mock_chroma, test_config):
        """Test tool manager integration and tool definitions"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        
        rag_system = RAGSystem(test_config)
        
        # Check tool definitions
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        
        assert len(tool_definitions) == 2  # search_course_content and get_course_outline
        
        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
        
        # Check that search tool has vector store reference
        assert rag_system.search_tool.store is rag_system.vector_store
        assert rag_system.outline_tool.store is rag_system.vector_store