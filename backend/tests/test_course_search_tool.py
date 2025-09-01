import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add the backend directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


class TestCourseSearchTool:
    """Test suite for CourseSearchTool functionality"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing"""
        mock_store = Mock()
        mock_store.get_course_link.return_value = "https://example.com/course"
        mock_store.get_lesson_link.return_value = "https://example.com/lesson-1"
        return mock_store
    
    @pytest.fixture
    def course_search_tool(self, mock_vector_store):
        """Create CourseSearchTool instance with mocked vector store"""
        return CourseSearchTool(mock_vector_store)
    
    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results for testing"""
        return SearchResults(
            documents=["Machine learning is a subset of AI", "Linear regression is fundamental"],
            metadata=[
                {"course_title": "Introduction to ML", "lesson_number": 1},
                {"course_title": "Introduction to ML", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
    
    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is correctly formatted"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_execute_successful_search(self, course_search_tool, mock_vector_store, sample_search_results):
        """Test successful search execution with results"""
        mock_vector_store.search.return_value = sample_search_results
        
        result = course_search_tool.execute("machine learning")
        
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )
        
        assert "[Introduction to ML - Lesson 1]" in result
        assert "[Introduction to ML - Lesson 2]" in result
        assert "Machine learning is a subset of AI" in result
        assert "Linear regression is fundamental" in result
    
    def test_execute_with_course_filter(self, course_search_tool, mock_vector_store, sample_search_results):
        """Test search with course name filter"""
        mock_vector_store.search.return_value = sample_search_results
        
        result = course_search_tool.execute("algorithms", course_name="ML Course")
        
        mock_vector_store.search.assert_called_once_with(
            query="algorithms",
            course_name="ML Course",
            lesson_number=None
        )
        assert "Introduction to ML" in result
    
    def test_execute_with_lesson_filter(self, course_search_tool, mock_vector_store, sample_search_results):
        """Test search with lesson number filter"""
        mock_vector_store.search.return_value = sample_search_results
        
        result = course_search_tool.execute("regression", lesson_number=2)
        
        mock_vector_store.search.assert_called_once_with(
            query="regression",
            course_name=None,
            lesson_number=2
        )
        assert "Linear regression is fundamental" in result
    
    def test_execute_with_both_filters(self, course_search_tool, mock_vector_store, sample_search_results):
        """Test search with both course and lesson filters"""
        mock_vector_store.search.return_value = sample_search_results
        
        result = course_search_tool.execute("AI", course_name="ML Course", lesson_number=1)
        
        mock_vector_store.search.assert_called_once_with(
            query="AI",
            course_name="ML Course",
            lesson_number=1
        )
        assert result is not None
    
    def test_execute_with_error(self, course_search_tool, mock_vector_store):
        """Test handling of search errors"""
        error_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        mock_vector_store.search.return_value = error_results
        
        result = course_search_tool.execute("test query")
        
        assert result == "Database connection failed"
    
    def test_execute_empty_results(self, course_search_tool, mock_vector_store):
        """Test handling of empty search results"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = course_search_tool.execute("nonexistent topic")
        
        assert result == "No relevant content found."
    
    def test_execute_empty_results_with_course_filter(self, course_search_tool, mock_vector_store):
        """Test empty results message includes course filter info"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = course_search_tool.execute("test", course_name="Physics")
        
        assert "No relevant content found in course 'Physics'." == result
    
    def test_execute_empty_results_with_lesson_filter(self, course_search_tool, mock_vector_store):
        """Test empty results message includes lesson filter info"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = course_search_tool.execute("test", lesson_number=3)
        
        assert "No relevant content found in lesson 3." == result
    
    def test_execute_empty_results_with_both_filters(self, course_search_tool, mock_vector_store):
        """Test empty results message includes both filters"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = course_search_tool.execute("test", course_name="Physics", lesson_number=2)
        
        assert "No relevant content found in course 'Physics' in lesson 2." == result
    
    def test_source_tracking(self, course_search_tool, mock_vector_store, sample_search_results):
        """Test that sources are properly tracked with links"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson-1",
            "https://example.com/lesson-2"
        ]
        
        # Execute search
        course_search_tool.execute("machine learning")
        
        # Check that sources are tracked
        assert len(course_search_tool.last_sources) == 2
        
        source1 = course_search_tool.last_sources[0]
        assert source1["text"] == "Introduction to ML - Lesson 1"
        assert source1["link"] == "https://example.com/lesson-1"
        
        source2 = course_search_tool.last_sources[1]
        assert source2["text"] == "Introduction to ML - Lesson 2"
        assert source2["link"] == "https://example.com/lesson-2"
    
    def test_source_fallback_to_course_link(self, course_search_tool, mock_vector_store, sample_search_results):
        """Test fallback to course link when lesson link not available"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None  # No lesson link
        mock_vector_store.get_course_link.return_value = "https://example.com/course"
        
        course_search_tool.execute("test query")
        
        # Should fall back to course link
        assert len(course_search_tool.last_sources) == 2
        for source in course_search_tool.last_sources:
            assert source["link"] == "https://example.com/course"
    
    def test_format_results_without_lesson_number(self, course_search_tool, mock_vector_store):
        """Test result formatting when lesson number is None"""
        results = SearchResults(
            documents=["Course introduction content"],
            metadata=[{"course_title": "Python Basics", "lesson_number": None}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = results
        mock_vector_store.get_course_link.return_value = "https://example.com/python"
        
        result = course_search_tool.execute("introduction")
        
        assert "[Python Basics]" in result  # No lesson number in header
        assert "Course introduction content" in result
        assert len(course_search_tool.last_sources) == 1
        assert course_search_tool.last_sources[0]["text"] == "Python Basics"


class TestToolManager:
    """Test suite for ToolManager functionality"""
    
    @pytest.fixture
    def tool_manager(self):
        """Create ToolManager instance"""
        return ToolManager()
    
    @pytest.fixture
    def mock_course_search_tool(self):
        """Create a mock CourseSearchTool"""
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {
            "name": "search_course_content",
            "description": "Search for course content"
        }
        mock_tool.execute.return_value = "Test search result"
        mock_tool.last_sources = [{"text": "Test Source", "link": "http://test.com"}]
        return mock_tool
    
    def test_register_tool(self, tool_manager, mock_course_search_tool):
        """Test tool registration"""
        tool_manager.register_tool(mock_course_search_tool)
        
        assert "search_course_content" in tool_manager.tools
        assert tool_manager.tools["search_course_content"] == mock_course_search_tool
    
    def test_get_tool_definitions(self, tool_manager, mock_course_search_tool):
        """Test getting all tool definitions"""
        tool_manager.register_tool(mock_course_search_tool)
        
        definitions = tool_manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
    
    def test_execute_tool(self, tool_manager, mock_course_search_tool):
        """Test tool execution"""
        tool_manager.register_tool(mock_course_search_tool)
        
        result = tool_manager.execute_tool("search_course_content", query="test")
        
        assert result == "Test search result"
        mock_course_search_tool.execute.assert_called_once_with(query="test")
    
    def test_execute_nonexistent_tool(self, tool_manager):
        """Test execution of nonexistent tool"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self, tool_manager, mock_course_search_tool):
        """Test getting sources from last search"""
        tool_manager.register_tool(mock_course_search_tool)
        
        sources = tool_manager.get_last_sources()
        
        assert len(sources) == 1
        assert sources[0]["text"] == "Test Source"
        assert sources[0]["link"] == "http://test.com"
    
    def test_reset_sources(self, tool_manager, mock_course_search_tool):
        """Test resetting sources"""
        tool_manager.register_tool(mock_course_search_tool)
        
        # Initial sources exist
        assert len(tool_manager.get_last_sources()) == 1
        
        # Reset sources
        tool_manager.reset_sources()
        
        # Sources should be empty
        assert mock_course_search_tool.last_sources == []
    
    def test_register_tool_without_name(self, tool_manager):
        """Test registering tool without name raises error"""
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"description": "Tool without name"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            tool_manager.register_tool(mock_tool)