import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add the backend directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestSearchResults:
    """Test suite for SearchResults class"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"course": "ML"}, {"course": "AI"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"course": "ML"}, {"course": "AI"}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {"documents": [], "metadatas": [], "distances": []}

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Database error")

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Database error"

    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        non_empty_results = SearchResults(["doc"], [{}], [0.1])

        assert empty_results.is_empty()
        assert not non_empty_results.is_empty()


class TestVectorStore:
    """Test suite for VectorStore functionality"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary directory for ChromaDB testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock ChromaDB client"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        return mock_client, mock_collection

    @pytest.fixture
    def vector_store(self, temp_chroma_path, mock_chroma_client):
        """Create VectorStore instance with mocked ChromaDB"""
        mock_client, mock_collection = mock_chroma_client

        with (
            patch("vector_store.chromadb.PersistentClient", return_value=mock_client),
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            store = VectorStore(
                chroma_path=temp_chroma_path,
                embedding_model="test-model",
                max_results=5,
            )

            # Set the mock collections
            store.course_catalog = mock_collection
            store.course_content = mock_collection

            return store, mock_collection

    @pytest.fixture
    def sample_course(self):
        """Create a sample course for testing"""
        return Course(
            title="Introduction to Machine Learning",
            course_link="https://example.com/ml-course",
            instructor="Dr. Jane Smith",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="What is ML?",
                    lesson_link="https://example.com/lesson-1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Linear Regression",
                    lesson_link="https://example.com/lesson-2",
                ),
            ],
        )

    @pytest.fixture
    def sample_chunks(self):
        """Create sample course chunks for testing"""
        return [
            CourseChunk(
                content="Machine learning is a subset of AI",
                course_title="Introduction to Machine Learning",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Linear regression is a fundamental algorithm",
                course_title="Introduction to Machine Learning",
                lesson_number=2,
                chunk_index=1,
            ),
        ]

    def test_initialization(self, temp_chroma_path):
        """Test VectorStore initialization"""
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_client_class,
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ) as mock_embedding,
        ):

            mock_client = Mock()
            mock_client_class.return_value = mock_client

            store = VectorStore(
                chroma_path=temp_chroma_path,
                embedding_model="test-model",
                max_results=10,
            )

            assert store.max_results == 10
            mock_client_class.assert_called_once()
            mock_embedding.assert_called_once_with(model_name="test-model")

            # Should create two collections
            assert mock_client.get_or_create_collection.call_count == 2

    def test_search_successful(self, vector_store):
        """Test successful search operation"""
        store, mock_collection = vector_store

        # Mock successful ChromaDB query
        mock_collection.query.return_value = {
            "documents": [["Machine learning content"]],
            "metadatas": [[{"course_title": "ML Course", "lesson_number": 1}]],
            "distances": [[0.1]],
        }

        results = store.search("machine learning")

        assert not results.is_empty()
        assert len(results.documents) == 1
        assert results.documents[0] == "Machine learning content"
        assert results.error is None

        # Verify ChromaDB query was called correctly
        mock_collection.query.assert_called_once_with(
            query_texts=["machine learning"], n_results=5, where=None  # max_results
        )

    def test_search_with_max_results_zero(self, temp_chroma_path):
        """Test search with max_results set to 0 - should identify the critical bug"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with (
            patch("vector_store.chromadb.PersistentClient", return_value=mock_client),
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            # Create store with max_results=0 (the bug in config)
            store = VectorStore(
                chroma_path=temp_chroma_path,
                embedding_model="test-model",
                max_results=0,  # This is the bug!
            )
            store.course_content = mock_collection

            # Mock ChromaDB response
            mock_collection.query.return_value = {
                "documents": [[]],  # No results due to n_results=0
                "metadatas": [[]],
                "distances": [[]],
            }

            results = store.search("test query")

            # Should return empty results due to max_results=0
            assert results.is_empty()

            # Verify that ChromaDB was called with n_results=0
            mock_collection.query.assert_called_once_with(
                query_texts=["test query"],
                n_results=0,  # This is the problem!
                where=None,
            )

    def test_search_with_course_filter(self, vector_store):
        """Test search with course name filter"""
        store, mock_collection = vector_store

        # Mock course name resolution
        mock_collection.query.side_effect = [
            # First call for course resolution
            {
                "documents": [["ML Course"]],
                "metadatas": [[{"title": "Machine Learning Course"}]],
                "distances": [[0.0]],
            },
            # Second call for content search
            {
                "documents": [["Course content"]],
                "metadatas": [[{"course_title": "Machine Learning Course"}]],
                "distances": [[0.1]],
            },
        ]

        results = store.search("algorithms", course_name="ML")

        assert not results.is_empty()
        assert len(results.documents) == 1

        # Should make two calls: one for course resolution, one for content search
        assert mock_collection.query.call_count == 2

        # Check content search call had correct filter
        content_search_call = mock_collection.query.call_args_list[1]
        assert content_search_call[1]["where"] == {
            "course_title": "Machine Learning Course"
        }

    def test_search_with_lesson_filter(self, vector_store):
        """Test search with lesson number filter"""
        store, mock_collection = vector_store

        mock_collection.query.return_value = {
            "documents": [["Lesson content"]],
            "metadatas": [[{"lesson_number": 2}]],
            "distances": [[0.1]],
        }

        results = store.search("regression", lesson_number=2)

        assert not results.is_empty()

        # Verify filter was applied
        mock_collection.query.assert_called_once_with(
            query_texts=["regression"], n_results=5, where={"lesson_number": 2}
        )

    def test_search_with_both_filters(self, vector_store):
        """Test search with both course and lesson filters"""
        store, mock_collection = vector_store

        # Mock course resolution
        mock_collection.query.side_effect = [
            # Course resolution
            {
                "documents": [["course"]],
                "metadatas": [[{"title": "ML Course"}]],
                "distances": [[0.0]],
            },
            # Content search
            {
                "documents": [["filtered content"]],
                "metadatas": [[{"course_title": "ML Course", "lesson_number": 1}]],
                "distances": [[0.1]],
            },
        ]

        results = store.search("test", course_name="ML", lesson_number=1)

        assert not results.is_empty()

        # Check that AND filter was applied
        content_search_call = mock_collection.query.call_args_list[1]
        expected_filter = {
            "$and": [{"course_title": "ML Course"}, {"lesson_number": 1}]
        }
        assert content_search_call[1]["where"] == expected_filter

    def test_search_course_not_found(self, vector_store):
        """Test search when course name cannot be resolved"""
        store, mock_collection = vector_store

        # Mock course resolution returning no results
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        results = store.search("test", course_name="Nonexistent Course")

        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.is_empty()

    def test_search_exception_handling(self, vector_store):
        """Test search exception handling"""
        store, mock_collection = vector_store

        # Make ChromaDB query raise an exception
        mock_collection.query.side_effect = Exception("Database connection failed")

        results = store.search("test query")

        assert results.error == "Search error: Database connection failed"
        assert results.is_empty()

    def test_resolve_course_name_successful(self, vector_store):
        """Test successful course name resolution"""
        store, mock_collection = vector_store

        mock_collection.query.return_value = {
            "documents": [["Course title"]],
            "metadatas": [[{"title": "Machine Learning Fundamentals"}]],
            "distances": [[0.1]],
        }

        resolved = store._resolve_course_name("ML")

        assert resolved == "Machine Learning Fundamentals"
        mock_collection.query.assert_called_once_with(query_texts=["ML"], n_results=1)

    def test_resolve_course_name_not_found(self, vector_store):
        """Test course name resolution when no match found"""
        store, mock_collection = vector_store

        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        resolved = store._resolve_course_name("Unknown Course")

        assert resolved is None

    def test_build_filter_no_filters(self, vector_store):
        """Test filter building with no filters"""
        store, _ = vector_store

        filter_dict = store._build_filter(None, None)

        assert filter_dict is None

    def test_build_filter_course_only(self, vector_store):
        """Test filter building with course filter only"""
        store, _ = vector_store

        filter_dict = store._build_filter("Test Course", None)

        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, vector_store):
        """Test filter building with lesson filter only"""
        store, _ = vector_store

        filter_dict = store._build_filter(None, 2)

        assert filter_dict == {"lesson_number": 2}

    def test_build_filter_both(self, vector_store):
        """Test filter building with both filters"""
        store, _ = vector_store

        filter_dict = store._build_filter("Test Course", 3)

        expected = {"$and": [{"course_title": "Test Course"}, {"lesson_number": 3}]}
        assert filter_dict == expected

    def test_add_course_metadata(self, vector_store, sample_course):
        """Test adding course metadata"""
        store, mock_collection = vector_store

        store.add_course_metadata(sample_course)

        # Verify ChromaDB add was called
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args

        # Check the data structure
        assert call_args[1]["documents"] == ["Introduction to Machine Learning"]
        assert call_args[1]["ids"] == ["Introduction to Machine Learning"]

        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == "Introduction to Machine Learning"
        assert metadata["instructor"] == "Dr. Jane Smith"
        assert metadata["course_link"] == "https://example.com/ml-course"
        assert metadata["lesson_count"] == 2
        assert "lessons_json" in metadata

    def test_add_course_content(self, vector_store, sample_chunks):
        """Test adding course content chunks"""
        store, mock_collection = vector_store

        store.add_course_content(sample_chunks)

        # Verify ChromaDB add was called
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args

        # Check documents
        assert len(call_args[1]["documents"]) == 2
        assert call_args[1]["documents"][0] == "Machine learning is a subset of AI"

        # Check metadata
        assert len(call_args[1]["metadatas"]) == 2
        assert (
            call_args[1]["metadatas"][0]["course_title"]
            == "Introduction to Machine Learning"
        )
        assert call_args[1]["metadatas"][0]["lesson_number"] == 1

        # Check IDs
        assert "Introduction_to_Machine_Learning_0" in call_args[1]["ids"]

    def test_add_empty_course_content(self, vector_store):
        """Test adding empty course content list"""
        store, mock_collection = vector_store

        store.add_course_content([])

        # Should not call ChromaDB add
        mock_collection.add.assert_not_called()

    def test_get_existing_course_titles(self, vector_store):
        """Test getting existing course titles"""
        store, mock_collection = vector_store

        mock_collection.get.return_value = {"ids": ["Course 1", "Course 2", "Course 3"]}

        titles = store.get_existing_course_titles()

        assert titles == ["Course 1", "Course 2", "Course 3"]
        mock_collection.get.assert_called_once()

    def test_get_course_count(self, vector_store):
        """Test getting course count"""
        store, mock_collection = vector_store

        mock_collection.get.return_value = {"ids": ["Course 1", "Course 2"]}

        count = store.get_course_count()

        assert count == 2

    def test_search_custom_limit(self, vector_store):
        """Test search with custom limit parameter"""
        store, mock_collection = vector_store

        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2", "doc3"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.1, 0.2, 0.3]],
        }

        results = store.search("test query", limit=3)

        # Should use custom limit instead of max_results
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=3, where=None  # Custom limit
        )
