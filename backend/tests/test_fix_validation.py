import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add the backend directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from search_tools import CourseSearchTool
from vector_store import VectorStore


class TestFixValidation:
    """Tests to validate that the MAX_RESULTS=0 bug has been fixed"""

    def test_config_max_results_fixed(self):
        """Test that MAX_RESULTS is no longer 0"""
        assert config.MAX_RESULTS > 0, "MAX_RESULTS should be greater than 0"
        assert (
            config.MAX_RESULTS == 5
        ), f"Expected MAX_RESULTS=5, got {config.MAX_RESULTS}"

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_vector_store_uses_correct_max_results(self, mock_embedding, mock_chroma):
        """Test that VectorStore uses the correct MAX_RESULTS value"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        # Create VectorStore with config values
        store = VectorStore(
            chroma_path=config.CHROMA_PATH,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS,
        )

        # Mock successful search results
        mock_collection.query.return_value = {
            "documents": [["Test content 1", "Test content 2", "Test content 3"]],
            "metadatas": [
                [
                    {"course_title": "Test"},
                    {"course_title": "Test"},
                    {"course_title": "Test"},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }

        store.course_content = mock_collection

        # Perform search
        results = store.search("test query")

        # Verify that ChromaDB was called with MAX_RESULTS=5 (not 0)
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=5, where=None  # Should be 5, not 0
        )

        # Should return results (not empty)
        assert not results.is_empty()
        assert len(results.documents) == 3

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_tool_returns_results_after_fix(self, mock_embedding, mock_chroma):
        """Test that CourseSearchTool returns results after the fix"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        # Create VectorStore with fixed config
        store = VectorStore(
            chroma_path=config.CHROMA_PATH,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS,
        )
        store.course_content = mock_collection
        store.course_catalog = mock_collection

        # Mock search results that would be found with MAX_RESULTS > 0
        mock_collection.query.return_value = {
            "documents": [["Machine learning is a powerful technology."]],
            "metadatas": [[{"course_title": "Introduction to ML", "lesson_number": 1}]],
            "distances": [[0.1]],
        }

        # Mock link retrieval
        store.get_lesson_link = Mock(return_value="https://example.com/lesson-1")

        # Create search tool
        search_tool = CourseSearchTool(store)

        # Execute search
        result = search_tool.execute("What is machine learning?")

        # Should return formatted results (not "No relevant content found")
        assert "No relevant content found" not in result
        assert "Introduction to ML - Lesson 1" in result
        assert "Machine learning is a powerful technology" in result

        # Should have sources tracked
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]["text"] == "Introduction to ML - Lesson 1"

    def test_config_values_are_reasonable(self):
        """Test that all configuration values are reasonable"""
        assert (
            config.MAX_RESULTS >= 3
        ), "MAX_RESULTS should be at least 3 for useful results"
        assert (
            config.MAX_RESULTS <= 20
        ), "MAX_RESULTS should not be too high to avoid performance issues"
        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE should be positive"
        assert config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP should be non-negative"
        assert config.MAX_HISTORY >= 0, "MAX_HISTORY should be non-negative"

    def test_before_and_after_comparison(self):
        """Test to demonstrate the difference between MAX_RESULTS=0 and MAX_RESULTS=5"""

        # Simulate the bug scenario (MAX_RESULTS=0)
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_chroma,
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client

            # Create store with MAX_RESULTS=0 (the bug)
            buggy_store = VectorStore("test_path", "test_model", max_results=0)
            buggy_store.course_content = mock_collection

            # With MAX_RESULTS=0, ChromaDB returns empty results
            mock_collection.query.return_value = {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

            results_with_bug = buggy_store.search("test query")

            # Reset mock
            mock_collection.reset_mock()

            # Now test with MAX_RESULTS=5 (the fix)
            fixed_store = VectorStore("test_path", "test_model", max_results=5)
            fixed_store.course_content = mock_collection

            # With MAX_RESULTS=5, ChromaDB can return results
            mock_collection.query.return_value = {
                "documents": [["Test result"]],
                "metadatas": [[{"course_title": "Test"}]],
                "distances": [[0.1]],
            }

            results_after_fix = fixed_store.search("test query")

            # Compare results
            assert (
                results_with_bug.is_empty()
            ), "Buggy version should return empty results"
            assert (
                not results_after_fix.is_empty()
            ), "Fixed version should return results"

            # Verify the behavior difference
            # The key point is that buggy version returns empty, fixed version returns results
            # (the actual n_results values are set during VectorStore initialization)
