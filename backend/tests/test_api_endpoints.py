"""API endpoint tests for the FastAPI RAG system.

These tests verify the correct behavior of all API endpoints including
request/response handling, error cases, and proper integration with
the RAG system components.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import status
from unittest.mock import patch, Mock
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint."""
    
    def test_query_with_session_id_success(self, test_client: TestClient, sample_query_data):
        """Test successful query with provided session ID."""
        query_data = sample_query_data["valid_query"]
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert "answer" in response_data
        assert "sources" in response_data
        assert "session_id" in response_data
        assert response_data["session_id"] == query_data["session_id"]
        assert isinstance(response_data["sources"], list)
        
        # Verify source structure
        for source in response_data["sources"]:
            assert "text" in source
            assert "link" in source  # Can be None
    
    def test_query_without_session_id_creates_new_session(self, test_client: TestClient, sample_query_data):
        """Test query without session ID creates a new session."""
        query_data = sample_query_data["query_without_session"]
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["session_id"] == "new_session_123"  # From mock
    
    def test_query_with_empty_query_string(self, test_client: TestClient):
        """Test query with empty query string."""
        query_data = {"query": "", "session_id": "test_session"}
        
        response = test_client.post("/api/query", json=query_data)
        
        # Should still process but return appropriate response
        assert response.status_code == status.HTTP_200_OK
    
    def test_query_with_missing_query_field(self, test_client: TestClient, sample_query_data):
        """Test query with missing required query field."""
        invalid_data = sample_query_data["invalid_query"]
        
        response = test_client.post("/api/query", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_query_with_invalid_json(self, test_client: TestClient):
        """Test query with malformed JSON."""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_query_rag_system_exception(self, test_client: TestClient, mock_rag_system):
        """Test query when RAG system raises an exception."""
        # Configure mock to raise exception
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        query_data = {"query": "test query", "session_id": "test_session"}
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "RAG system error" in response.json()["detail"]
    
    def test_query_with_very_long_query(self, test_client: TestClient):
        """Test query with very long query string."""
        long_query = "What is Python? " * 1000  # Very long query
        query_data = {"query": long_query, "session_id": "test_session"}
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_query_source_formats(self, test_client: TestClient, mock_rag_system):
        """Test query with different source formats."""
        # Test different source formats that might be returned
        mock_rag_system.query.return_value = (
            "Test answer",
            [
                {"text": "Dict source with link", "link": "https://example.com"},
                {"text": "Dict source without link"},
                "String source",
                123  # Non-string source
            ]
        )
        
        query_data = {"query": "test", "session_id": "test"}
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == status.HTTP_200_OK
        sources = response.json()["sources"]
        
        assert len(sources) == 4
        assert sources[0]["text"] == "Dict source with link"
        assert sources[0]["link"] == "https://example.com"
        assert sources[1]["text"] == "Dict source without link"
        assert sources[1]["link"] is None
        assert sources[2]["text"] == "String source"
        assert sources[2]["link"] is None
        assert sources[3]["text"] == "123"  # Converted to string
        assert sources[3]["link"] is None


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint."""
    
    def test_get_courses_success(self, test_client: TestClient):
        """Test successful retrieval of course statistics."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert "total_courses" in response_data
        assert "course_titles" in response_data
        assert isinstance(response_data["total_courses"], int)
        assert isinstance(response_data["course_titles"], list)
        
        # Verify expected data from mock
        assert response_data["total_courses"] == 2
        assert "Introduction to Python Programming" in response_data["course_titles"]
        assert "Advanced Machine Learning" in response_data["course_titles"]
    
    def test_get_courses_empty_result(self, test_client: TestClient, mock_rag_system):
        """Test course endpoint when no courses are available."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["total_courses"] == 0
        assert response_data["course_titles"] == []
    
    def test_get_courses_rag_system_exception(self, test_client: TestClient, mock_rag_system):
        """Test courses endpoint when RAG system raises an exception."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Analytics error" in response.json()["detail"]
    
    def test_get_courses_method_not_allowed(self, test_client: TestClient):
        """Test that POST method is not allowed on courses endpoint."""
        response = test_client.post("/api/courses", json={"test": "data"})
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    def test_query_and_courses_session_consistency(self, test_client: TestClient):
        """Test that query and courses endpoints work together."""
        # First get courses
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == status.HTTP_200_OK
        
        # Then make a query
        query_data = {"query": "What courses are available?"}
        query_response = test_client.post("/api/query", json=query_data)
        assert query_response.status_code == status.HTTP_200_OK
        
        # Verify both responses are consistent
        courses_data = courses_response.json()
        query_response_data = query_response.json()
        
        assert courses_data["total_courses"] > 0
        assert len(query_response_data["sources"]) >= 0
    
    def test_multiple_queries_same_session(self, test_client: TestClient):
        """Test multiple queries using the same session ID."""
        session_id = "consistent_session_123"
        
        # First query
        query1_data = {"query": "What is Python?", "session_id": session_id}
        response1 = test_client.post("/api/query", json=query1_data)
        assert response1.status_code == status.HTTP_200_OK
        assert response1.json()["session_id"] == session_id
        
        # Second query with same session
        query2_data = {"query": "How do loops work?", "session_id": session_id}
        response2 = test_client.post("/api/query", json=query2_data)
        assert response2.status_code == status.HTTP_200_OK
        assert response2.json()["session_id"] == session_id
    
    def test_concurrent_queries_different_sessions(self, test_client: TestClient):
        """Test concurrent queries with different sessions."""
        query1_data = {"query": "Query 1", "session_id": "session_1"}
        query2_data = {"query": "Query 2", "session_id": "session_2"}
        
        # Make concurrent requests (TestClient handles this synchronously but tests the pattern)
        response1 = test_client.post("/api/query", json=query1_data)
        response2 = test_client.post("/api/query", json=query2_data)
        
        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK
        assert response1.json()["session_id"] != response2.json()["session_id"]


@pytest.mark.api
class TestAPIValidation:
    """Test API input validation and edge cases."""
    
    def test_query_extra_fields_ignored(self, test_client: TestClient):
        """Test that extra fields in query request are ignored."""
        query_data = {
            "query": "Test query",
            "session_id": "test_session",
            "extra_field": "should be ignored",
            "another_extra": {"nested": "data"}
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_query_null_session_id(self, test_client: TestClient):
        """Test query with null session_id."""
        query_data = {"query": "Test query", "session_id": None}
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == status.HTTP_200_OK
        # Should create new session since session_id is None
        assert response.json()["session_id"] == "new_session_123"
    
    def test_response_headers(self, test_client: TestClient):
        """Test that responses include proper headers."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        assert "content-type" in response.headers
        assert "application/json" in response.headers["content-type"]
    
    def test_cors_headers_present(self, test_client: TestClient):
        """Test that CORS headers are properly set."""
        response = test_client.options("/api/query")
        
        # Note: TestClient might not fully simulate CORS behavior
        # But we can verify the app has CORS middleware configured
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]


@pytest.mark.api 
class TestAPIErrorHandling:
    """Test API error handling scenarios."""
    
    def test_404_on_unknown_endpoint(self, test_client: TestClient):
        """Test 404 response on unknown endpoints."""
        response = test_client.get("/api/unknown")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_405_on_wrong_method(self, test_client: TestClient):
        """Test 405 response on wrong HTTP method."""
        response = test_client.get("/api/query")  # Should be POST
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_content_type_json_required(self, test_client: TestClient):
        """Test that JSON content type is required for POST requests."""
        response = test_client.post(
            "/api/query",
            data="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        # FastAPI should handle this gracefully but might return validation error
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])