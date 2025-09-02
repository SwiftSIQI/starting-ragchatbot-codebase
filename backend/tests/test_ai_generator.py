import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add the backend directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from search_tools import ToolManager


class TestAIGenerator:
    """Test suite for AIGenerator functionality"""
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        return Mock()
    
    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AIGenerator instance with mocked client"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            return AIGenerator(
                api_key="test-key",
                model="test-model",
                base_url="https://test-api.com"
            )
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager"""
        mock_manager = Mock(spec=ToolManager)
        mock_manager.get_tool_definitions.return_value = [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        ]
        mock_manager.execute_tool.return_value = "Mock search results"
        return mock_manager
    
    @pytest.fixture
    def mock_text_response(self):
        """Create a mock text response from Anthropic"""
        mock_content = Mock()
        mock_content.text = "This is a test response"
        
        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"
        
        return mock_response
    
    @pytest.fixture
    def mock_tool_use_response(self):
        """Create a mock tool use response from Anthropic"""
        # Mock tool use content block
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_use_123"
        mock_tool_content.input = {"query": "machine learning"}
        
        # Mock response with tool use
        mock_response = Mock()
        mock_response.content = [mock_tool_content]
        mock_response.stop_reason = "tool_use"
        
        return mock_response
    
    def test_initialization(self):
        """Test AIGenerator initialization"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            ai_gen = AIGenerator("test-key", "claude-3", "https://api.test.com")
            
            # Check that client was created with correct parameters
            mock_anthropic.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.test.com"
            )
            
            # Check base parameters
            assert ai_gen.model == "claude-3"
            assert ai_gen.base_params["model"] == "claude-3"
            assert ai_gen.base_params["temperature"] == 0
            assert ai_gen.base_params["max_tokens"] == 800
    
    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client, mock_text_response):
        """Test response generation without tools"""
        mock_anthropic_client.messages.create.return_value = mock_text_response
        
        result = ai_generator.generate_response("What is machine learning?")
        
        assert result == "This is a test response"
        mock_anthropic_client.messages.create.assert_called_once()
        
        # Check call parameters
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["messages"][0]["content"] == "What is machine learning?"
        assert call_args[1]["model"] == "test-model"
        assert "tools" not in call_args[1]
    
    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_client, mock_text_response):
        """Test response generation with conversation history"""
        mock_anthropic_client.messages.create.return_value = mock_text_response
        
        history = "Previous conversation context"
        result = ai_generator.generate_response("Follow-up question", conversation_history=history)
        
        assert result == "This is a test response"
        
        # Check that system prompt includes history
        call_args = mock_anthropic_client.messages.create.call_args
        assert "Previous conversation context" in call_args[1]["system"]
    
    def test_generate_response_with_tools_no_tool_use(self, ai_generator, mock_anthropic_client, 
                                                     mock_text_response, mock_tool_manager):
        """Test response generation with tools but no tool use"""
        mock_anthropic_client.messages.create.return_value = mock_text_response
        
        result = ai_generator.generate_response(
            "General question",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        assert result == "This is a test response"
        
        # Check that tools were provided in the call
        call_args = mock_anthropic_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tool_choice"]["type"] == "auto"
    
    def test_generate_response_with_tool_execution(self, ai_generator, mock_anthropic_client, 
                                                  mock_tool_use_response, mock_tool_manager):
        """Test response generation with tool execution"""
        # Mock the final response after tool execution
        mock_final_content = Mock()
        mock_final_content.text = "Based on the search, machine learning is..."
        
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        mock_final_response.stop_reason = "end_turn"
        
        # First call returns tool use, second call returns final response
        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response
        ]
        
        result = ai_generator.generate_response(
            "What is machine learning?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        assert result == "Based on the search, machine learning is..."
        
        # Check that tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning"
        )
        
        # Check that two API calls were made
        assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_handle_sequential_tool_execution_workflow(self, ai_generator, mock_anthropic_client, 
                                                      mock_tool_use_response, mock_tool_manager):
        """Test the complete sequential tool execution workflow"""
        # Set up the final response after tool execution
        mock_final_content = Mock()
        mock_final_content.text = "Tool-based response"
        
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        mock_final_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
            "tools": [{"name": "search_course_content"}],
            "tool_choice": {"type": "auto"}
        }
        
        result = ai_generator._handle_sequential_tool_execution(
            mock_tool_use_response, 
            base_params, 
            mock_tool_manager
        )
        
        assert result == "Tool-based response"
        
        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify final API call structure
        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args[1]["messages"]
        
        # Should have original message, assistant's tool use, and tool result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"  # Tool result
        
        # Verify tools are still available in the API call
        assert "tools" in call_args[1]
        assert "tool_choice" in call_args[1]
    
    def test_handle_multiple_tool_calls(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test handling multiple tool calls in one response"""
        # Mock response with multiple tool uses
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "search_course_content"
        mock_tool_content_1.id = "tool_1"
        mock_tool_content_1.input = {"query": "first query"}
        
        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "search_course_content"
        mock_tool_content_2.id = "tool_2"
        mock_tool_content_2.input = {"query": "second query"}
        
        mock_multi_tool_response = Mock()
        mock_multi_tool_response.content = [mock_tool_content_1, mock_tool_content_2]
        
        # Mock final response
        mock_final_content = Mock()
        mock_final_content.text = "Combined response"
        
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "system": "system"
        }
        
        result = ai_generator._handle_tool_execution(
            mock_multi_tool_response,
            base_params,
            mock_tool_manager
        )
        
        assert result == "Combined response"
        
        # Should execute both tools
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Check the final API call includes both tool results
        call_args = mock_anthropic_client.messages.create.call_args
        tool_result_message = call_args[1]["messages"][2]
        assert len(tool_result_message["content"]) == 2  # Two tool results
    
    def test_system_prompt_structure(self, ai_generator):
        """Test the system prompt contains expected components"""
        system_prompt = ai_generator.SYSTEM_PROMPT
        
        # Check for key components
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "Tool Usage Guidelines" in system_prompt
        assert "Sequential Tool Strategy" in system_prompt
        assert "Up to 2 tool call rounds per query maximum" in system_prompt
        assert "Response Protocol" in system_prompt
        assert "Brief, Concise and focused" in system_prompt
    
    def test_error_handling_in_tool_execution(self, ai_generator, mock_anthropic_client, 
                                             mock_tool_use_response, mock_tool_manager):
        """Test error handling when tool execution fails"""
        # Make tool execution raise an exception
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Mock the final response after tool execution (even with error)
        mock_final_content = Mock()
        mock_final_content.text = "I apologize, there was an error processing your request."
        
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        mock_final_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "system": "system",
            "tools": [{"name": "search_course_content"}],
            "tool_choice": {"type": "auto"}
        }
        
        # Should not raise exception, but handle gracefully
        result = ai_generator._handle_sequential_tool_execution(
            mock_tool_use_response,
            base_params,
            mock_tool_manager
        )
        
        # Should still return a response even if tool execution failed
        assert result is not None
        
        # Verify tool results contain error message
        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args[1]["messages"]
        tool_result_message = messages[2]["content"]
        assert any("Tool execution error" in str(result["content"]) for result in tool_result_message)
    
    def test_base_params_configuration(self, ai_generator):
        """Test that base parameters are properly configured"""
        base_params = ai_generator.base_params
        
        assert base_params["model"] == "test-model"
        assert base_params["temperature"] == 0
        assert base_params["max_tokens"] == 800
        assert len(base_params) == 3  # Only these three parameters
    
    def test_api_call_structure_without_history(self, ai_generator, mock_anthropic_client, mock_text_response):
        """Test API call structure when no conversation history provided"""
        mock_anthropic_client.messages.create.return_value = mock_text_response
        
        ai_generator.generate_response("Test query")
        
        call_args = mock_anthropic_client.messages.create.call_args
        
        # Should contain only base system prompt
        assert call_args[1]["system"] == ai_generator.SYSTEM_PROMPT
        assert "Previous conversation" not in call_args[1]["system"]
    
    def test_api_call_structure_with_history(self, ai_generator, mock_anthropic_client, mock_text_response):
        """Test API call structure when conversation history is provided"""
        mock_anthropic_client.messages.create.return_value = mock_text_response
        
        history = "User: Hello\nAssistant: Hi there!"
        ai_generator.generate_response("Test query", conversation_history=history)
        
        call_args = mock_anthropic_client.messages.create.call_args
        
        # Should contain system prompt with history
        system_content = call_args[1]["system"]
        assert ai_generator.SYSTEM_PROMPT in system_content
        assert "Previous conversation:" in system_content
        assert history in system_content
    
    def test_sequential_tool_execution_single_round(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test sequential execution with single tool round"""
        # Mock tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_1"
        mock_tool_content.input = {"query": "python"}
        
        mock_tool_response = Mock()
        mock_tool_response.content = [mock_tool_content]
        mock_tool_response.stop_reason = "tool_use"
        
        # Mock final response (no more tool use)
        mock_final_content = Mock()
        mock_final_content.text = "Python is a programming language"
        
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        mock_final_response.stop_reason = "end_turn"
        
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "system": "system",
            "tools": [{"name": "search_course_content"}],
            "tool_choice": {"type": "auto"}
        }
        
        result = ai_generator._handle_sequential_tool_execution(
            mock_tool_response,
            base_params,
            mock_tool_manager
        )
        
        assert result == "Python is a programming language"
        
        # Should execute one tool
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="python")
        
        # Should make one API call for final response
        assert mock_anthropic_client.messages.create.call_count == 1
    
    def test_sequential_tool_execution_two_rounds(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test sequential execution with two tool rounds"""
        # Mock first tool use response
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "get_course_outline"
        mock_tool_content_1.id = "tool_1"
        mock_tool_content_1.input = {"course_name": "Python Course"}
        
        mock_first_response = Mock()
        mock_first_response.content = [mock_tool_content_1]
        mock_first_response.stop_reason = "tool_use"
        
        # Mock second tool use response
        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "search_course_content"
        mock_tool_content_2.id = "tool_2"
        mock_tool_content_2.input = {"query": "lesson 3", "course_filter": "Python Course"}
        
        mock_second_response = Mock()
        mock_second_response.content = [mock_tool_content_2]
        mock_second_response.stop_reason = "tool_use"
        
        # Mock final response (no more tool use)
        mock_final_content = Mock()
        mock_final_content.text = "Lesson 3 covers variables and data types"
        
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        mock_final_response.stop_reason = "end_turn"
        
        # Configure API responses: first round -> second round -> final
        mock_anthropic_client.messages.create.side_effect = [
            mock_second_response,  # First API call after first tool execution
            mock_final_response    # Second API call after second tool execution
        ]
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is in lesson 3 of Python Course?"}],
            "system": "system",
            "tools": [{"name": "get_course_outline"}, {"name": "search_course_content"}],
            "tool_choice": {"type": "auto"}
        }
        
        result = ai_generator._handle_sequential_tool_execution(
            mock_first_response,
            base_params,
            mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Lesson 3 covers variables and data types"
        
        # Should execute both tools
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Python Course")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="lesson 3", course_filter="Python Course")
        
        # Should make two API calls (one for each round)
        assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_sequential_tool_execution_max_rounds_limit(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test that sequential execution stops at max rounds limit"""
        # Mock tool use responses that would continue beyond limit
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_1"
        mock_tool_content.input = {"query": "test"}
        
        mock_tool_response = Mock()
        mock_tool_response.content = [mock_tool_content]
        mock_tool_response.stop_reason = "tool_use"
        
        # Configure API to always return tool use (would loop forever without limit)
        mock_anthropic_client.messages.create.return_value = mock_tool_response
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "system": "system",
            "tools": [{"name": "search_course_content"}],
            "tool_choice": {"type": "auto"}
        }
        
        result = ai_generator._handle_sequential_tool_execution(
            mock_tool_response,
            base_params,
            mock_tool_manager,
            max_rounds=2
        )
        
        # Should terminate at max rounds and return last response
        assert result is not None
        
        # Should execute tools exactly 2 times (max_rounds)
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Should make exactly 2 API calls (max_rounds)
        assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_sequential_tool_execution_api_error(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test error handling when API call fails in sequential execution"""
        # Mock initial tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_1"
        mock_tool_content.input = {"query": "test"}
        
        mock_tool_response = Mock()
        mock_tool_response.content = [mock_tool_content]
        mock_tool_response.stop_reason = "tool_use"
        
        # Make API call fail
        mock_anthropic_client.messages.create.side_effect = Exception("API call failed")
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "system": "system",
            "tools": [{"name": "search_course_content"}],
            "tool_choice": {"type": "auto"}
        }
        
        result = ai_generator._handle_sequential_tool_execution(
            mock_tool_response,
            base_params,
            mock_tool_manager
        )
        
        # Should return error message
        assert "Error in tool execution round 1: API call failed" in result
        
        # Should still execute the tool from initial response
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="test")
    
    def test_execute_tools_helper(self, ai_generator, mock_tool_manager):
        """Test the _execute_tools helper method"""
        # Mock response with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "python"}
        
        mock_response = Mock()
        mock_response.content = [mock_tool_content]
        
        mock_tool_manager.execute_tool.return_value = "Search results"
        
        result = ai_generator._execute_tools(mock_response, mock_tool_manager)
        
        assert len(result) == 1
        assert result[0]["type"] == "tool_result"
        assert result[0]["tool_use_id"] == "tool_123"
        assert result[0]["content"] == "Search results"
        
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="python")
    
    def test_execute_tools_error_handling(self, ai_generator, mock_tool_manager):
        """Test _execute_tools handles individual tool errors"""
        # Mock response with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "python"}
        
        mock_response = Mock()
        mock_response.content = [mock_tool_content]
        
        # Make tool execution fail
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")
        
        result = ai_generator._execute_tools(mock_response, mock_tool_manager)
        
        assert len(result) == 1
        assert result[0]["type"] == "tool_result"
        assert result[0]["tool_use_id"] == "tool_123"
        assert "Tool execution error: Tool failed" in result[0]["content"]
    
    def test_fallback_to_legacy_execution(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test fallback to legacy execution when sequential fails"""
        # Mock tool use response for initial API call
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_1"
        mock_tool_content.input = {"query": "test"}
        
        mock_tool_response = Mock()
        mock_tool_response.content = [mock_tool_content]
        mock_tool_response.stop_reason = "tool_use"
        
        # Mock successful legacy response
        mock_final_content = Mock()
        mock_final_content.text = "Fallback response"
        
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        
        # Mock the sequential method to raise an exception by patching it directly
        original_sequential_method = ai_generator._handle_sequential_tool_execution
        
        def failing_sequential(*args, **kwargs):
            raise Exception("Sequential method failed")
        
        ai_generator._handle_sequential_tool_execution = Mock(side_effect=failing_sequential)
        
        # Configure API calls - initial returns tool use, legacy call returns final response
        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response,  # Initial API call returns tool use
            mock_final_response  # Legacy execution succeeds
        ]
        
        result = ai_generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        assert result == "Fallback response"
        
        # Should call API twice - initial, then legacy (sequential never reaches API)
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Verify sequential method was attempted
        ai_generator._handle_sequential_tool_execution.assert_called_once()
        
        # Restore original method
        ai_generator._handle_sequential_tool_execution = original_sequential_method