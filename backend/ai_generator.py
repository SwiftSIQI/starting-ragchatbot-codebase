from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content**: For searching specific course content and materials
2. **get_course_outline**: For getting course structure, including title, course link, and complete lesson list with numbers and titles

Tool Usage Guidelines:
- Use **search_course_content** for questions about specific course content or detailed educational materials
- Use **get_course_outline** for questions about course structure, lesson lists, course overview, or syllabus information
- **Up to 2 tool call rounds per query maximum** - you can make additional tool calls after seeing initial results
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Sequential Tool Strategy:
- First round: Use tools to gather initial information
- Second round (optional): Refine search or gather additional details based on first round results
- Consider whether additional tool calls will improve your response quality
- Examples: Get course outline first, then search specific lessons; Search one topic, then compare with another

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, base_url: str = ""):
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            try:
                return self._handle_sequential_tool_execution(
                    response, api_params, tool_manager
                )
            except Exception as e:
                # Fallback to legacy single-round execution
                print(f"Sequential tool execution failed, falling back: {e}")
                return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return response.content[0].text

    def _execute_tools(self, response, tool_manager):
        """
        Execute all tool calls in a response and return formatted results.

        Args:
            response: The response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            List of tool results formatted for API consumption
        """
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Add error result instead of failing completely
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution error: {str(e)}",
                        }
                    )

        return tool_results

    def _handle_sequential_tool_execution(
        self,
        initial_response,
        base_params: Dict[str, Any],
        tool_manager,
        max_rounds: int = 2,
    ):
        """
        Handle sequential tool execution across multiple rounds.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters including tools
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool call rounds (default: 2)

        Returns:
            Final response text after all tool execution rounds
        """
        messages = base_params["messages"].copy()
        current_response = initial_response
        round_count = 0

        while round_count < max_rounds and current_response.stop_reason == "tool_use":

            round_count += 1

            # Add AI's tool use response to conversation
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute tools and collect results
            tool_results = self._execute_tools(current_response, tool_manager)

            if not tool_results:  # Tool execution failed
                break

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})

            # Prepare next API call WITH tools still available
            next_params = {
                **base_params,  # Includes tools and other original parameters
                "messages": messages,
                "system": base_params["system"],
            }

            # Get next response from Claude
            try:
                current_response = self.client.messages.create(**next_params)
            except Exception as e:
                # Return error message if API call fails
                return f"Error in tool execution round {round_count}: {str(e)}"

        # Return final response text
        return (
            current_response.content[0].text
            if current_response.content
            else "No response generated"
        )

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Legacy single-round tool execution for backwards compatibility.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute tools and collect results
        tool_results = self._execute_tools(initial_response, tool_manager)

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
