# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
- `uv sync` - Install Python dependencies using the modern uv package manager
- Requires Python 3.13+ and uv package manager

### Running the Application
- `./run.sh` - Start the full RAG system (creates docs directory, starts FastAPI server on port 8001)
- `cd backend && uv run uvicorn app:app --reload --port 8001` - Manual server start with hot reload
- Server serves frontend static files and provides API endpoints at `/api/query` and `/api/courses`
- Web interface available at `http://localhost:8001`
- API documentation at `http://localhost:8001/docs`

### Configuration
- Environment variables loaded from `.env` file in project root
- Key variables: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` (for custom API endpoints like DeepSeek)
- Configuration centralized in `backend/config.py` with dataclass pattern

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** designed for educational course materials with a clean modular architecture:

### Core RAG Pipeline
1. **Document Processing**: Structured course documents → parsed metadata + content chunks
2. **Vector Storage**: ChromaDB with dual collections (`course_catalog` for metadata, `course_content` for chunks)
3. **Semantic Search**: Tool-based architecture where AI decides when to search course materials
4. **Response Generation**: Anthropic Claude API (configurable endpoint) with tool calling capabilities

### Key Components

**RAGSystem (`rag_system.py`)**: Main orchestrator that coordinates all components. Handles document ingestion, query processing, and maintains the tool manager for search operations.

**DocumentProcessor (`document_processor.py`)**: Parses structured course documents with specific format requirements (Course Title, Instructor, Lesson sections). Implements intelligent text chunking with sentence-aware splitting (800 char chunks, 100 char overlap) to preserve semantic coherence.

**VectorStore (`vector_store.py`)**: ChromaDB interface with sophisticated course name resolution using vector similarity. Supports filtered search by course and lesson number. Dual-collection design separates metadata from content for optimal search performance.

**AIGenerator (`ai_generator.py`)**: Anthropic API client with custom base URL support. Handles tool execution workflow where AI autonomously decides whether to search course materials based on query context. Maintains conversation history and implements prompt caching optimizations.

**SearchTools (`search_tools.py`)**: Plugin-based tool architecture with CourseSearchTool implementing semantic course name matching. Tools track sources for transparency and return formatted results with course/lesson context.

### Data Models (`models.py`)
- `Course`: Contains metadata (title, instructor, link) and list of lessons
- `Lesson`: Individual lesson with number, title, and optional link
- `CourseChunk`: Text chunk with course context, lesson number, and positional index

### Document Format Requirements
Course documents must follow structured format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson N: [lesson title]
Lesson Link: [url]
[lesson content...]
```

### Frontend Integration
- Vanilla JavaScript frontend with real-time chat interface
- Session management with conversation history
- Markdown rendering for AI responses with source attribution
- Course statistics sidebar showing available materials

### Vector Search Strategy
- Uses `all-MiniLM-L6-v2` embedding model for semantic similarity
- Smart course name resolution handles partial matches and fuzzy search
- Chunk-level search with metadata preservation for accurate source attribution
- Configurable result limits and search parameters through Config class

### Tool Execution Flow
1. User query → RAG System processes with conversation history
2. AI Generator calls Claude API with tool definitions
3. AI autonomously decides whether to use search_course_content tool
4. If searching: CourseSearchTool executes vector search with optional filters
5. Search results returned to AI as context for final response generation
6. Response includes both answer and transparent source citations

## Important Implementation Notes

- **Custom API Endpoints**: System supports custom Anthropic-compatible endpoints via `ANTHROPIC_BASE_URL` (currently configured for DeepSeek)
- **Chunking Strategy**: Sentence-aware splitting prevents mid-sentence cuts while maintaining configurable chunk sizes
- **Session Isolation**: Each user conversation maintains independent session state with configurable history limits
- **Source Tracking**: Complete traceability from search results to final citations in user interface
- **Error Handling**: Comprehensive error handling at each layer with user-friendly fallback messages

## Document Processing Behavior

The system automatically loads documents from `docs/` directory on startup. Documents are processed incrementally (existing courses are not re-processed). Document processor expects structured format and extracts both metadata and content for dual-collection storage in ChromaDB.

- use uv to run python files or add any dependencies

- 不需要执行 `./run.sh` 命令, 我会自己手动执行它.