# Contributing Guide

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- uv (recommended) or pip

### Initial Setup

```bash
# Fork and clone
git clone https://github.com/YOUR-USERNAME/agemem.git
cd agemem

# Create virtual environment and install dependencies
uv sync

# Install with all optional dependencies
uv sync --all-extras

# Verify installation
python -m unittest tests.test_agemem -v
```

## Code Standards

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for all function parameters and return values
- Use docstrings for all public functions and classes
- Maximum line length: 88 characters (Black default)
- Use 4 spaces for indentation

### Type Hints

```python
from typing import Optional, List
from pathlib import Path

def search(
    query: str,
    top_k: int = 5,
    filters: Optional[dict] = None
) -> List[tuple[MemoryEntry, float]]:
    """
    Search for relevant memories.

    Args:
        query: Search query string
        top_k: Maximum results to return
        filters: Optional metadata filters

    Returns:
        List of (entry, score) tuples sorted by relevance
    """
```

### Docstring Convention

```python
def calculate_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec_a: First vector (must be unit normalized)
        vec_b: Second vector (must be unit normalized)

    Returns:
        Cosine similarity score between -1 and 1

    Raises:
        ValueError: If vectors have different dimensions

    Example:
        >>> a = np.array([1, 0, 0])
        >>> b = np.array([0.8, 0.6, 0])
        >>> calculate_similarity(a, b)
        0.8
    """
```

### Formatting Tools

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy .
```

## Testing

### Test Philosophy

All tests are **offline** — no LLM calls, no network access. This ensures:
- Tests run reliably in any environment
- No API costs or rate limits
- Fast test execution
- Deterministic results

### Running Tests

```bash
# Run all tests with verbose output
python -m unittest tests.test_agemem -v

# Run specific test file
python -m unittest tests.test_query_expansion -v

# Run specific test class
python -m unittest tests.test_agemem.TestLTMStore -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Structure

```python
import unittest
from unittest.mock import MagicMock
from core.types import MemoryEntry
from memory.ltm_store import LTMStore


class TestLTMStore(unittest.TestCase):
    """Test cases for LTMStore functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AgememConfig(
            LTM_MAX_ENTRIES=100,
            ENABLE_SEMANTIC_SEARCH=False
        )
        self.store = LTMStore(config=self.config)

    def tearDown(self):
        """Clean up after tests."""
        # Remove any persisted files
        pass

    def test_add_entry_increases_count(self):
        """Test that adding an entry increases store count."""
        initial_count = len(self.store.entries())
        entry = MemoryEntry(content="test content", learning_score=0.8)
        self.store.add(entry)
        self.assertEqual(len(self.store.entries()), initial_count + 1)

    def test_search_returns_relevant_entries(self):
        """Test that search ranks results by relevance."""
        # Add entries
        self.store.add(MemoryEntry(content="Python programming"))
        self.store.add(MemoryEntry(content="Java programming"))
        self.store.add(MemoryEntry(content="Cooking recipes"))

        # Search
        results = self.store.search("Python code", top_k=2)

        # Verify
        self.assertEqual(len(results), 2)
        self.assertIn("Python", results[0][0].content)
```

### Key Test Cases

| Test ID | What It Validates |
|---------|-------------------|
| T01 | TokenCounter estimation accuracy |
| T02-T05 | LTMStore CRUD operations |
| T06-T12 | STMContext lifecycle |
| T13-T15 | SystemRules firing conditions |
| T16-T18 | MemoryAgent decision parsing |
| T19 | LTM promotion via learning score |
| T20 | Double-boundary overflow invariant (critical) |
| BUG1-BUG3 | Regression tests for known issues |

### Mocking LLM Calls

```python
from unittest.mock import MagicMock
from agents.llm_client import LLMClient

def _mock_llm(response: str = "Mock response") -> LLMClient:
    """Create a mock LLMClient for testing."""
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = response
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[choice],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5)
    )
    return LLMClient(mock_client, default_model="test-model")
```

## Pull Request Process

### Before Submitting

1. **Run all tests**: `python -m unittest discover tests/ -v`
2. **Format code**: `black . && isort .`
3. **Lint**: `flake8 .`
4. **Type check**: `mypy .`
5. **Update documentation** if changing public APIs
6. **Add tests** for new functionality

### PR Checklist

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Test coverage maintained or improved

## Code Quality
- [ ] Follows PEP 8 style
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] No new warnings
```

### Review Process

1. All PRs require at least one maintainer review
2. Address feedback promptly
3. Maintain a clean git history (squash if needed)
4. Once approved, a maintainer will merge

## Architecture Principles

### Separation of Concerns

- **Orchestrator**: Only place that writes to LTM/STM
- **SystemRules**: Pure detection, no execution
- **MemoryAgent**: Analysis only, returns decisions
- **LTMStore/STMContext**: Storage, no LLM calls

### Data Flow

```
User → Orchestrator → (SystemRules + MemoryAgent + LearningScorer)
                    → LTMStore / STMContext
                    → LLMClient
                    → Response
```

### Key Invariants

1. Context within limits at **end** of every turn
2. Pinned messages never evicted
3. LTM persists on every write
4. STM persists after every turn

## Project Structure

```
agemem/
├── core/           # Data types, configuration, utilities
├── memory/         # LTM/STM storage implementations
├── triggers/       # Rule engine
├── agents/         # LLM-facing components
├── tools/          # Utility tools (web search, etc.)
├── skills/         # Skill detection and loading
├── prompts/        # Prompt registry
├── ingest/         # Document ingestion pipeline
└── tests/          # Unit tests
```

## Getting Help

- Check existing [GitHub Issues](https://github.com/gianpd/agemem/issues)
- Open a new issue with the `question` label
- Review inline documentation in source files

## Recognition

All contributors are recognized in the project README and release notes. Thank you for helping make AgeMem better!