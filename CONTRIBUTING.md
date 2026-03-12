# Contributing to AgeMem

Thank you for your interest in contributing to AgeMem! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/agemem.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `python -m unittest tests.test_agemem -v`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/gianpd/agemem.git
cd agemem

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

### Running Tests

```bash
# Run all tests
python -m unittest tests.test_agemem -v

# Run specific test file
python -m unittest tests.test_query_expansion -v

# Run with coverage
python -m pytest tests/ --cov=.
```

## Code Style

### Python Style Guide

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use type hints for all function parameters and return values
- Use docstrings for all public functions and classes
- Keep functions focused and small (under 50 lines when possible)
- Use meaningful variable and function names

### Formatting

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (Black formatter default)
- Use single quotes for strings, double quotes for docstrings
- Add trailing commas in multi-line structures

### Linting

We use the following tools for code quality:

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

### Example Code Style

```python
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    """Represents a single memory entry in long-term storage."""
    
    content: str
    learning_score: float
    timestamp: float
    metadata: Optional[dict] = None
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if entry meets quality threshold.
        
        Args:
            threshold: Minimum learning score for high quality
            
        Returns:
            True if entry is high quality, False otherwise
        """
        return self.learning_score >= threshold
```

## Testing

### Test Requirements

- All new features must include tests
- Tests should be independent and not rely on external services
- Use descriptive test names that explain what is being tested
- Aim for high test coverage (>80%)

### Test Structure

```python
import unittest
from core.types import MemoryEntry
from memory.ltm_store import LTMStore

class TestLTMStore(unittest.TestCase):
    """Test cases for LTMStore functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.store = LTMStore()
    
    def test_add_entry_increases_count(self):
        """Test that adding an entry increases store count."""
        initial_count = len(self.store.entries)
        entry = MemoryEntry(content="test", learning_score=0.8, timestamp=1.0)
        self.store.add(entry)
        self.assertEqual(len(self.store.entries), initial_count + 1)
    
    def test_search_returns_relevant_entries(self):
        """Test that search returns relevant entries."""
        # Test implementation
        pass
```

### Running Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test class
python -m unittest tests.test_agemem.TestLTMStore

# Run with verbose output
python -m unittest tests.test_agemem -v
```

## Pull Request Process

### Before Submitting

1. Ensure your code follows the style guidelines
2. Run all tests and ensure they pass
3. Update documentation if needed
4. Add tests for new functionality
5. Update the changelog if applicable

### PR Guidelines

- **Title**: Use a clear, descriptive title
- **Description**: Explain what the PR does and why
- **Related Issues**: Link to any related issues
- **Testing**: Describe how you tested the changes
- **Screenshots**: Include screenshots for UI changes

### PR Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested the changes manually

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

## Related Issues
Closes #(issue number)
```

### Review Process

1. All PRs require at least one review from a maintainer
2. Address any feedback from reviewers
3. Once approved, a maintainer will merge your PR
4. Your contribution will be included in the next release

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: OS, Python version, dependencies
- **Screenshots**: If applicable

### Feature Requests

When requesting features, please include:

- **Description**: Clear description of the feature
- **Use Case**: Why this feature would be useful
- **Proposed Solution**: How you think it should work
- **Alternatives**: Any alternative solutions you've considered

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

## Community

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and discussions
- **Pull Requests**: For code contributions

### Getting Help

If you need help with contributing:

1. Check the existing documentation
2. Search existing issues and discussions
3. Create a new issue with the `question` label
4. Join our community discussions

### Recognition

All contributors will be recognized in the project's README and release notes. We appreciate every contribution, no matter how small!

## License

By contributing to AgeMem, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

Thank you for contributing to AgeMem! Your help makes this project better for everyone.