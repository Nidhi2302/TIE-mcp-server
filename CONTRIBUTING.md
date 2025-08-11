# Contributing to TIE MCP Server

We welcome contributions to the TIE MCP Server project! This document provides guidelines for contributing to the project.

**Project Maintainer:** Nidhi Trivedi ([nj20383@gmail.com](mailto:nj20383@gmail.com))

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of MITRE ATT&CK framework
- Familiarity with Model Context Protocol (MCP)

### Development Environment Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/Nidhi2302/TIE-mcp-server.git
   cd TIE-mcp-server
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Install project in editable mode
   pip install -e .
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

4. **Verify Installation**
   ```bash
   # Run tests
   pytest
   
   # Check code style
   ruff check src/ tests/
   black --check src/ tests/
   
   # Type checking
   mypy src/
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, concise code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=tie_mcp --cov-report=html

# Run specific test files
pytest tests/unit/test_server.py -v
```

### 4. Code Quality Checks

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
ruff check src/ tests/ --fix

# Type checking
mypy src/
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` adding tests
- `chore:` maintenance tasks

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference any related issues
- Include screenshots if applicable
- List any breaking changes

## Code Style Guidelines

### Python Style
- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 88 characters
- Use descriptive variable names
- Add docstrings for all public functions and classes

### Documentation
- Update README.md for user-facing changes
- Add docstrings following Google style
- Update API documentation if applicable
- Include examples for new features

### Testing
- Write tests for all new functionality
- Maintain test coverage above 80%
- Use descriptive test names
- Mock external dependencies
- Test both success and failure cases

## Project Structure

```
tie-mcp-server/
├── src/tie_mcp/           # Main application code
│   ├── server.py          # MCP server implementation
│   ├── core/             # Core TIE functionality
│   ├── models/           # Model management
│   ├── storage/          # Database operations
│   └── utils/            # Utility functions
├── tests/                # Test suite
│   ├── unit/             # Unit tests
│   └── performance/      # Performance tests
├── data/                 # Data files (excluded from Git)
├── notebooks/            # Jupyter notebooks
└── docker/               # Docker configuration
```

## Types of Contributions

### Bug Reports
- Use GitHub Issues
- Include system information
- Provide minimal reproduction case
- Include error messages and logs

### Feature Requests
- Use GitHub Issues
- Describe the problem you're solving
- Explain the proposed solution
- Consider backward compatibility

### Code Contributions
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements

### Documentation
- README improvements
- Code comments
- API documentation
- Examples and tutorials

## Review Process

1. **Automated Checks**: All PRs must pass CI/CD checks
2. **Code Review**: At least one maintainer review required
3. **Testing**: Verify tests pass and coverage is maintained
4. **Documentation**: Ensure documentation is updated
5. **Merge**: Squash and merge after approval

## Release Process

Releases follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check README.md and code comments

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's technical standards

## Security

- Report security issues privately
- Don't include sensitive data in commits
- Follow security best practices
- Use secure coding practices

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for contributing to TIE MCP Server!