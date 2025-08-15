# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com/en/1.1.0/)
and this project adheres to Semantic Versioning where practical.

## [Unreleased]

### Planned
- Multi-version CI matrix validation refinements (Python 3.10–3.13).
- Additional unit tests for optional TensorFlow / implicit code paths to raise coverage.

### Pending Documentation
- Expanded testing strategy section.

## [0.3.0] - 2025-08-15

### Added
- Optional dependency guards for TensorFlow, Keras, and implicit recommenders; modules import cleanly without heavy deps.
- Test wrapper server class introduced for MCP tooling exposure in [server.py](src/tie_mcp/server.py).
- Conditional recommender exposure logic in [recommender/__init__.py](src/tie_mcp/core/tie/recommender/__init__.py).

### Changed
- Refactored factorization recommender to remove nested bare except/pass blocks (Bandit B110) in [factorization_recommender.py](src/tie_mcp/core/tie/recommender/factorization_recommender.py).
- Renamed SQLAlchemy Dataset.metadata attribute to extra_metadata while preserving DB column name "metadata" in [database.py](src/tie_mcp/storage/database.py).
- Relaxed type annotations (use of Any) to avoid hard TensorFlow import-time requirements across recommender modules.

### Fixed
- Import-time crashes when TensorFlow / implicit not installed.
- Bandit B110 issues in factorization recommender.
- SQLAlchemy reserved attribute error for metadata attribute.

### Security
- Eliminated silent exception swallowing in factorization recommender initialization.

## [0.2.0] - 2025-08-10

### Added
- Documentation restructuring and architecture diagrams in [README.md](README.md).

### Changed
- Updated Ruff version and formatting project-wide.

## [0.1.0] - 2025-08-01
### Added
- Initial project structure, core TIE MCP components, recommender abstractions.