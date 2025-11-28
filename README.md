# Chisel Image Processing Library

A Python library for image processing with skeletonization capabilities.

## Development Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Chisel-Image-Processing-Library
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/                 # Source code
├── tests/               # Test files
├── docs/                # Documentation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Development

### Running Tests

You can run tests in multiple ways:

#### Using Test Scripts (Recommended)

**macOS/Linux:**
```bash
./run_tests.sh
```

**Windows:**
```cmd
run_tests.cmd
```

The test scripts will:
- Run all tests with pytest
- Generate coverage reports (HTML and terminal)
- Display coverage metrics
- Exit with appropriate status codes

#### Using pytest Directly

```bash
pytest tests/
```

#### With Coverage Report

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

### Code Quality

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

## License

[Add license information here]

## Contributing

[Add contribution guidelines here]
