# CoreML Qwen2.5 Converter

This project provides tools to convert Qwen2.5 models to CoreML format and benchmark their performance against PyTorch models.

## Setup

### Requirements
- Python 3.11
- macOS 15 or later (required for CoreML)

### Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Exporting Qwen2.5 to CoreML

The export script converts Qwen2.5 models to CoreML format with both FP16 and INT4 quantized versions:

```bash
python export.py
```

This will create two model files:
- `StatefulQwen2.51.5BInstructFp16.mlpackage` - FP16 precision model
- `StatefulQwen2.51.5BInstructInt4Block32.mlpackage` - INT4 quantized model

### Benchmarking

The benchmark script allows you to compare performance between PyTorch and CoreML models in two modes:

1. Generation mode - Tests token generation with KV Cache performance:
```bash
python benchmark.py --benchmark-mode generation --prompt "Hello, my dear friend" --n-cycles 128 --print-output
```

2. Context mode - Tests context preparation performance (for large tensors):
```bash
python benchmark.py --benchmark-mode context --input-length 1024 --n-cycles 128
```

#### Benchmark Options

- `--torch-model`: Path or HuggingFace ID for the PyTorch model (default: "Qwen/Qwen2.5-1.5B-Instruct")
- `--coreml-model`: Path to the CoreML model package (default: "StatefulQwen2.51.5BInstructFp16.mlpackage")
- `--benchmark-mode`: Choose between "generation" or "context" mode
- `--n-cycles`: Number of generation cycles (default: 128)
- `--prompt`: Input prompt for text generation (generation mode only)
- `--print-output`: Print generated output (generation mode only)
- `--input-length`: Input length for context mode (default: 1024) 