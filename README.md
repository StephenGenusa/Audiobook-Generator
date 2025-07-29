# Kokoro TTS Audiobook Generator

A production-ready Python pipeline that converts large text documents into high-quality MP3 audiobooks using the Kokoro text-to-speech model. Features intelligent text chunking, GPU acceleration with CPU fallback, and comprehensive error recovery for reliable processing of books, articles, and long-form content.

## Features

### Core Capabilities
- **High-Quality TTS**: Uses Kokoro neural TTS models with multiple voice options
- **Intelligent Text Processing**: Hierarchical chunking (paragraphs → sentences → forced splits) with token limit respect
- **Hardware Optimization**: Automatic GPU acceleration with seamless CPU fallback
- **Professional Audio Output**: 24kHz WAV generation with configurable MP3 compression
- **Robust Error Handling**: Comprehensive validation, recovery mechanisms, and detailed progress tracking

### Advanced Features
- **Multiple Voice Support**: American and British English variants (af_heart, af_bella, am_michael, etc.)
- **Configurable Parameters**: Speech speed, silence duration, audio quality, and processing options
- **Memory Management**: Temporary file handling with automatic cleanup
- **Production Logging**: Detailed progress tracking with optional debug output and filtered warnings

## Installation

### Prerequisites
- **Create a Python virtual environment**
- **Install Kokoro - pip install -q kokoro>=0.9.4 soundfile**
- **Force install GPU support if you have it from pytorch.org**


### System Requirements
- **Audio**: FFmpeg for MP3 conversion (installed with pydub)
- **Memory**: 2GB+ RAM recommended for large documents
- **Storage**: Temporary space ~3x final output size during processing

## Usage

### Basic Usage
```bash
python audiobook.py chapter01.txt chapter01.mp3
```

### Advanced Configuration
```bash
python audiobook.py book.txt audiobook.mp3 --voice af_bella --speed 1.2 --silence 2.0 --bitrate 192k --verbose
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--voice` | `af_heart` | Voice identifier (af_heart, af_bella, am_michael, etc.) |
| `--speed` | `1.0` | Speech rate multiplier (0.5-2.0) |
| `--silence` | `1.5` | Silence duration between chunks (seconds) |
| `--bitrate` | `128k` | MP3 encoding quality (128k, 192k, 256k) |
| `--cpu-only` | False | Disable GPU acceleration |
| `--verbose` | False | Enable debug logging |
| `--show-kokoro-warnings` | False | Show model warning messages |

## Architecture Overview

### Text Processing Pipeline
1. **Text Cleaning**: Whitespace normalization, paragraph detection
2. **Hierarchical Chunking**: 
   - Primary: Split by paragraphs (respecting semantic boundaries)
   - Fallback: Split oversized paragraphs by sentences
   - Emergency: Character-based splitting with word boundary preservation
3. **Token Validation**: Conservative estimation with pipeline verification for edge cases

### Audio Generation
- **Model Management**: Efficient GPU/CPU model loading with memory optimization
- **Voice Processing**: Kokoro pipeline with phoneme sequence generation
- **Quality Control**: Segment validation, concatenation with silence insertion

### Output Processing
- **Format Handling**: 24kHz WAV intermediate files with int16 precision
- **Concatenation**: Seamless audio merging with configurable silence gaps
- **Compression**: High-quality MP3 encoding with VBR optimization

## Performance Characteristics

### Processing Speed
- **GPU Mode**: ~150-200 characters/second (typical)
- **CPU Mode**: ~50-100 characters/second (varies by hardware)
- **Memory Usage**: ~2-4GB during processing (depends on text size and hardware)

### Quality Metrics
- **Audio Quality**: 24kHz sample rate, 16-bit depth
- **Compression**: ~8KB text per 1MB audio (128k MP3)
- **Reliability**: >95% chunk success rate on well-formatted text

## Examples

### High-Quality Academic Paper
```bash
python audiobook.py paper.txt paper_audiobook.mp3 --voice am_michael --speed 0.9 --bitrate 256k --silence 1.5
```

### CPU-Only Processing
```bash
python audiobook.py document.txt output.mp3 --cpu-only --verbose
```

## Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Use CPU-only mode for large documents
python audiobook.py input.txt output.mp3 --cpu-only
```

**Text Encoding Issues**
- Ensure input files are UTF-8 encoded
- Tool auto-detects latin-1 and cp1252 as fallbacks

**Audio Quality Concerns**
```bash
# Increase bitrate and reduce speed for better clarity
python audiobook.py input.txt output.mp3 --bitrate 256k --speed 0.8
```

### Performance Optimization
- **Large Files**: CPU-only mode often more stable for >100k characters
- **Quality vs Speed**: Lower speed (0.8-0.9) improves pronunciation clarity
- **Memory**: Close other GPU applications before processing

## Error Handling

The tool implements comprehensive error recovery:
- **Text Processing**: Multiple fallback chunking strategies
- **Audio Generation**: Per-chunk error isolation with continuation
- **Hardware Failures**: Automatic GPU-to-CPU fallback
- **File Operations**: Temporary directory cleanup and validation

Failed chunks are logged but don't halt processing, ensuring maximum content recovery.

## Technical Details

### Dependencies
- **kokoro**: Core TTS model and phoneme processing
- **pydub**: Audio format conversion and concatenation
- **torch**: Neural network execution and GPU acceleration
- **numpy**: Audio array processing and numerical operations  

### File Processing
- **Chunking**: Respects 450-token limit with semantic preservation
- **Audio Format**: 24kHz/16-bit WAV → MP3 conversion pipeline
- **Memory Management**: Streaming processing with temporary file cleanup

## License

MIT License

Copyright (c) 2025 by Stephen Genusa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

This is a production-ready tool with extensive error handling and validation. When contributing:

1. **Maintain Error Recovery**: Preserve the multi-level fallback mechanisms
2. **Test Edge Cases**: Validate with malformed text, GPU failures, and memory constraints  
3. **Preserve Performance**: Benchmark changes against large document processing
4. **Documentation**: Update README for new features or significant changes

## Acknowledgments

- Built on the Kokoro TTS model for high-quality neural speech synthesis
- Implements production-grade text processing patterns for TTS pipelines
- Designed for reliable processing of large documents with comprehensive error handling

