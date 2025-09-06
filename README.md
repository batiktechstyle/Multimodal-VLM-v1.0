# Multimodal VLM v1.0

https://github.com/user-attachments/assets/4851bd73-0e7c-4056-9d30-21f772d3b47d

A comprehensive multimodal vision-language model application supporting both image and video inference tasks. This application integrates four specialized models for document processing, OCR, spatial reasoning, and video understanding.

## System Requirements

- **GPU**: NVIDIA H200 MIG 3g.71gb (104GB VRAM)
- **CUDA**: Version 12.8
- **PyTorch**: 2.8.0+cu128
- **Python**: 3.10+

## Supported Models

### Camel-Doc-OCR-062825
- **Base**: Qwen2.5-VL-7B-Instruct
- **Specialization**: Document retrieval, structured extraction, analysis, and direct Markdown generation
- **Use Cases**: PDF processing, document analysis, OCR tasks

### Megalodon-OCR-Sync-0713
- **Base**: Qwen2.5-VL-3B-Instruct
- **Specialization**: Context-aware multimodal document extraction and analysis
- **Strengths**: Layout parsing, mathematical content, chart and table recognition

### ViLaSR-7B
- **Focus**: Spatial reasoning in visual-language tasks
- **Features**: Interwoven thinking with visual drawing capabilities
- **Best For**: Complex spatial reasoning queries and tip-based analysis
- **Note**: Demo supports text-only reasoning, which may underrepresent full model capabilities

### Video-MTR
- **Specialization**: Multi-turn reasoning for long-form video understanding
- **Capabilities**: Iterative key segment selection and deeper question comprehension
- **Warning**: May not perform optimally on all video inference tasks in this deployment

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers gradio spaces pillow opencv-python numpy qwen-vl-utils
```

## Usage

### Image Inference
1. Select desired model from the radio button options
2. Enter your query in the text input field
3. Upload an image file (PNG, JPG, JPEG supported)
4. Adjust advanced parameters if needed
5. Click Submit to generate response

### Video Inference
1. Choose appropriate model (Video-MTR recommended for video tasks)
2. Input your video-related query
3. Upload video file (MP4 format supported)
4. Configure generation parameters as needed
5. Submit for processing

## Advanced Configuration

### Generation Parameters
- **Max New Tokens**: 1 to 16,384 (default: 8,192)
- **Temperature**: 0.1 to 4.0 (default: 0.6)
- **Top-p**: 0.05 to 1.0 (default: 0.9)
- **Top-k**: 1 to 1,000 (default: 50)
- **Repetition Penalty**: 1.0 to 2.0 (default: 1.2)

### Input Limitations
- **Maximum Input Token Length**: 4,096 tokens
- **Video Processing**: Automatically downsamples to 10 evenly spaced frames
- **Image Formats**: PIL-compatible formats (PNG, JPEG, etc.)

## Example Use Cases

### Document Processing
```
Query: "convert this page to doc [text] precisely for markdown"
Use Model: Camel-Doc-OCR-062825 or Megalodon-OCR-Sync-0713
```

### Mathematical Content
```
Query: "fill the correct numbers"
Use Model: Megalodon-OCR-Sync-0713
```

### Video Analysis
```
Query: "explain the video in detail"
Use Model: Video-MTR
```

### Table Extraction
```
Query: "convert this page to doc [table] precisely for markdown"
Use Model: Camel-Doc-OCR-062825 or Megalodon-OCR-Sync-0713
```

## Technical Architecture

### Model Loading
- All models loaded with `torch.float16` precision for memory efficiency
- Models deployed on single CUDA device with automatic device detection
- Processor and model instances cached for optimal performance

### Video Processing Pipeline
- Automatic frame extraction using OpenCV
- Uniform frame sampling (10 frames per video)
- Timestamp preservation for temporal context
- RGB color space conversion for model compatibility

### Streaming Generation
- Real-time text streaming using `TextIteratorStreamer`
- Threaded generation for non-blocking UI
- Buffer management for smooth output delivery

## Performance Considerations

- **GPU Memory**: Models require significant VRAM (104GB available)
- **Processing Speed**: Varies by model size and input complexity
- **Concurrent Users**: Queue system supports up to 40 simultaneous requests
- **Video Processing**: Frame extraction adds processing overhead

## Limitations

1. **Video Model Performance**: Current deployment may not reflect optimal video inference capabilities
2. **ViLaSR Functionality**: Demo environment limits full spatial reasoning features
3. **Token Constraints**: Input length capped at 4,096 tokens
4. **Format Support**: Limited to specific image and video formats

## Development Notes

- Application built with Gradio for web interface
- Spaces GPU decorator for resource management
- Custom CSS styling with storj_theme
- Error handling and validation throughout processing pipeline
- Support for both raw output and formatted markdown display

## Troubleshooting

### Common Issues
- **CUDA Out of Memory**: Reduce max_new_tokens or batch size
- **Model Loading Errors**: Verify model availability and trust_remote_code settings
- **Video Processing Failures**: Check video format and file integrity
- **Slow Generation**: Monitor GPU utilization and consider parameter adjustment

### System Status
- CUDA Available: True
- Device Count: 1
- Current Device: 0 (NVIDIA H200 MIG 3g.71gb)

## Contributing

Report bugs and issues through the Hugging Face Spaces discussion board. Feature requests and improvements welcome through the official repository channels.
