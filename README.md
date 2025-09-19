# Monitrix

An advanced monitoring tool with OCR and image recognition capabilities for extracting numerical values from medical monitoring equipment displays.

## Features

- **🔍 Object Detection**: YOLO-based detection of monitoring displays and numerical regions
- **📖 OCR Recognition**: EasyOCR-powered text recognition for extracting numerical values
- **📐 Perspective Correction**: Automatic perspective transformation for distorted displays
- **🎯 Multi-Object Tracking**: Stationary object tracking with IoU-based continuity
- **📊 Performance Metrics**: Comprehensive evaluation metrics with accuracy, precision, recall
- **🎬 Video Processing**: Batch processing and video output generation
- **⚡ GPU Acceleration**: CUDA support for high-performance inference

## Installation

### Requirements

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install ultralytics>=8.3.0
pip install easyocr>=1.7.0
pip install opencv-python>=4.5.0
pip install numpy>=1.25.0
pip install torch torchvision
pip install scikit-learn
pip install matplotlib
pip install pandas
pip install tqdm
```

### Install Monitrix

```bash
# From source
git clone https://github.com/your-username/monitrix.git
cd monitrix
pip install -e .
```

## Quick Start

### Basic Usage

```python
from monitrix.yolo import YOLOc300, YOLOInitConfig, YOLOPredictConfig
from monitrix.eocr import EasyocrNumberReader
from monitrix.manager import ImageGroupTextReader
from monitrix.geometry import coordinate_cv2jit
from monitrix.projective import ProjectiveMatrixCV2

# Initialize detector
detector = YOLOc300()

# Initialize OCR reader
ocr_reader = EasyocrNumberReader()

# Create image processor
processor = ImageGroupTextReader(
    reader=ocr_reader,
    result_type=type(ocr_reader.result_type),
    projecter=ProjectiveMatrixCV2,
    coordinate=coordinate_cv2jit,
    perspective=True,
    aspect_ratio=4/3
)

# Process video or images
results = []
for result in detector.predict("path/to/video.mp4", stream=True):
    processed_result = processor.apply(result)
    results.append(processed_result)
```

### Video Processing

```python
from monitrix.resultsobject import VideoResults

# Collect results
video_results = VideoResults(results)

# Export to CSV
video_results.to_csv("results.csv")

# Generate output video
video_results.to_video("output.mp4", fps=30)

# Get DataFrame for analysis
df = video_results.to_df()
print(df.head())
```

### Performance Evaluation

```python
from monitrix.metric import mdataframe

# Convert results to metric DataFrame
mdf = mdataframe(df)

# Add ground truth data
mdf_with_truth = mdf.add_true(ground_truth_df)

# Calculate metrics
metrics = mdf_with_truth.metrics(group=["monitor_id", "class"])

# Get scores
for metric in metrics:
    scores = metric.scores()
    print(f"Accuracy: {scores['accuracy']:.3f}")
    print(f"Precision: {scores['precision_micro']:.3f}")
    print(f"Recall: {scores['recall_micro']:.3f}")
    print(f"F1-Score: {scores['f1_micro']:.3f}")
```

## Configuration

### YOLO Detection Settings

```python
from monitrix.yolo import YOLOInitConfig, YOLOPredictConfig

# Initialize configuration
init_config = YOLOInitConfig(
    model="path/to/model.pt",
    task="segment",
    verbose=False
)

# Prediction configuration
predict_config = YOLOPredictConfig(
    conf=0.83,
    iou=0.7,
    max_det=15,
    imgsz=736,
    batch=16
)

detector = YOLOc300(init_config)
results = detector.predict(source, config=predict_config)
```

### OCR Settings

```python
from monitrix.eocr import EasyocrInitConfig, EasyocrPredictConfig

# OCR initialization
ocr_init = EasyocrInitConfig(
    lang_list=["en"],
    gpu=True,
    detector=True
)

# OCR prediction settings
ocr_predict = EasyocrPredictConfig(
    allowlist="0123456789.",
    min_size=10,
    contrast_ths=0.3,
    batch_size=1
)

reader = EasyocrNumberReader(ocr_init, ocr_predict)
```

## Architecture

### Core Components

- **Detector**: Abstract base class for object detection implementations
- **NumberReader**: OCR-based numerical value extraction
- **PostProcess**: Pipeline for result refinement and tracking
- **Projective**: Perspective transformation for view correction
- **Coordinate**: Geometric operations and shape manipulation

### Processing Pipeline

1. **Object Detection**: Identify monitoring displays and numerical regions
2. **Tracking**: Maintain object continuity across frames
3. **Perspective Correction**: Apply geometric transformations
4. **OCR Recognition**: Extract numerical values from regions
5. **Post-Processing**: Filter and validate results
6. **Evaluation**: Calculate performance metrics

## Supported Formats

### Input

- **Images**: JPG, PNG, BMP, TIFF
- **Videos**: MP4, AVI, MOV
- **Streams**: RTSP, USB cameras
- **Batch**: Multiple image files

### Output

- **CSV**: Structured numerical data
- **Video**: Annotated output videos
- **JSON**: Serialized results
- **DataFrames**: Pandas-compatible analysis

## Performance

Monitrix is optimized for real-time monitoring applications:

- **GPU Acceleration**: CUDA support for inference
- **Batch Processing**: Efficient multi-image handling
- **Memory Management**: Automatic CUDA memory cleanup
- **Vectorized Operations**: NumPy and PyTorch optimizations

## License

This project is licensed under the GNU Affero General Public License v3 or later (AGPLv3+). See the [LICENSE](LICENSE) file for details.

## Citation

If you use Monitrix in your research, please cite:

```bibtex
@software{monitrix,
  title={Monitrix: Advanced Monitoring Tool with OCR and Image Recognition},
  author={Arisaka, Naoya},
  year={2025},
  url={https://github.com/your-username/monitrix}
}
```

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition
- [OpenCV](https://opencv.org/) for computer vision operations
- [PyTorch](https://pytorch.org/) for deep learning framework
