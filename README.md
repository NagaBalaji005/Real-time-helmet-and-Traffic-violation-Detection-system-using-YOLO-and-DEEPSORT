# 🚦 AI-Powered Traffic Violation Detection System

A comprehensive traffic violation detection system using **YOLOv8**, **PaddleOCR**, **PostgreSQL**, and **FastAPI** with a modern web dashboard.

## 🎯 System Overview

This system detects and tracks various traffic violations in real-time from video streams and images, providing law enforcement with automated violation detection capabilities.

### 🔍 Key Features

### ✅ Recent Improvements (Latest Update)

1. **Strict Plate Validation**: 
   - Validates complete Indian plate format (AA BB CC DDDD)
   - Rejects incomplete plates (e.g., "BR 01 8627" - missing series)
   - State code validation (only valid Indian state codes accepted)
2. **Smart Duplicate Handling**:
   - Same image: Violations are combined (e.g., pu-1 with no_helmet + mobile_usage = 1 record)
   - Different images: Same plate creates separate records
3. **Driver-Focused Helmet Detection**: Only checks driver helmet status

## 🔍 Detection Capabilities

- **Helmet Violations**: Detects riders with/without helmets (driver only, not passengers)
  - If driver has helmet, no violation is recorded (even if passenger doesn't have helmet)
- **Mobile Usage**: Identifies drivers using mobile phones
- **Triple Riding**: Detects triple riding violations
- **Speed Monitoring**: Calculates vehicle speed and detects overspeeding
- **Number Plate Recognition**: Extracts license plate numbers using **PaddleOCR** with:
  - Combined raw text (no normalization that breaks valid plates)
  - Strict Indian plate format validation (AA BB CC DDDD)
  - State code validation (rejects invalid codes like JS)
  - Complete remapping for all 4 parts
- **Vehicle Tracking**: Tracks vehicles across frames for consistent monitoring
- **Multi-Vehicle Support**: Accurately maps violations to their respective number plates
- **Smart Deduplication**: 
  - Records only one violation per type per vehicle (driver-focused)
  - Same image violations are combined
  - Different images with same plate create separate records

## 🚀 Quick Start

### 1. Start the System
```bash
python simple_server.py
```

### 2. Access Dashboard
Open your browser and go to: **http://localhost:8081/dashboard**

### 3. Upload Media
- **Images**: Upload traffic images for instant violation detection
- **Videos**: Upload traffic videos for comprehensive analysis

## 🔧 Technical Details

### OCR Processing (PaddleOCR)
- **No Normalization**: Uses combined raw text directly to preserve correct plate format
- **Strict Validation**: 
  - State code must be valid Indian state code
  - RTO code must be exactly 2 digits
  - Series is optional but must be 1-3 letters if present
  - Number must be exactly 4 digits
- **Remapping**: Applies character corrections for common OCR errors
- **Rejects Invalid Plates**: Incomplete plates (missing parts) are rejected

### Violation Grouping
- **Same Image**: All violations from same image are combined into one record
- **Different Images**: Same plate in different images creates separate records
- **Example**: 
  - pu-1.jpg with no_helmet + mobile_usage → 1 combined record
  - pu-1.jpg and pu-2.jpg both with same plate → 2 separate records

## 📁 Project Structure

```
final-year-project/
├── 📄 Core Files
│   ├── simple_server.py              # Main FastAPI server
│   ├── improved_plate_processor.py    # Video processing pipeline (with PaddleOCR)
│   ├── improved_image_processor.py    # Image processing pipeline (with PaddleOCR)
│   ├── plate_ocr_paddle.py           # Standalone PaddleOCR plate recognition
│   ├── train.py                      # Model training script
│   └── setup_database.py             # Database setup utility
│
├── 🌐 Web Interface
│   └── dashboard/
│       └── index.html                 # Modern web dashboard
│
├── 🤖 AI Models & Training
│   ├── runs/train/final_model_refined/  # Trained YOLOv8 refined model
│   │   ├── weights/best.pt            # Best model weights
│   │   ├── results.csv                 # Training metrics
│   │   └── *.png                       # Training visualizations
│   └── dataset/
│       ├── data.yaml                   # Dataset configuration
│       ├── raw/                         # Input videos
│       ├── train/                       # Training data
│       │   ├── images/                  # Training images
│       │   └── labels/                 # Training labels
│       └── val/                         # Validation data
│           ├── images/                  # Validation images
│           └── labels/                  # Validation labels
│
├── 🔧 Core Modules (src/)
│   ├── config.py                     # System configuration
│   ├── detector.py                   # YOLOv8 detection engine (PyTorch 2.6 compatible)
│   ├── pipeline.py                   # Main processing pipeline
│   ├── database.py                   # PostgreSQL operations
│   ├── ocr.py                        # PaddleOCR number plate recognition (UltimateOCR)
│   ├── tracker.py                    # Vehicle tracking
│   ├── speed.py                      # Speed calculation
│   └── api/main.py                   # API endpoints
│
├── 📊 Output & Results
│   ├── output/                       # Processing results
│   ├── uploads/                      # Uploaded files
│   └── processed_output/             # Processed media
│
├── 🗄️ Database
│   ├── database_setup.sql            # Database schema
│   └── requirements.txt              # Python dependencies
│
└── 🐍 Virtual Environment
    └── venv/                         # Python virtual environment
```

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd final-year-project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Database (Optional)
```bash
python setup_database.py
```

### 5. Start the System
```bash
python simple_server.py
```

## 🎬 Usage Examples

### Process Images
```bash
python improved_image_processor.py dataset/train/images/new1.jpg
```

### Process Videos
```bash
python improved_plate_processor.py dataset/raw/traffic_video.mp4
```

### Standalone OCR Testing
```bash
python plate_ocr_paddle.py dataset/train/images/new1.jpg
```

### Upload via Dashboard
1. Open **http://localhost:8081/dashboard**
2. Click **"Upload Media"**
3. Drag & drop or select image/video file
4. Click **"Process Media"**
5. View results in the dashboard

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dashboard` | GET | Web dashboard interface |
| `/stats` | GET | System statistics |
| `/violations` | GET | List of violations |
| `/upload-video` | POST | Upload & process media |
| `/health` | GET | System health check |
| `/docs` | GET | API documentation |

## 🎯 Detection Classes

The system detects **5 main classes**:

1. **helmet** - Riders wearing helmets ✅
2. **no_helmet** - Riders without helmets ❌
3. **mobile_usage** - Drivers using phones 📱
4. **triple_riding** - Triple riding violations 🏍️
5. **number_plate** - License plates 🚗

## 📈 Model Performance

### Training Results
- **mAP50**: 93.4% (excellent detection accuracy)
- **mAP50-95**: 50.4% (good localization precision)
- **Training Time**: ~48 hours on CPU
- **Dataset Size**: 180 training + 144 validation images

### Detection Performance
- **Processing Speed**: ~8-10 FPS on CPU
- **Confidence Threshold**: 0.7 (configurable)
- **Supported Formats**: MP4, AVI, MOV, JPG, PNG
- **Resolution**: Up to 4K supported

## 🔧 Configuration

### Model Settings (`src/config.py`)
```python
# Detection Classes
CLASSES = {
    0: 'helmet',
    1: 'no_helmet', 
    2: 'mobile_usage',
    3: 'triple_riding',
    4: 'number_plate'
}

# Detection Thresholds
CONFIDENCE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5

# Speed Monitoring
SPEED_LIMIT = 50  # km/h
SPEED_THRESHOLD = 1.2  # 20% over limit
```

### Dataset Configuration (`dataset/data.yaml`)
```yaml
path: D:/final-year-project/dataset
train: train/images
val: val/images
nc: 5
names:
  0: helmet
  1: no_helmet
  2: mobile_usage
  3: triple_riding
  4: number_plate
```

## 🔄 System Workflow

1. **Media Upload**: User uploads image/video via dashboard
2. **File Detection**: System detects file type (image/video)
3. **Processing**: 
   - Images → `improved_image_processor.py`
   - Videos → `improved_plate_processor.py`
4. **Detection**: YOLOv8 detects violations in frames
5. **OCR**: EasyOCR extracts number plates
6. **Tracking**: Vehicles tracked across frames
7. **Speed Calculation**: Speed estimated from tracking
8. **Results**: Violations logged and displayed in dashboard

## 📊 Output Files

### Processing Results
- **`violations_log.json`**: Complete violation records
- **`annotated_*.jpg`**: Images with detection boxes
- **`detected_*.mp4`**: Videos with violation annotations
- **`vehicle_summary.json`**: Vehicle tracking summary

### Dashboard Data
- Real-time violation statistics
- Interactive violation table
- System performance metrics
- Upload and processing status

## 🎓 Educational Value

This project demonstrates:
- **Computer Vision**: YOLOv8 object detection
- **OCR Technology**: PaddleOCR number plate recognition with preprocessing
- **Web Development**: FastAPI + HTML/CSS/JS
- **Database Design**: PostgreSQL integration
- **System Integration**: End-to-end pipeline
- **AI Model Training**: Custom dataset preparation
- **Spatial Analysis**: IoU-based object association
- **Multi-vehicle Tracking**: Accurate violation-to-plate mapping

## 📝 License

This project is for educational and research purposes.

## 🎉 **AI-powered traffic violation detection system is ready!**

**Dashboard**: http://localhost:8081/dashboard  
**API Docs**: http://localhost:8081/docs  
**Model**: `runs/train/final_model_refined/weights/best.pt`  
**OCR Engine**: PaddleOCR with UltimateOCR pipeline  
**Performance**: High accuracy with smart violation mapping

## 🔑 Key Features

- ✅ **PyTorch 2.6+ Compatible**: Automatic compatibility patches
- ✅ **PaddleOCR Integration**: Advanced OCR with preprocessing
- ✅ **Smart Violation Mapping**: Accurate multi-vehicle support
- ✅ **Driver-Focused Detection**: One violation per type per vehicle
- ✅ **Adaptive Distance**: Image-size aware spatial matching
- ✅ **Quality Scoring**: OCR confidence and quality metrics

**All rights reserved@2026**
