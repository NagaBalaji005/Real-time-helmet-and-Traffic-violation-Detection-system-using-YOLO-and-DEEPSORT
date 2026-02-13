"""
FastAPI Backend for Traffic Violation Detection System
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict, Optional
import json
import os
from datetime import datetime, timedelta
from pydantic import BaseModel

# Import pipeline components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import TrafficViolationPipeline
from database import get_db, DatabaseManager
from config import API_CONFIG

app = FastAPI(
    title="Traffic Violation Detection API",
    description="API for detecting and managing traffic violations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for dashboard
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

# Global pipeline instance
pipeline = None

# Pydantic models
class Violation(BaseModel):
    id: Optional[int] = None
    type: str
    plate_number: Optional[str] = None
    speed: Optional[float] = None
    speed_limit: Optional[float] = None
    timestamp: str
    location: Optional[str] = None
    image_path: Optional[str] = None
    confidence: Optional[float] = None

class VideoUpload(BaseModel):
    filename: str
    content_type: str

class ProcessingRequest(BaseModel):
    video_path: str
    model_path: Optional[str] = None
    device: str = "auto"
    save_output: bool = True

# Database manager instance
db_manager = DatabaseManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup"""
    global pipeline
    try:
        # Use the newly trained refined model
        model_path = 'runs/train/final_model_refined/weights/best.pt'
        pipeline = TrafficViolationPipeline(model_path=model_path)
        print("Pipeline initialized successfully with refined model")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Traffic Violation Detection API",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global pipeline
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_initialized": pipeline is not None,
        "components": {}
    }
    
    if pipeline:
        try:
            # Check detector
            detector_info = pipeline.detector.get_model_info()
            health_status["components"]["detector"] = {
                "status": "ok",
                "device": detector_info.get("device", "unknown"),
                "model_path": detector_info.get("model_path", "unknown")
            }
            
            # Check tracker
            tracker_stats = pipeline.tracker.get_tracker_stats()
            health_status["components"]["tracker"] = {
                "status": "ok",
                "active_tracks": tracker_stats.get("active_tracks", 0)
            }
            
            # Check speed estimator
            speed_info = pipeline.speed_estimator.get_estimator_info()
            health_status["components"]["speed_estimator"] = {
                "status": "ok",
                "fps": speed_info.get("fps", 0)
            }
            
            # Check OCR
            ocr_stats = pipeline.ocr.get_ocr_statistics()
            health_status["components"]["ocr"] = {
                "status": "ok",
                "languages": ocr_stats.get("languages", [])
            }
            
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["error"] = str(e)
    
    return health_status

@app.post("/process-video")
async def process_video(request: ProcessingRequest):
    """Process a video file for traffic violations"""
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        # Process video
        output_path = None
        if request.save_output:
            output_dir = "output/processed_videos"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        
        # Run processing
        pipeline.process_video(
            request.video_path,
            output_path,
            show_display=False,
            save_output=request.save_output
        )
        
        # Get results
        results = {
            "status": "completed",
            "video_path": request.video_path,
            "output_path": output_path,
            "total_violations": len(pipeline.violations),
            "processing_time": datetime.now().isoformat(),
            "violations": []
        }
        
        # Convert violations to API format
        for violation in pipeline.violations:
            violation_data = {
                "type": violation["type"],
                "plate_number": violation.get("plate_number"),
                "speed": violation.get("speed"),
                "speed_limit": violation.get("speed_limit"),
                "timestamp": violation.get("timestamp"),
                "image_path": violation.get("image_path"),
                "confidence": violation.get("confidence")
            }
            results["violations"].append(violation_data)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/violations")
async def get_violations(
    violation_type: Optional[str] = None,
    plate_number: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get violations with optional filtering"""
    try:
        # Get violations from database
        violations = db_manager.get_violations(limit=limit, offset=offset, violation_type=violation_type)
        
        # Convert to dict format for API response
        violation_list = []
        for violation in violations:
            violation_dict = {
                'id': violation.id,
                'type': violation.violation_type,
                'severity': violation.severity,
                'description': violation.description,
                'plate_number': violation.number_plate,
                'vehicle_type': violation.vehicle_type,
                'vehicle_color': violation.vehicle_color,
                'speed_limit': violation.speed_limit,
                'actual_speed': violation.actual_speed,
                'speed_unit': violation.speed_unit,
                'location': violation.location,
                'camera_id': violation.camera_id,
                'image_path': violation.image_path,
                'confidence': violation.confidence_score,
                'timestamp': violation.timestamp.isoformat()
            }
            violation_list.append(violation_dict)
        
        # Apply additional filters
        if plate_number:
            violation_list = [v for v in violation_list if v.get('plate_number') == plate_number]
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
                violation_list = [v for v in violation_list 
                                 if datetime.fromisoformat(v['timestamp']) >= start_dt]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
                violation_list = [v for v in violation_list 
                                 if datetime.fromisoformat(v['timestamp']) <= end_dt]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")
        
        return {
            "violations": violation_list,
            "total_count": len(violation_list),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/violations/{violation_id}")
async def get_violation(violation_id: int):
    """Get a specific violation by ID"""
    global violations_db
    
    for violation in violations_db:
        if violation["id"] == violation_id:
            return violation
    
    raise HTTPException(status_code=404, detail="Violation not found")

@app.post("/violations")
async def create_violation(violation: Violation):
    """Create a new violation record"""
    global violations_db, violation_id_counter
    
    violation_data = violation.dict()
    violation_data["id"] = violation_id_counter
    violation_data["created_at"] = datetime.now().isoformat()
    
    violations_db.append(violation_data)
    violation_id_counter += 1
    
    return violation_data

@app.get("/violations/{violation_id}/image")
async def get_violation_image(violation_id: int):
    """Get violation evidence image"""
    global violations_db
    
    for violation in violations_db:
        if violation["id"] == violation_id:
            image_path = violation.get("image_path")
            if image_path and os.path.exists(image_path):
                return FileResponse(image_path)
            else:
                raise HTTPException(status_code=404, detail="Image not found")
    
    raise HTTPException(status_code=404, detail="Violation not found")

@app.get("/stats")
async def get_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get violation statistics"""
    try:
        # Get statistics from database
        stats = db_manager.get_violation_stats()
        
        # Get recent violations for additional analysis
        recent_violations = db_manager.get_violations(limit=1000)
        
        # Speed violations analysis
        speed_violations = [v for v in recent_violations if v.violation_type == "speed"]
        avg_speed = 0
        max_speed = 0
        if speed_violations:
            speeds = [v.actual_speed or 0 for v in speed_violations]
            avg_speed = sum(speeds) / len(speeds)
            max_speed = max(speeds)
        
        # Recent activity (last 24 hours)
        recent_violations_24h = 0
        if recent_violations:
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_violations_24h = len([v for v in recent_violations 
                                       if v.timestamp >= cutoff_time])
        
        # Violations by date (last 7 days)
        violations_by_date = {}
        for violation in recent_violations:
            date = violation.timestamp.date().isoformat()
            violations_by_date[date] = violations_by_date.get(date, 0) + 1
        
        return {
            "total_violations": stats['total_violations'],
            "violation_types": stats['by_type'],
            "violations_by_severity": stats['by_severity'],
            "violations_by_date": violations_by_date,
            "speed_violations": {
                "count": len(speed_violations),
                "average_speed": round(avg_speed, 2),
                "max_speed": max_speed
            },
            "recent_violations_24h": recent_violations_24h,
            "period": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/dashboard")
async def get_dashboard():
    """Serve the dashboard HTML"""
    dashboard_path = "dashboard/index.html"
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    else:
        return {"message": "Dashboard not found. Please create dashboard/index.html"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing"""
    # Create upload directory
    upload_dir = "data/raw"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return {
        "message": "Video uploaded successfully",
        "filename": file.filename,
        "file_path": file_path,
        "file_size": len(content)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["debug"]
    )
