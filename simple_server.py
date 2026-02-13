#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple FastAPI Server for Traffic Violation Detection
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add src to path for database imports
sys.path.append('src')

# Try to import database manager
try:
    from database import DatabaseManager
    db_manager = DatabaseManager()
    print("✅ Database connected successfully")
    USE_REAL_DB = True
except Exception as e:
    print(f"⚠️ Database connection failed: {e}")
    print("📝 Using mock data instead")
    db_manager = None
    USE_REAL_DB = False

# Create FastAPI app
app = FastAPI(title="Traffic Violation Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for dashboard assets and images
app.mount("/dashboard", StaticFiles(directory="dashboard"), name="dashboard")
app.mount("/output", StaticFiles(directory="output"), name="output")

# Serve traffic.png from root
@app.get("/traffic.png")
async def get_traffic_image():
    return FileResponse("traffic.png", media_type='image/png')

# Load real violation data from demo output
def load_real_violations():
    """Load real violations from output directory"""
    try:
        import json
        violations_file = 'output/violations_log.json'
        
        violations = []
        if os.path.exists(violations_file):
            try:
                with open(violations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # ✅ FIX: Handle both list format (new) and dict format (old)
                if isinstance(data, list):
                    # New format: violations_log.json is a flat list
                    violations = data
                elif isinstance(data, dict) and 'violations' in data:
                    # Old format: violations are in a 'violations' key
                    violations.extend(data['violations'])
                else:
                    print(f"⚠️ Unexpected violations file format: {type(data)}")
                    return []
            except Exception as e:
                print(f"Error loading violations file: {e}")
                return []
        
        # Convert to dashboard format
        formatted_violations = []
        for i, violation in enumerate(violations[:50]):  # Limit to 50 for performance
            # Skip violations without valid plate numbers
            plate_number = violation.get('plate_number', 'DEMO123')
            if not plate_number or str(plate_number).startswith('TRACK_'):
                continue
                
            formatted_violations.append({
                "id": i + 1,
                "type": violation.get('type', 'unknown'),
                "plate_number": plate_number,
                "speed": violation.get('speed', 45),
                "speed_limit": 50,
                "timestamp": violation.get('timestamp', datetime.now().isoformat()),
                "location": violation.get('location', 'Traffic Camera'),
                "confidence": violation.get('confidence', 0.8)
            })
        return formatted_violations
    except Exception as e:
        print(f"Error loading real violations: {e}")
        return []


# Load real violations
real_violations = load_real_violations()

# Track processed images to avoid reprocessing
processed_images = set()
processed_images_file = "output/processed_images.txt"

# Load previously processed images
if os.path.exists(processed_images_file):
    with open(processed_images_file, 'r', encoding='utf-8') as f:
        processed_images = set(line.strip() for line in f if line.strip())
    print(f"📊 Loaded {len(processed_images)} previously processed images")

@app.get("/")
async def root():
    return {"message": "Traffic Violation Detection API is running!"}

@app.get("/stats")
async def get_stats():
    """Get violation statistics"""
    print("📊 /stats endpoint called")
    if USE_REAL_DB and db_manager:
        try:
            # Get real statistics from database
            stats = db_manager.get_violation_stats()
            
            # ✅ FIX: Apply legacy class name mapping to database stats
            if 'violation_types' in stats:
                violation_types = stats['violation_types']
                
                # Map phone_usage -> mobile_usage
                if 'phone_usage' in violation_types:
                    violation_types['mobile_usage'] = violation_types.get('mobile_usage', 0) + violation_types.pop('phone_usage')
                
                # Map traffic_violation -> triple_riding
                if 'traffic_violation' in violation_types:
                    violation_types['triple_riding'] = violation_types.get('triple_riding', 0) + violation_types.pop('traffic_violation')
                
                # Ensure all keys exist
                violation_types.setdefault('helmet', 0)
                violation_types.setdefault('speed', 0)
                violation_types.setdefault('mobile_usage', 0)
                violation_types.setdefault('triple_riding', 0)
                
                stats['violation_types'] = violation_types
            
            print(f"✅ Returning stats from database: {stats}")
            return stats
        except Exception as e:
            print(f"❌ Error getting stats from database: {e}")
            # Fallback to mock data
            pass
    
    # Reload violations from files (dynamic loading)
    current_violations = load_real_violations()
    print(f"📊 Loaded {len(current_violations)} violations from file")
    
    if current_violations:
        # Count violation types - count ALL violations, not just unique plates
        violation_types = {}
        for violation in current_violations:
            vtype = violation.get('type', '')
            # ✅ FIX: Map legacy class names
            if vtype == 'phone_usage':
                vtype = 'mobile_usage'
            if vtype == 'traffic_violation':
                vtype = 'triple_riding'
            
            if vtype == 'no_helmet':
                violation_types['helmet'] = violation_types.get('helmet', 0) + 1
            elif vtype == 'mobile_usage':
                violation_types['mobile_usage'] = violation_types.get('mobile_usage', 0) + 1
            elif vtype == 'overspeed':
                violation_types['speed'] = violation_types.get('speed', 0) + 1
            elif vtype == 'triple_riding':
                violation_types['triple_riding'] = violation_types.get('triple_riding', 0) + 1
        
        stats = {
            "total_violations": len(current_violations),
            "recent_violations_24h": len(current_violations),
            "violation_types": violation_types
        }
        print(f"✅ Returning stats: {stats}")
        return stats
    
    # Fallback if no real data
    print("⚠️ No violations found, returning zero stats")
    return {
        "total_violations": 0,
        "recent_violations_24h": 0,
        "violation_types": {
            "helmet": 0,
            "speed": 0,
            "mobile_usage": 0,
            "triple_riding": 0
        }
    }

@app.get("/violations")
async def get_violations(limit: int = 50):
    """Get violations list grouped by plate number"""
    print(f"📊 /violations endpoint called with limit={limit}")
    print(f"📊 USE_REAL_DB={USE_REAL_DB}, db_manager={'available' if db_manager else 'not available'}")
    
    if USE_REAL_DB and db_manager:
        try:
            print("📊 Attempting to get violations from database...")
            # Get real violations from database
            violations = db_manager.get_violations(limit=limit * 2)  # Get more to allow grouping
            print(f"📊 Retrieved {len(violations)} raw violations from database")
            
            # Debug: print first few violations
            if violations:
                print(f"🔍 Sample violation from DB: {violations[0].__dict__ if hasattr(violations[0], '__dict__') else violations[0]}")
            else:
                print("⚠️ No violations returned from database - checking if database is empty or query is filtering")
            
            # Group violations by plate number and source image only (not timestamp)
            # Same image = combine violations, different images = separate records
            grouped_violations = {}
            for violation in violations:
                plate = violation.number_plate or 'Unknown'
                source_img = getattr(violation, 'source_image', 'unknown')
                # Use plate + source image only
                group_key = f"{plate}_{source_img}"

                if group_key not in grouped_violations:
                    grouped_violations[group_key] = {
                        "id": len(grouped_violations) + 1,
                        "plate_number": plate,
                        "violations": {},  # Change to dict to aggregate by type
                        "first_timestamp": violation.timestamp.isoformat(),
                        "location": violation.location,
                        "total_violations": 0,
                        "confidence": violation.confidence_score or 0.8
                    }

                # Aggregate violations by type
                v_type = violation.violation_type
                # ✅ FIX: Map legacy class names
                if v_type == 'phone_usage':
                    v_type = 'mobile_usage'
                if v_type == 'traffic_violation':
                    v_type = 'triple_riding'
                
                if v_type not in grouped_violations[group_key]["violations"]:
                    grouped_violations[group_key]["violations"][v_type] = {
                        "type": v_type,
                        "count": 0,
                        "speed": violation.actual_speed,
                        "speed_limit": violation.speed_limit,
                        "timestamp": violation.timestamp.isoformat(),
                        "confidence": violation.confidence_score,
                        "description": violation.description,
                        "severity": violation.severity
                    }
                grouped_violations[group_key]["violations"][v_type]["count"] += 1
                # Set total_violations to number of unique types
                grouped_violations[group_key]["total_violations"] = len(grouped_violations[group_key]["violations"])

            # Convert to list and sort by timestamp (most recent first)
            formatted_violations = list(grouped_violations.values())
            formatted_violations.sort(key=lambda x: x["first_timestamp"], reverse=True)
            
            print(f"📊 Converted {len(formatted_violations)} violations to dict format")
            print(f"✅ Returning {min(limit, len(formatted_violations))} grouped violations from database")

            return {
                "violations": formatted_violations[:limit],
                "total": len(formatted_violations)
            }
        except Exception as e:
            print(f"❌ Error getting violations from database: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to file-based data
            pass

    # Reload violations from files (dynamic loading) and group them
    current_violations = load_real_violations()
    print(f"📊 Loaded {len(current_violations)} violations from file")
    
    if current_violations:
        # Group by plate number and source image only (not timestamp)
        # Same image = combine violations, different images = separate records
        grouped_violations = {}
        for violation in current_violations:
            plate = violation.get('plate_number', 'Unknown')
            source_img = violation.get('source_image', violation.get('source', 'unknown'))
            # Create unique group key: plate + source image only
            # Same image violations are combined, different images create separate records
            group_key = f"{plate}_{source_img}"

            timestamp = violation.get('timestamp', datetime.now().isoformat())
            
            if group_key not in grouped_violations:
                grouped_violations[group_key] = {
                    "id": len(grouped_violations) + 1,
                    "plate_number": plate,
                    "violations": {},  # Change to dict to aggregate by type
                    "first_timestamp": timestamp,
                    "location": violation.get('location', 'Traffic Camera'),
                    "total_violations": 0,
                    "confidence": violation.get('confidence', 0.8)
                }

            # Aggregate violations by type - handle legacy naming
            v_type = violation.get('type', '')
            # ✅ FIX: Map legacy class names
            if v_type == 'phone_usage':
                v_type = 'mobile_usage'
            if v_type == 'traffic_violation':
                v_type = 'triple_riding'
            
            if v_type not in grouped_violations[group_key]["violations"]:
                grouped_violations[group_key]["violations"][v_type] = {
                    "type": v_type,
                    "count": 0,
                    "speed": violation.get('speed'),
                    "speed_limit": violation.get('speed_limit'),
                    "timestamp": timestamp,
                    "confidence": violation.get('confidence', 0.8),
                    "description": f"{violation['type']} violation detected",
                    "severity": violation.get('severity', 'medium')
                }
            grouped_violations[group_key]["violations"][v_type]["count"] += 1
            # Set total_violations to number of unique types
            grouped_violations[group_key]["total_violations"] = len(grouped_violations[group_key]["violations"])

        # Convert to list and sort by timestamp
        formatted_violations = list(grouped_violations.values())
        formatted_violations.sort(key=lambda x: x["first_timestamp"], reverse=True)

        print(f"✅ Returning {min(limit, len(formatted_violations))} grouped violations from file")
        return {
            "violations": formatted_violations[:limit],
            "total": len(formatted_violations)
        }

    # Fallback if no real data
    print("⚠️ No violations found, returning empty list")
    return {
        "violations": [],
        "total": 0
    }

@app.get("/dashboard")
async def dashboard():
    """Serve the dashboard"""
    return FileResponse("dashboard/index.html")

@app.post("/upload-video")
async def upload_media(file: UploadFile = File(...)):
    """Upload and process image or video for violations"""
    global real_violations
    try:
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Check if this image has already been processed
        if file.filename in processed_images:
            print(f"⚠️ Image {file.filename} has already been processed, skipping...")
            return {
                "message": f"Image {file.filename} has already been processed.",
                "filename": file.filename,
                "status": "skipped",
                "violations_detected": 0
            }

        # Determine file type
        file_type = "video" if file.content_type.startswith('video/') else "image"
        print(f"🎬 Processing uploaded {file_type}: {file.filename}")

        # Process media using appropriate processor
        import subprocess
        import json
        
        # Run media processing
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        if file_type == "video":
            # Use video processor
            result = subprocess.run([
                sys.executable, "improved_plate_processor.py", 
                file_path, "--output", output_dir
            ], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)  # 10 minute timeout for videos
        else:
            # Use image processor
            result = subprocess.run([
                sys.executable, "improved_image_processor.py",
                file_path
            ], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=200)  # 2 minute timeout
        
        print(f"📊 Processing result: returncode={result.returncode}")
        # Safely print output, handling encoding issues
        try:
            stdout_text = result.stdout[-500:] if result.stdout else ""
            stderr_text = result.stderr if result.stderr else ""
            print(f"📝 STDOUT: {stdout_text}")
            if stderr_text:
                print(f"📝 STDERR: {stderr_text}")
        except Exception as e:
            print(f"⚠️ Could not print subprocess output: {e}")
        
        if result.returncode == 0:
            print(f"✅ {file_type.capitalize()} processed successfully")
            
            # Load the processed violations - only get NEW violations from this processing run
            violations_file = os.path.join(output_dir, "violations_log.json")
            # Also check the output directory for image processing
            if not os.path.exists(violations_file):
                violations_file = "output/violations_log.json"
            
            if os.path.exists(violations_file):
                formatted_violations = []
                
                # ✅ Check if vehicle_summary.json exists (for VIDEO processing with multiple tracks)
                vehicle_summary_file = os.path.join(output_dir, "vehicle_summary.json")
                if not os.path.exists(vehicle_summary_file):
                    vehicle_summary_file = "output/vehicle_summary.json"
                
                # CASE 1: VIDEO with vehicle_summary.json - consolidate tracks by plate
                if file_type == "video" and os.path.exists(vehicle_summary_file):
                    print("📊 VIDEO processing: Using vehicle_summary.json to consolidate tracks")
                    try:
                        with open(vehicle_summary_file, 'r', encoding='utf-8') as f:
                            vehicle_summary = json.load(f)
                        
                        print(f"📊 Loaded vehicle summary with {len(vehicle_summary.get('vehicles', []))} vehicles")
                        
                        # ✅ FIX: Load violations_log.json to get accurate speeds
                        violations_speed_map = {}
                        max_speed_from_summary = vehicle_summary.get('max_speed_detected', 0.0)
                        
                        try:
                            with open(violations_file, 'r', encoding='utf-8') as f:
                                violations_data = json.load(f)
                            
                            # Handle both list format and dict format
                            if isinstance(violations_data, list):
                                all_violations = violations_data
                            elif isinstance(violations_data, dict) and 'violations' in violations_data:
                                all_violations = violations_data.get('violations', [])
                            else:
                                all_violations = []
                            
                            # Build a map of plate -> max speed from violations
                            for violation in all_violations:
                                plate = violation.get('plate_number')
                                speed_str = violation.get('speed', '0')
                                v_type = violation.get('type', '')
                                
                                if plate and speed_str != 'N/A':
                                    try:
                                        # Extract numeric speed value
                                        speed_val = float(''.join(c for c in str(speed_str).split()[0] if c.isdigit() or c == '.'))
                                        if plate not in violations_speed_map or speed_val > violations_speed_map[plate]:
                                            violations_speed_map[plate] = speed_val
                                    except (ValueError, AttributeError, IndexError):
                                        pass
                                
                                # ✅ If this is an overspeed violation but speed is low/missing, check if vehicle has overspeed in violations list
                                # This handles cases where final overspeed check found higher speed than logged violations
                                elif plate and v_type == 'overspeed' and plate not in violations_speed_map:
                                    # Use max_speed_detected as fallback if it's above speed limit
                                    if max_speed_from_summary > 40:  # Speed limit
                                        violations_speed_map[plate] = max_speed_from_summary
                                        print(f"⚠️ Using max_speed_detected ({max_speed_from_summary:.1f} km/h) for overspeed violation of {plate}")
                            
                            print(f"📊 Loaded speed data for {len(violations_speed_map)} plates from violations_log.json")
                        except Exception as e:
                            print(f"⚠️ Error loading violations_log.json for speed data: {e}")
                        
                        # Group vehicles by plate number and combine their data
                        plate_data = {}
                        for vehicle in vehicle_summary.get('vehicles', []):
                            plate = vehicle.get('plate_number')
                            
                            # Skip vehicles without valid plates
                            if not plate or plate == 'N/A' or str(plate).startswith('TRACK_'):
                                continue
                            
                            if plate not in plate_data:
                                plate_data[plate] = {
                                    'violations': set(),
                                    'max_speed': 0.0,
                                    'first_seen': vehicle.get('first_seen'),
                                    'last_seen': vehicle.get('last_seen'),
                                    'vehicle_type': 'Two-Wheeler'
                                }
                            
                            # Combine violations from all tracks
                            for violation in vehicle.get('violations', []):
                                # ✅ FIX: Map legacy class names
                                if violation == 'phone_usage':
                                    violation = 'mobile_usage'
                                if violation == 'traffic_violation':
                                    violation = 'triple_riding'
                                plate_data[plate]['violations'].add(violation)
                            
                            # Track maximum speed across all tracks
                            vehicle_max_speed = vehicle.get('max_speed', 0.0)
                            if vehicle_max_speed and vehicle_max_speed > plate_data[plate]['max_speed']:
                                plate_data[plate]['max_speed'] = vehicle_max_speed
                            
                            # Update timestamps
                            if vehicle.get('last_seen'):
                                plate_data[plate]['last_seen'] = vehicle.get('last_seen')
                        
                        # ✅ FIX: Override vehicle max_speed with actual speed from violations_log.json
                        for plate in plate_data:
                            if plate in violations_speed_map:
                                actual_speed = violations_speed_map[plate]
                                print(f"🔧 Overriding speed for {plate}: {plate_data[plate]['max_speed']:.1f} -> {actual_speed:.1f} km/h")
                                plate_data[plate]['max_speed'] = actual_speed
                        
                        print(f"📊 Combined data for {len(plate_data)} unique plates")
                        
                        # Insert one record per plate per violation type
                        for plate, data in plate_data.items():
                            violations_list = list(data['violations'])
                            max_speed = data['max_speed']
                            
                            # Check if overspeed should be added
                            speed_limit = 40.0
                            if max_speed > speed_limit and 'overspeed' not in violations_list:
                                violations_list.append('overspeed')
                            
                            # Insert each violation type separately
                            for violation_type in violations_list:
                                formatted_violation = {
                                    "id": len(formatted_violations) + 1,
                                    "type": violation_type,
                                    "plate_number": plate,
                                    "vehicle_type": data.get('vehicle_type', 'Two-Wheeler'),
                                    "speed": max_speed if violation_type == 'overspeed' else 0.0,
                                    "location": 'Traffic Camera',
                                    "camera_id": 'CAM001',
                                    "timestamp": data.get('last_seen', datetime.now().isoformat()),
                                    "severity": 'high' if max_speed > 50 else 'medium',
                                    "description": f"{violation_type.replace('_', ' ').title()} detected" + (f" - {max_speed:.1f} km/h in {speed_limit} km/h zone" if violation_type == 'overspeed' else ''),
                                    "source_image": file.filename
                                }
                                formatted_violations.append(formatted_violation)
                                
                                # Insert into database
                                if USE_REAL_DB and db_manager:
                                    db_violation_data = {
                                        'violation_type': violation_type,
                                        'number_plate': plate,
                                        'severity': formatted_violation['severity'],
                                        'description': formatted_violation['description'],
                                        'vehicle_type': formatted_violation['vehicle_type'],
                                        'vehicle_color': 'Unknown',
                                        'speed_limit': speed_limit,
                                        'actual_speed': max_speed if violation_type == 'overspeed' else 0.0,
                                        'speed_unit': 'km/h',
                                        'location': formatted_violation['location'],
                                        'camera_id': formatted_violation['camera_id'],
                                        'image_path': '',
                                        'confidence_score': 0.85
                                    }
                                    
                                    print(f"🔍 Inserting violation: {violation_type} for {plate} (speed: {max_speed:.1f} km/h)")
                                    try:
                                        db_manager.add_violation(db_violation_data)
                                        print(f"✅ Inserted violation: {violation_type} - {plate}")
                                    except Exception as e:
                                        print(f"❌ Failed to insert violation: {e}")
                                        import traceback
                                        traceback.print_exc()
                        
                        print(f"✅ Processed {len(formatted_violations)} violations for {len(plate_data)} unique plates")
                    
                    except Exception as e:
                        print(f"❌ Error processing vehicle summary: {e}")
                        import traceback
                        traceback.print_exc()
                
                # CASE 2: IMAGE or VIDEO without vehicle_summary.json - use violations_log.json directly
                else:
                    print("📊 IMAGE/Simple processing: Using violations_log.json directly")
                    try:
                        with open(violations_file, 'r', encoding='utf-8') as f:
                            violations_data = json.load(f)
                        
                        # Handle both list format and dict format
                        if isinstance(violations_data, list):
                            all_violations = violations_data
                        elif isinstance(violations_data, dict) and 'violations' in violations_data:
                            all_violations = violations_data.get('violations', [])
                        else:
                            print(f"⚠️ Unexpected violations format: {type(violations_data)}")
                            all_violations = []
                        
                        print(f"📊 Loaded {len(all_violations)} violations from violations_log.json")
                        
                        # Process each violation
                        for violation in all_violations:
                            if not isinstance(violation, dict):
                                continue
                            
                            plate_number = violation.get('plate_number')
                            
                            # Skip violations without valid plate numbers
                            if not plate_number or plate_number == 'N/A' or str(plate_number).startswith('TRACK_'):
                                print(f"⚠️ Skipping violation without valid plate: {violation.get('type', 'unknown')}")
                                continue
                            
                            violation_type = violation.get('type', 'unknown')
                            # ✅ FIX: Map legacy class names
                            if violation_type == 'phone_usage':
                                violation_type = 'mobile_usage'
                            if violation_type == 'traffic_violation':
                                violation_type = 'triple_riding'
                            
                            # Create formatted violation
                            formatted_violation = {
                                "id": len(formatted_violations) + 1,
                                "type": violation_type,
                                "plate_number": plate_number,
                                "vehicle_type": violation.get('vehicle_type', 'Two-Wheeler'),
                                "speed": violation.get('speed', 'N/A'),
                                "location": violation.get('location', 'Traffic Camera'),
                                "camera_id": violation.get('camera_id', 'image_processor'),
                                "timestamp": violation.get('timestamp', datetime.now().isoformat()),
                                "severity": violation.get('severity', 'medium'),
                                "description": violation.get('description', f'{violation_type} violation detected'),
                                "source_image": file.filename
                            }
                            formatted_violations.append(formatted_violation)
                            
                            # Insert into database
                            if USE_REAL_DB and db_manager:
                                # Handle speed values
                                actual_speed = violation.get('speed', 0.0)
                                if isinstance(actual_speed, str):
                                    try:
                                        if actual_speed != 'N/A':
                                            actual_speed = float(''.join(c for c in actual_speed.split()[0] if c.isdigit() or c == '.'))
                                        else:
                                            actual_speed = 0.0
                                    except (ValueError, AttributeError, IndexError):
                                        actual_speed = 0.0
                                
                                speed_limit = violation.get('speed_limit', 50.0)
                                if isinstance(speed_limit, str):
                                    try:
                                        speed_limit = float(''.join(c for c in speed_limit.split()[0] if c.isdigit() or c == '.'))
                                    except (ValueError, AttributeError, IndexError):
                                        speed_limit = 50.0
                                
                                db_violation_data = {
                                    'violation_type': violation_type,
                                    'number_plate': plate_number,
                                    'severity': violation.get('severity', 'medium'),
                                    'description': violation.get('description', f"{violation_type} violation detected"),
                                    'vehicle_type': violation.get('vehicle_type', 'Two-Wheeler'),
                                    'vehicle_color': violation.get('vehicle_color', 'Unknown'),
                                    'speed_limit': speed_limit,
                                    'actual_speed': actual_speed,
                                    'speed_unit': violation.get('speed_unit', 'km/h'),
                                    'location': violation.get('location', 'Traffic Camera'),
                                    'camera_id': violation.get('camera_id', 'image_processor'),
                                    'image_path': violation.get('image_path', ''),
                                    'confidence_score': violation.get('confidence', 0.8)
                                }
                                
                                print(f"🔍 Inserting violation: {violation_type} for {plate_number}")
                                try:
                                    db_manager.add_violation(db_violation_data)
                                    print(f"✅ Inserted violation: {violation_type} - {plate_number}")
                                except Exception as e:
                                    print(f"❌ Failed to insert violation: {e}")
                                    import traceback
                                    traceback.print_exc()
                        
                        print(f"✅ Processed {len(formatted_violations)} violations from violations_log.json")
                    
                    except Exception as e:
                        print(f"❌ Error processing violations_log.json: {e}")
                        import traceback
                        traceback.print_exc()

                # Update global violations
                real_violations.extend(formatted_violations)

                print(f"✅ Updated global violations: {len(formatted_violations)} new violations")

                # Mark this image as processed
                processed_images.add(file.filename)
                with open(processed_images_file, 'w', encoding='utf-8') as f:
                    for img in processed_images:
                        f.write(f"{img}\n")
                print(f"✅ Marked {file.filename} as processed")

                if len(formatted_violations) > 0:
                    return {
                        "message": f"{file_type.capitalize()} processed successfully! Found {len(formatted_violations)} violations.",
                        "filename": file.filename,
                        "file_type": file_type,
                        "status": "completed",
                        "violations_detected": len(formatted_violations)
                    }
                else:
                    return {
                        "message": f"{file_type.capitalize()} processed successfully! No violations detected.",
                        "filename": file.filename,
                        "file_type": file_type,
                        "status": "completed",
                        "violations_detected": len(formatted_violations)
                    }
            else:
                print("⚠️ No violations file found")
                return {
                    "message": f"{file_type.capitalize()} processed successfully! No violations detected.",
                    "filename": file.filename,
                    "file_type": file_type,
                    "status": "completed",
                    "violations_detected": 0
                }
        else:
            print(f"❌ {file_type.capitalize()} processing failed: {result.stderr}")
            return {
                "message": f"{file_type.capitalize()} processing failed",
                "filename": file.filename,
                "file_type": file_type,
                "status": "error",
                "error": result.stderr
            }
        
    except Exception as e:
        print(f"❌ Error processing media: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing media: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple Traffic Violation Detection Server...")
    print("📊 Dashboard: http://127.0.0.1:8081/dashboard")
    print("🔧 API Docs: http://127.0.0.1:8081/docs")
    uvicorn.run(app, host="127.0.0.1", port=8081)