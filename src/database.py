from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/traffic_violations')

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Violation(Base):
    __tablename__ = "violations"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Violation details
    violation_type = Column(String(50), nullable=False)  # e.g., Over-speed, No Helmet
    severity = Column(String(20), nullable=True)  # low, medium, high, critical
    description = Column(Text, nullable=True)  # extra notes about the violation
    
    # Vehicle details
    number_plate = Column(String(20), nullable=False)
    vehicle_type = Column(String(20), nullable=True)  # car, bike, truck, etc.
    vehicle_color = Column(String(20), nullable=True)
    
    # Speed details (if applicable)
    speed_limit = Column(Float, nullable=True)
    actual_speed = Column(Float, nullable=True)
    speed_unit = Column(String(10), default='km/h')
    
    # Location and camera
    location = Column(String(100), nullable=True)
    camera_id = Column(String(20), nullable=True)
    
    # Evidence
    image_path = Column(Text, nullable=True)
    
    # Metadata
    confidence_score = Column(Float, nullable=True)  # model's prediction confidence
    timestamp = Column(DateTime, default=func.now())



def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def init_database():
    """Initialize database with sample data"""
    db = SessionLocal()
    try:
        # Create tables
        create_tables()
        print("✓ Database initialized successfully")
        
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        db.rollback()
    finally:
        db.close()

class DatabaseManager:
    def __init__(self):
        self.db = SessionLocal()
    
    def add_violation(self, violation_data: dict):
        """Add a new violation to database"""
        from datetime import datetime, timedelta

        try:
            # Check for duplicate violations within the last 5 minutes for the same plate and violation type
            five_minutes_ago = datetime.now() - timedelta(minutes=5)
            existing = self.db.query(Violation).filter(
                Violation.number_plate == violation_data['number_plate'],
                Violation.violation_type == violation_data['violation_type'],
                Violation.timestamp >= five_minutes_ago
            ).first()

            if existing:
                print(f"⚠️ Skipping duplicate violation: {violation_data['violation_type']} for plate {violation_data['number_plate']} (already exists within 5 minutes)")
                return existing

            violation = Violation(**violation_data)
            self.db.add(violation)
            self.db.commit()
            self.db.refresh(violation)
            return violation
        except Exception as e:
            self.db.rollback()
            print(f"Error adding violation: {e}")
            return None
    
    def get_violations(self, limit: int = 100, offset: int = 0, violation_type: str = None):
        """Get violations with optional filtering"""
        query = self.db.query(Violation)
        
        if violation_type:
            query = query.filter(Violation.violation_type == violation_type)
        
        return query.order_by(Violation.timestamp.desc()).offset(offset).limit(limit).all()
    
    def get_violation_stats(self):
        """Get violation statistics"""
        from datetime import datetime, timedelta

        # Total violations
        total = self.db.query(Violation).count()

        # Violations in last 24 hours
        yesterday = datetime.now() - timedelta(days=1)
        recent_24h = self.db.query(Violation).filter(Violation.timestamp >= yesterday).count()

        # Count by violation type (map database types to dashboard types)
        violation_types = {}
        type_mapping = {
            'no_helmet': 'helmet',
            'mobile_usage': 'mobile_usage',
            'phone_usage': 'mobile_usage',
            'overspeed': 'speed',
            'triple_riding': 'triple_riding',
            'traffic_violation': 'triple_riding'
        }

        for db_type, dashboard_type in type_mapping.items():
            count = self.db.query(Violation).filter(Violation.violation_type == db_type).count()
            violation_types[dashboard_type] = violation_types.get(dashboard_type, 0) + count

        return {
            'total_violations': total,
            'recent_violations_24h': recent_24h,
            'violation_types': violation_types
        }
    
    def clean_duplicates(self, number_plate: str, violation_type: str):
        """
        Remove duplicate violations for a given number plate and violation type,
        keeping only the most recent violation (by max id).
        """
        try:
            # Find duplicates count
            duplicates = self.db.execute(text('''
                SELECT number_plate, violation_type, COUNT(*) as count
                FROM violations
                WHERE number_plate = :number_plate AND violation_type = :violation_type
                GROUP BY number_plate, violation_type
                HAVING COUNT(*) > 1
            '''), {'number_plate': number_plate, 'violation_type': violation_type}).fetchall()

            if duplicates:
                print(f"Found duplicates for plate {number_plate} and violation {violation_type}, cleaning up...")

                # Delete duplicates keeping only the most recent (max id)
                self.db.execute(text('''
                    DELETE FROM violations
                    WHERE id NOT IN (
                        SELECT MAX(id)
                        FROM violations
                        WHERE number_plate = :number_plate AND violation_type = :violation_type
                    )
                    AND number_plate = :number_plate AND violation_type = :violation_type
                '''), {'number_plate': number_plate, 'violation_type': violation_type})

                self.db.commit()
                print("Duplicate cleanup completed.")
            else:
                print("No duplicates found.")
        except Exception as e:
            self.db.rollback()
            print(f"Error cleaning duplicates: {e}")

    
    def close(self):
        """Close database connection"""
        self.db.close()