-- Create the database
CREATE DATABASE traffic_violations;
\c traffic_violations;

-- Create the violations table with more details
CREATE TABLE violations (
    id SERIAL PRIMARY KEY,                          -- Unique ID for each violation
    violation_type VARCHAR(50) NOT NULL,            -- e.g., Over-speed, No Helmet
    severity VARCHAR(50) DEFAULT 'medium',          -- Severity of the violation
    number_plate VARCHAR(20) NOT NULL,              -- Vehicle registration number
    speed FLOAT CHECK (speed >= 0),                  -- Vehicle speed at violation
    speed_limit FLOAT CHECK (speed_limit >= 0),      -- Applicable speed limit at location
    location VARCHAR(100) NOT NULL,                 -- Violation location
    latitude DECIMAL(9,6),                          -- GPS latitude
    longitude DECIMAL(9,6),                         -- GPS longitude
    officer_name VARCHAR(50),                       -- Officer who recorded the violation
    fine_amount NUMERIC(10,2) CHECK (fine_amount >= 0), -- Fine amount for the violation
    status VARCHAR(20) DEFAULT 'Pending' CHECK (status IN ('Pending', 'Paid', 'Challenged')), -- Violation status
    image_path TEXT,                                 -- Path to stored image proof
    video_path TEXT,                                 -- Optional video proof path
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- When the violation occurred
    description TEXT                                       -- Additional notes
);
