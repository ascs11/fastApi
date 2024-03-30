import threading
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.routing import APIRoute
import numpy as np
from pydantic import BaseModel
from typing import Annotated

import uvicorn
from OSC_Receiver_Simple import EEGProcessor
import models as models
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from typing import List
from pythonosc import dispatcher, osc_server
from FE import FE  # Import your FE class
from prediction import predic  # Import your predic class
from threading import Thread


app = FastAPI()

# Create the database models and tables
models.Base.metadata.create_all(bind=engine)

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Depends(get_db)

# Define the request models
class HistoryItem(BaseModel):
    timestamp: str
    duration: str
    result: str

# Initialize your featureObj and predic objects
featureObj = FE()
predic_obj = predic()
eeg_processor = EEGProcessor(featureObj, predic_obj)

# Define the FastAPI endpoints
@app.get("/history", response_model=List[HistoryItem])
async def get_history(db: Session = db_dependency):
    history_items = db.query(models.History).all()
    return [{"timestamp": item.timestamp, "duration": item.duration, "result": item.result} for item in history_items]

@app.get("/history/{item_id}", response_model=HistoryItem)
async def get_history_item(item_id: int, db: Session = db_dependency):
    history_item = db.query(models.History).filter(models.History.id == item_id).first()
    if not history_item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History item not found")
    return {"timestamp": history_item.timestamp, "duration": history_item.duration, "result": history_item.result}

@app.get("/history/{item_id}/result", response_model=str)
async def get_history_result(item_id: int, db: Session = db_dependency):
    history_item = db.query(models.History).filter(models.History.id == item_id).first()
    if not history_item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History item not found")
    return history_item.result

eeg_processor = EEGProcessor(featureObj, predic_obj)

@app.post("/start_eeg_processing")
def start_eeg_processing():
    try:
        eeg_processor.start_processing()
        return {"message": "EEG processing started"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/stop_eeg_processing")
def stop_eeg_processing():
    try:
        eeg_processor.stop_server(None, None)
        return {"message": "EEG processing stopped"}
    except Exception as e:
        return {"error": str(e)}

# Call the function to start the EEG processing
start_eeg_processing()
