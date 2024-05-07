import asyncio
import threading
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from typing import Annotated
import uvicorn
from OSC_Receiver_Simple import EEGProcessor
import models as models
from sqlalchemy.orm import Session
from typing import List
from pythonosc import dispatcher, osc_server
from FE import FE  # Import your FE class
from prediction import predic  # Import your predic class
from threading import Thread
import random
from datetime import datetime
from collections import Counter
import numpy as np
import requests
from FE import FE  # Assuming FE is your feature extraction class
from prediction import predic  # Assuming prediction is your prediction function

app = FastAPI()

# Initialize your featureObj and predic objects
featureObj = FE()
predic_obj = predic()
stop_event = threading.Event()  # Define the stop_event

class StartProcessingRequest(BaseModel):
    patientID: int


async def run_eeg_processor(dispatcher, ip, port, eeg_processor, patient_id):
    
    dispatcher.map("/muse/eeg", eeg_processor.on_new_eeg_data)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port " + str(port))

    try:
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()

        await asyncio.wait_for(asyncio.sleep(20), timeout=20)

    except asyncio.CancelledError:
        pass
    finally:
        # Insert last prediction into the database
        if eeg_processor.last_prediction is not None:
            eeg_processor.insert_prediction_to_db(eeg_processor.last_prediction, patient_id)

        server.shutdown()
        server.server_close()

    print("Script finished running")





@app.post("/start_eeg_processing")
async def trigger_eeg_processor(data: StartProcessingRequest):
    patient_id = data.patientID
    print(patient_id)
    ip = "0.0.0.0"
    port = 5000

    eeg_processor = EEGProcessor(featureObj, predic_obj, patient_id)
    dispatcher_obj = dispatcher.Dispatcher()  
    asyncio.create_task(run_eeg_processor(dispatcher_obj, ip, port, eeg_processor, patient_id))

    return {"message": "EEG processor started"}


@app.post("/stop_eeg_processing")
async def stop_eeg_processing():
    global stop_event
    stop_event.set()
    return {"message": "Stopping EEG processor after 30 seconds"}




