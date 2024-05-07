import random
import signal
from datetime import datetime
import threading
from typing import Counter
import uuid
import numpy as np
import requests
from FE import FE  # Assuming FE is your feature extraction class
from prediction import predic  # Assuming prediction is your prediction function

class EEGProcessor:
    def __init__(self, featureObj, predic, batch_size=30, buffer_size=100):
        self.buffer = []
        self.featureObj = featureObj
        self.predic = predic
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.last_prediction = None
        self.prediction_history = []
        

    def on_new_eeg_data(self, address: str, *args):
        """
        To handle EEG data emitted from the OSC server
        """
        dateTimeObj = datetime.now()
        printStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
        for arg in args:
            printStr += "," + str(arg)
        data = list(args)
        timestamp_unix = dateTimeObj.timestamp()
        data = [timestamp_unix] + data[0:5]
        self.buffer.append(data)

        if len(self.buffer) >= self.buffer_size:
            self.process_buffer()

    def process_buffer(self):
        """
        Process the data in the buffer
        """
        self.prediction_history = []
        data_batch = np.array(self.buffer[:self.batch_size])
        ret, feat_names = self.featureObj.generate_feature_vectors_from_samples(np.array(self.buffer), 150, 1., cols_to_ignore=-1)
        ret_2d = ret.reshape(1, -1)
        prediction = self.predic.predctionVal(ret_2d)

        print("prediction:", prediction)
        self.last_prediction = prediction
        print("Last Prediction:", self.last_prediction)
        if self.last_prediction is not None:
            self.prediction_history.append(self.last_prediction)

        self.buffer = []

    def insert_most_common_prediction_to_db(self):
        """
        Insert the most common prediction into the eegsession table
        """
        most_common_prediction = None
        if self.prediction_history:
            prediction_counter = Counter(self.prediction_history)
            most_common_prediction, _ = prediction_counter.most_common(1)[0]

        if most_common_prediction is not None:
            print("Calling insert_prediction_to_db with prediction:", most_common_prediction)
            self.insert_prediction_to_db(most_common_prediction)

    def insert_prediction_to_db(self, prediction,  patient_ID):
        """
        Insert the prediction into the eegsession table
        """

        print("Function Called")

        
        min_value = 100
        max_value = 999
        SessionID = random.randint(min_value, max_value)
        # Get current timestamp
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
        

        urlSession = "https://infinite-wave-71025-404d3d4feff8.herokuapp.com/api/addSession"
        data = {
            "HeadbandID": 105,
            "Duration": 30,
            "Result": prediction,
            "Timestamp": timestamp,
            "patient_ID": patient_ID,
            "doctor_ID": None
        }

        print("Request Body:", data)
        response = requests.post(urlSession, json=data)

        print("Status Code:", response.status_code)
        if response.status_code == 200:
            print("Data inserted successfully")
        else:
            print("Data Error:", response.text)
