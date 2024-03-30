import signal
from datetime import datetime
import threading
import numpy as np
from pythonosc import dispatcher, osc_server
from FE import FE  # Assuming FE is your feature extraction class
from prediction import predic  # Assuming prediction is your prediction function
import mysql.connector  # For database connection

class EEGProcessor:
    def __init__(self, featureObj, predic, batch_size=30, buffer_size=100):
        self.buffer = []
        self.featureObj = featureObj
        self.predic = predic
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.last_prediction = None  # Variable to store the last prediction

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
        data_batch = np.array(self.buffer[:self.batch_size])
        ret, feat_names = self.featureObj.generate_feature_vectors_from_samples(np.array(self.buffer), 150, 1., cols_to_ignore=-1)
        ret_2d = ret.reshape(1, -1)
        prediction = self.predic.predctionVal(ret_2d)

        print("prediction:", prediction)
        self.last_prediction = prediction

        del self.buffer[:self.batch_size]

    def insert_prediction_to_db(self, prediction):
        """
        Insert the prediction into the eegsession table
        """
        # Define your database connection details (replace with yours)
        username = "root"
        password = ""
        hostname = "localhost"
        database = "Sukoon"

        connection = mysql.connector.connect(
            user=username, password=password, host=hostname, database=database
        )
        cursor = connection.cursor()

        # Get current timestamp
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")

        sql = "INSERT INTO eegsession (SessionID, HeadbandID, Duration, Result, Timestamp, patient_ID, doctor_ID) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        # Fill data based on your logic (replace with appropriate values)
        data = (1, 1212, "hamza", prediction, timestamp, 1, 1212)  # Modify these values as needed
        cursor.execute(sql, data)

        connection.commit()
        cursor.close()
        connection.close()

def stop_server(signum, frame):
    """
    Signal handler for stopping the server
    """
    raise KeyboardInterrupt

if __name__ == "__main__":
    ip = "0.0.0.0"
    port = 5000

    # Initialize your featureObj and predic objects
    featureObj = FE()
    predic_obj = predic()

    eeg_processor = EEGProcessor(featureObj, predic_obj)

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", eeg_processor.on_new_eeg_data)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port " + str(port))

    # Set a timeout for 30 seconds
    signal.signal(signal.SIGALRM, stop_server)
    signal.alarm(30)

    try:
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        server_thread.join()
    except KeyboardInterrupt:
        pass
    except:
        pass
    finally:
        signal.alarm(0)  # Disable the alarm

        server.server_close()

        # Insert last prediction into the database
        if eeg_processor.last_prediction is not None:
            eeg_processor.insert_prediction_to_db(eeg_processor.last_prediction)

    print("Script finished running")