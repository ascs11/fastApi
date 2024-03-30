from keras.models import load_model
from keras.models import model_from_json

# Load the model from H5 file
model = load_model('models/BiLSTM.h5')

# Convert the model to JSON format
model_json = model.to_json()

# Save the JSON model to a file
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# Later, you can load the JSON model back into a Keras model
with open('model.json', 'r') as json_file:
    json_saved_model = json_file.read()

model_from_json = model_from_json(json_saved_model)
