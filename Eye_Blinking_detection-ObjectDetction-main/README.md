file 1- Action Detection Refined.ipynb

Function 1: mediapipe_detection
Purpose: Perform MediaPipe holistic detection on an image or video frame.

Arguments:

image: The input image or video frame.
model: The MediaPipe holistic model.
Returns:

image: The image or video frame with landmarks drawn on it.
results: A MediaPipe HolisticResults object containing the detected landmarks.
Detailed Description:

This function takes an image or video frame and performs MediaPipe holistic detection on it. It first converts the image from BGR to RGB format, makes it writable, processes the image using the MediaPipe holistic model, makes the image writable again, and converts it back to BGR format. The function returns the image with landmarks drawn on it and a MediaPipe HolisticResults object containing the detected landmarks.

Function 2: draw_landmarks
Purpose: Draw landmarks on an image or video frame.

Arguments:

image: The input image or video frame.
results: A MediaPipe HolisticResults object containing the detected landmarks.
Returns:

image: The image or video frame with landmarks drawn on it.
Detailed Description:

This function takes an image or video frame and a MediaPipe HolisticResults object and draws the detected landmarks on the image or video frame. It draws face connections, pose connections, left hand connections, and right hand connections. The function returns the image or video frame with landmarks drawn on it.

Function 3: draw_styled_landmarks
Purpose: Draw stylized landmarks on an image or video frame.

Arguments:

image: The input image or video frame.
results: A MediaPipe HolisticResults object containing the detected landmarks.
Returns:

image: The image or video frame with stylized landmarks drawn on it.
Detailed Description:

This function takes an image or video frame and a MediaPipe HolisticResults object and draws stylized landmarks on the image or video frame. It draws face connections, pose connections, left hand connections, and right hand connections with different colors and thicknesses. The function returns the image or video frame with stylized landmarks drawn on it.

Function 4: extract_keypoints
Purpose: Extract keypoints from a MediaPipe HolisticResults object.

Arguments:

results: A MediaPipe HolisticResults object containing the detected landmarks.
Returns:

keypoints: A NumPy array containing the extracted keypoints.
Detailed Description:

This function takes a MediaPipe HolisticResults object and extracts the keypoints from it. It extracts the pose keypoints, face keypoints, left hand keypoints, and right hand keypoints and concatenates them into a single NumPy array. The function returns the NumPy array containing the extracted keypoints.

Function 5: collect_data
Purpose: Collect data for training a machine learning model.

Arguments:

actions: A list of actions to collect data for.
no_sequences: The number of sequences to collect for each action.
sequence_length: The length of each sequence.
Returns:

None
Detailed Description:

This function collects data for training a machine learning model. It loops through the actions, sequences, and video frames and performs the following steps for each frame:

Reads the frame from the webcam.
Performs MediaPipe holistic detection on the frame.
Draws landmarks on the frame.
Extracts keypoints from the MediaPipe results.
Saves the keypoints to a NumPy file.
The function breaks the loop if the 'q' key is pressed.

Function 6: preprocess_data
Purpose: Preprocess the collected data for training a machine learning model.

Arguments:

data: The collected data.
Returns:

X: A NumPy array containing the preprocessed data.
y: A NumPy array containing the labels.
Detailed Description:

This function preprocesses the collected data for training a machine learning model. It performs the following steps:

Loads the data from the NumPy files.
Preprocesses the data (e.g., normalizing, scaling).
Splits the data into training and testing sets.
The function returns the preprocessed data and the labels.

Function 7: train_model
Purpose: Train a machine learning model for action recognition.

Arguments:

X: The preprocessed data.
y: The labels.
Returns:

model: The trained machine learning model.
Detailed Description:

This function trains a machine learning model for action recognition. It performs the following steps:

Creates a sequential model.
Adds LSTM layers to the model.
Adds dense layers to the model.
Compiles the model.
Trains the model on the preprocessed data.
The function returns the trained machine learning model.

Function 8: evaluate_model
Purpose: Evaluate the performance of a machine learning model for action recognition.

Arguments:

model: The trained machine learning model.
X_test: The test data.
y_test: The test labels.
Returns:

loss: The loss of the model on the test data.
accuracy: The accuracy of the model on the test data.
Detailed Description:

This function evaluates the performance of a machine learning model for action recognition. It performs the following steps:

Evaluates the model on the test data.
Calculates the loss and accuracy of the model.
The function returns the loss and accuracy of the model on the test data.

Function 9: save_model
Purpose: Save a machine learning model to a file.

Arguments:

model: The machine learning model to save.
filename: The filename to save the model to.
Returns:

None
Detailed Description:

This function saves a machine learning model to a file. It performs the following steps:

Serializes the model.
Saves the serialized model to a file.
The function does not return anything.

Function 10: load_model
Purpose: Load a machine learning model from a file.

Arguments:

filename: The filename of the model to load.
Returns:

model: The loaded machine learning model.
Detailed Description:

This function loads a machine learning model from a file. It performs the following steps:

Loads the serialized model from the file.
Deserializes the model.
The function returns the loaded machine learning model.

Function 11: predict
Purpose: Predict the action from a sequence of keypoints.

Arguments:

model: The trained machine learning model.
sequence: A sequence of keypoints.
Returns:

prediction: The predicted action.
Detailed Description:

This function predicts the action from a sequence of keypoints. It performs the following steps:

Reshapes the sequence of keypoints to fit the model's input shape.
Predicts the action using the model.
The function returns the predicted action.



file 2- Action Detection Tutorial.ipynb

Function 1: mediapipe_detection
Purpose: Perform MediaPipe holistic detection on an image or video frame.

Arguments:

image: The input image or video frame.
model: The MediaPipe holistic model.
Returns:

image: The image or video frame with landmarks drawn on it.
results: A MediaPipe HolisticResults object containing the detected landmarks.
Detailed Description:

This function takes an image or video frame and performs MediaPipe holistic detection on it. It first converts the image from BGR to RGB format, makes it writable, processes the image using the MediaPipe holistic model, makes the image writable again, and converts it back to BGR format. The function returns the image with landmarks drawn on it and a MediaPipe HolisticResults object containing the detected landmarks.

Function 2: draw_landmarks
Purpose: Draw landmarks on an image or video frame.

Arguments:

image: The input image or video frame.
results: A MediaPipe HolisticResults object containing the detected landmarks.
Returns:

image: The image or video frame with landmarks drawn on it.
Detailed Description:

This function takes an image or video frame and a MediaPipe HolisticResults object and draws the detected landmarks on the image or video frame. It draws face connections, pose connections, left hand connections, and right hand connections. The function returns the image or video frame with landmarks drawn on it.

Function 3: draw_styled_landmarks
Purpose: Draw stylized landmarks on an image or video frame.

Arguments:

image: The input image or video frame.
results: A MediaPipe HolisticResults object containing the detected landmarks.
Returns:

image: The image or video frame with stylized landmarks drawn on it.
Detailed Description:

This function takes an image or video frame and a MediaPipe HolisticResults object and draws stylized landmarks on the image or video frame. It draws face connections, pose connections, left hand connections, and right hand connections with different colors and thicknesses. The function returns the image or video frame with stylized landmarks drawn on it.

Function 4: extract_keypoints
Purpose: Extract keypoints from a MediaPipe HolisticResults object.

Arguments:

results: A MediaPipe HolisticResults object containing the detected landmarks.
Returns:

keypoints: A NumPy array containing the extracted keypoints.
Detailed Description:

This function takes a MediaPipe HolisticResults object and extracts the keypoints from it. It extracts the pose keypoints, face keypoints, left hand keypoints, and right hand keypoints and concatenates them into a single NumPy array. The function returns the NumPy array containing the extracted keypoints.

Function 5: collect_data
Purpose: Collect data for training a machine learning model.

Arguments:

actions: A list of actions to collect data for.
no_sequences: The number of sequences to collect for each action.
sequence_length: The length of each sequence.
Returns:

None
Detailed Description:

This function collects data for training a machine learning model. It loops through the actions, sequences, and video frames and performs the following steps for each frame:

Reads the frame from the webcam.
Performs MediaPipe holistic detection on the frame.
Draws landmarks on the frame.
Extracts keypoints from the MediaPipe results.
Saves the keypoints to a NumPy file.
The function breaks the loop if the 'q' key is pressed.

Function 6: preprocess_data
Purpose: Preprocess the collected data for training a machine learning model.

Arguments:

data: The collected data.
Returns:

X: A NumPy array containing the preprocessed data.
y: A NumPy array containing the labels.
Detailed Description:

This function preprocesses the collected data for training a machine learning model. It performs the following steps:

Loads the data from the NumPy files.
Preprocesses the data (e.g., normalizing, scaling).
Splits the data into training and testing sets.
The function returns the preprocessed data and the labels.

Function 7: train_model
Purpose: Train a machine learning model for action recognition.

Arguments:

X: The preprocessed data.
y: The labels.
Returns:

model: The trained machine learning model.
Detailed Description:

This function trains a machine learning model for action recognition. It performs the following steps:

Creates a sequential model.
Adds LSTM layers to the model.
Adds dense layers to the model.
Compiles the model.
Trains the model on the preprocessed data.
The function returns the trained machine learning model.

Function 8: evaluate_model
Purpose: Evaluate the performance of a machine learning model for action recognition.

Arguments:

model: The trained machine learning model.
X_test: The test data.
y_test: The test labels.
Returns:

loss: The loss of the model on the test data.
accuracy: The accuracy of the model on the test data.
Detailed Description:

This function evaluates the performance of a machine learning model for action recognition. It performs the following steps:

Evaluates the model on the test data.
Calculates the loss and accuracy of the model.
The function returns the loss and accuracy of the model on the test data.

Function 9: save_model
Purpose: Save a machine learning model to a file.

Arguments:

model: The machine learning model to save.
filename: The filename to save the model to.
Returns:

None
Detailed Description:

This function saves a machine learning model to a file. It performs the following steps:

Serializes the model.
Saves the serialized model to a file.
The function does not return anything.

Function 10: load_model
Purpose: Load a machine learning model from a file.

Arguments:

filename: The filename of the model to load.
Returns:

model: The loaded machine learning model.
Detailed Description:

This function loads a machine learning model from a file. It performs the following steps:

Loads the serialized model from the file.
Deserializes the model.
The function returns the loaded machine learning model.

Function 11: predict
Purpose: Predict the action from a sequence of keypoints.

Arguments:

model: The trained machine learning model.
sequence: A sequence of keypoints.
Returns:

prediction: The predicted action.
Detailed Description:

This function predicts the action from a sequence of keypoints. It performs the following steps:

Reshapes the sequence of keypoints to fit the model's input shape.
Predicts the action using the model.
The function returns the predicted action.