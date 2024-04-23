import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tkinter import Tk, Label, Button, Entry, StringVar, filedialog, messagebox

def preprocess_data(data):
    # Encode 'Message Type' column
    data['Message Type'] = LabelEncoder().fit_transform(data['Message Type'])

    # Split 'Message Content' column into a list of floats
    data['Message Content'] = data['Message Content'].apply(lambda x: [float(i) for i in str(x).split(',')])

    # Encode 'Priority' column
    data['Priority'] = LabelEncoder().fit_transform(data['Priority'])

    # Select columns for training
    columns_to_use = ['Message Type', 'Message Content', 'Priority', 'Source Vehicle', 'Destination Vehicle']
    X = data[columns_to_use].values

    # Normalize the Message Content feature
    scaler = MinMaxScaler()
    X[:, 1] = list(scaler.fit_transform(np.array(X[:, 1].tolist())))

    # Reshape the input for the fully connected neural network
    X = np.array([np.array([x[0], *x[1], x[2]]) for x in X])

    return X

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def on_open_file_click():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    data = pd.read_csv(file_path)
    X = preprocess_data(data)
    predictions = model.predict(X)

    output = ""
    for i, pred in enumerate(predictions):
        anomaly = "Anomaly" if pred[0] > 0.5 else "Normal"
        output += f"Record {i + 1}: {anomaly}\n"

    result_text.set(output)

def on_exit_click():
    window.destroy()

# Load the saved model
model_path = 'ids.h5'
model = load_model(model_path)

# Create the main window
window = Tk()
window.title("IDS Interface")
window.geometry("400x500")

# Add labels, entry, and buttons
welcome_label = Label(window, text="Welcome to the IDS Interface", font=("Arial", 14))
welcome_label.pack(pady=10)

instruction_label = Label(window, text="Upload your dataset (.csv) to analyze:", font=("Arial", 12))
instruction_label.pack(pady=10)

open_file_button = Button(window, text="Open File", command=on_open_file_click)
open_file_button.pack(pady=10)

result_text = StringVar()
result_label = Label(window, textvariable=result_text, font=("Arial", 12), justify="left")
result_label.pack(pady=10)

exit_button = Button(window, text="Exit", command=on_exit_click)
exit_button.pack(pady=10)

# Start the main loop
window.mainloop()