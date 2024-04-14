import streamlit as st
import tflite_runtime.interpreter as tflite
import csv
import numpy as np
import time
import matplotlib.pyplot as plt

# Load DECOMPRESS DATA LOADING
import time




# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="tflite_model_default.tflite")
interpreter.allocate_tensors()  # Needed before execution!

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Display the input details
st.write(f"Input Tensor Shape: {input_details[0]['shape']}")

# Arrhythmia Dataset information
st.write(
    """
    Arrhythmia Dataset
    Number of Samples: 109446
    Number of Categories: 5
    Sampling Frequency: 125Hz
    Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
    """
)

characters = ["N", "S", "V", "F", "Q"]

# Load the data
with open("mitbih_test.csv", "r") as file:
    reader = csv.reader(file)
    list_reader = list(reader)

    num_samples = 0
    diff_time = 0
    accurate_count = 0

    predicted_character_count = {'N': 0, 'S': 0, 'V': 0, 'F': 0, 'Q': 0}

    for row in list_reader:
        inp = np.array(row, dtype=np.float32)[:186]
        inp = np.expand_dims(inp, axis=0)
        inp = np.expand_dims(inp, axis=2)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]["index"], inp)

        # Perform inference
        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]["index"])
        results = np.squeeze(output_data)

        # Find the index of the maximum probability
        max_prob_index = np.argmax(results)

        # Get the predicted character
        predicted_character = characters[max_prob_index]

        # Calculate inference time
        num_samples += 1
        diff_time += stop_time - start_time

        # Calculate accuracy
        ground_truth_label = int(float(row[-1]))
        if max_prob_index == ground_truth_label:
            accurate_count += 1

        # Count the occurrence of each predicted character
        predicted_character_count[predicted_character] += 1

# Calculate accuracy
accuracy = (accurate_count / num_samples) * 100
def main():


    st.markdown(f"<h2>DECOMPRESSING</h2>", unsafe_allow_html=True)

    # Create a progress bar
    progress_bar = st.progress(0)
    

    # Start time
    start_time = time.time()

    # Update the progress bar
    while time.time() - start_time < 5:
        progress_bar.progress((time.time() - start_time)/5)

    # Finish loading
    progress_bar.empty()

if __name__=='__main__':
    main()

# Display accuracy and average inference time
st.write(f"Accuracy: {accuracy} %")
st.write("Average Inference Time: {:.3f} ms".format((diff_time / num_samples) * 1000))

# Display the count of predicted characters based on user selection
visualization_option = st.selectbox(
    "Select Visualization Type",
    ["Bar Chart", "Pie Chart", "Area Graph"]
)

if visualization_option == "Bar Chart":
    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(predicted_character_count.keys(), predicted_character_count.values())
    ax.set_xlabel("Characters")
    ax.set_ylabel("Count")
    ax.set_title("Count of Predicted Characters")
    for key, value in predicted_character_count.items():
        ax.text(key, value + 0.1, str(value), ha='center', va='bottom')
    st.pyplot(fig)
elif visualization_option == "Pie Chart":
    # Pie chart
    fig, ax = plt.subplots()
    ax.pie(predicted_character_count.values(), labels=predicted_character_count.keys(), autopct='%1.1f%%')
    ax.set_title("Count of Predicted Characters")
    st.pyplot(fig)
elif visualization_option == "Area Graph":
    # Area graph
    fig, ax = plt.subplots()
    ax.fill_between(predicted_character_count.keys(), predicted_character_count.values(), color="skyblue", alpha=0.4)
    ax.plot(predicted_character_count.keys(), predicted_character_count.values(), color="Slateblue", alpha=0.6)
    ax.set_xlabel("Characters")
    ax.set_ylabel("Count")
    ax.set_title("Count of Predicted Characters")
    for key, value in predicted_character_count.items():
        ax.text(key, value + 0.1, str(value), ha='center', va='bottom')
    st.pyplot(fig)