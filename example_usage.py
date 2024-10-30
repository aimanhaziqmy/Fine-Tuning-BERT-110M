import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# Load the ONNX model
ort_session = ort.InferenceSession("model.onnx")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-xxx/checkpoint-2630')

# Prepare input text
text = "zalora.com.my"
inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)

# Get the input names of the model
input_names = [input.name for input in ort_session.get_inputs()]

# Prepare the input dict
ort_inputs = {name: inputs[name] for name in input_names}

# Run inference
ort_outputs = ort_session.run(None, ort_inputs)

# Process the output (for classification tasks)
predicted_class = np.argmax(ort_outputs[0], axis=1)
print(f"Predicted class: {predicted_class[0]}")

# If you want to see the probabilities
probabilities = ort_outputs[0][0]
print(f"Class probabilities: {probabilities}")
