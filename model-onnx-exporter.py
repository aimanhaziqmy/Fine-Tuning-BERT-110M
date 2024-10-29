from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import onnx
import onnxruntime

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('bert-phishing-classifier_teacher/checkpoint-2630')
tokenizer = AutoTokenizer.from_pretrained('bert-phishing-classifier_teacher/checkpoint-2630')

# Create a dummy input
dummy_text = "This is a dummy input for ONNX export"
dummy_inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# ONNX export
torch.onnx.export(
    model, 
    (dummy_inputs.input_ids, dummy_inputs.attention_mask),  # model inputs
    "model.onnx",  # where to save the model
    input_names=['input_ids', 'attention_mask'],  # the model's input names
    output_names=['output'],  # the model's output names
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},  # variable length axes
                  'attention_mask': {0: 'batch_size', 1: 'sequence'},
                  'output': {0: 'batch_size', 1: 'sequence'}},
    opset_version=14  # Increase the opset version to 14
)

print("Model exported to ONNX format successfully.")
