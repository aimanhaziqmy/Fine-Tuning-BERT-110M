from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# model path checkpoint
model_path = "bert-phishing-classifier_teacher/checkpoint-2630"

# load the model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# load the  tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

text = "https://iaygf9q91.test.demo.aimanhaziq.my"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# set the model to evaluation mode
model.eval()

# Disable gradient calculation
with torch.no_grad():
    # make predictions
    outputs = model(**inputs)

predicted_class = torch.argmax(outputs.logits, dim=1).item()
print(f"Predicted class: {predicted_class}")

# If you want to see the probabilities for each class
probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
print(f"Class probabilities: {probabilities}")
