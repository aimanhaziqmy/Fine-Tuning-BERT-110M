from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding


dataset_dict = load_dataset("shawhin/phishing-site-classification")

print(dataset_dict)

# define pre-trained model path
model_path = "google-bert/bert-base-uncased"

#load model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load model with binary classification head
id2label = {0: "Safe", 1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe": 1}
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, id2label=id2label, label2id=label2id,) 

### IF WE RUN THIS IN COMPUTER IT IS COMPUTATIONALLY EXPENSIVE, SO WE FREEZE ALL BASE MODEL PARAMETERS

# freeze all base model parameters
for name, param in model.base_model.named_parameters():
    param.requires_grad = False

# --> transfer learning we leave all base parameter frozen and trained only on classification head

# unfreeze base model pooling layers
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True

# define text preprocessing
def preprocess_function(examples):
    # return tokenized text with truncation
    return tokenizer(examples["text"], truncation=True, padding=True)

# preprocess all datasets
tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load metrics
accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    # get predictions
    predictions, labels = eval_pred #tuple

    # apply softmax to get probabilities
    probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)

    # use probability of the positive class for ROC AUC
    positive_class_probs = probabilities[:, 1]

    # compute auc
    auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)["roc_auc"], 3)

    # predict most probable class
    predicted_classes = np.argmax(predictions, axis=1)

    # compute accuracy
    acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)["accuracy"], 3)

    return {"accuracy": acc, "auc": auc}

# hyperparameters
lr = 2e-4
batch_size = 8
num_epochs = 10

training_args = TrainingArguments(
    output_dir="bert-phishing-classifier_teacher",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# validation data
# apply model to validation data
predictions = trainer.predict(tokenized_datasets["validation"])

# Extract the logits and labels from the predictions object
logits = predictions.predictions
labels = predictions.label_ids

# Use your compute_metrics function
metrics = compute_metrics((logits, labels))
print(metrics)

