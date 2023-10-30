from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

# Define the model and tokenizer name
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a sentiment analysis pipeline using the model
classifier = pipeline("sentiment-analysis", model=model_name)

# Perform sentiment analysis on two example sentences
results = classifier(["We are happy to show the transformers Library.", "We hope you don't hate it."])

# Print the sentiment analysis results
for result in results:
    print(result)

# Tokenize and convert text to token IDs
text = "We are happy to show the transformers Library."
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# You can also use the tokenizer function directly
input_ids = tokenizer(text)

# Print tokenization results
print(f"    Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
print(f"Token IDs: {input_ids}")

# Define a list of training texts
X_train = ["We are happy to show the transformers Library.", "We hope you don't hate it."]

# Tokenize and prepare the batch for model input
batch = tokenizer(
    X_train,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Perform inference with the model
with torch.no_grad():
    outputs = model(**batch)  # For PyTorch, unpack the batch data
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)

    # Convert label IDs to human-readable labels
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)
