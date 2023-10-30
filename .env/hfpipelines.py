# Import necessary libraries
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Define the pre-trained model's name
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the pre-trained model and its corresponding tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sentiment analysis using the pipeline
classifier = pipeline("sentiment-analysis", model=model_name)
results = classifier([
    "We are happy to show the transformers Library.",
    "We hope you don't hate it."
])

# Display the sentiment analysis results
for result in results:
    print(result)

# Tokenization and token ID conversion
text_to_tokenize = "We are happy to show the transformers Library."

# Tokenization is the process of breaking text into individual words or tokens.
tokens = tokenizer.tokenize(text_to_tokenize)

# Convert the tokens to their corresponding token IDs.
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# You can also tokenize and obtain token IDs in a single step using the tokenizer.
input_ids = tokenizer(text_to_tokenize)

# Display the tokens and their corresponding IDs
print(f"    Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
print(f"Token IDs: {input_ids}")

# Define training data
X_train = [
    "We are happy to show the transformers Library.",
    "We hope you don't hate it."
]

# Tokenize the training data in batches with padding and truncation
batch = tokenizer(
    X_train,
    padding=True,  # Add padding to make all sequences the same length
    truncation=True,  # Truncate sequences that exceed the specified maximum length
    max_length=512,  # Set the maximum sequence length
    return_tensors="pt"  # Return PyTorch tensors
)

# Perform sentiment analysis using the loaded model
with torch.no_grad():
    # Feed the tokenized training data to the model
    outputs = model(**batch)  # Unpack batch data for PyTorch

    # Apply softmax to get probability scores for each sentiment class
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)

    # Find the label with the highest probability as the predicted sentiment
    labels = torch.argmax(predictions, dim=1)
    print(labels)

    # Convert label IDs to human-readable labels using the model's configuration
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)


