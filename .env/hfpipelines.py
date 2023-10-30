from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


classifier = pipeline("sentiment-analysis", model = model_name)
results = classifier(["We are happy to show the transformers Library.",
                    "We hope you don't hate it."])

for result in results:
    print(result)


tokens = tokenizer.tokenize("We are happy to show the transformers Library.")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
# using tokenizer function
input_ids = tokenizer("We are happy to show the transformers Library.")

print(f"    Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
print(f"Token IDs: {input_ids}")


X_train  = ["We are happy to show the transformers Library.",
                    "We hope you don't hate it."]

batch =  tokenizer(X_train, padding = True, truncation = True, max_length = 512, return_tensors = "pt")

with torch.no_grad():
    # outputs = model(batch) # For TensorFlow we don't need to unpack batch data
    outputs = model(**batch) # For pytorch we need to unpack batch data
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)

    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)












