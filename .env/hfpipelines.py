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
