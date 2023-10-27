from transformers import pipeline

import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

classifier = pipeline("sentiment-analysis", model_name)
results = classifier(["We are happy to show the transformers Library.",
                    "We hope you don't hate it."])

for result in results:
    print(result)
