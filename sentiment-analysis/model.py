from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA (GPU) is available.")
else:
    device = torch.device('cpu')
    print("CUDA (GPU) is not available. Using CPU.")

pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model.classifier = nn.Identity()
model.add_module("classifier1", nn.Linear(768, 400))  # Add your first linear layer
model.add_module("classifier2", nn.Linear(400, 64))  # Add your second linear layer for classification
model.add_module("classifier3", nn.Linear(64, 5))

model.to(device)

model.load_state_dict(torch.load('./model/sentiment_bert.pt'))

def get_rating(text) :
    input = tokenizer(text ,  return_tensors="pt")
    outputs = model(**input.to(device))
    output1 = model.classifier1(outputs.logits)
    output2 = model.classifier2(output1)
    output3 = model.classifier3(output2)
    pos = torch.argmax(output3[0])
    return pos.item()+1

print(get_rating('Maggi is too hot but it is tasty'))