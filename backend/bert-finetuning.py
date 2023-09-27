from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import TrainingArguments
from transformers import AdamW
import torch.nn as nn
import random


pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


dataframe = pd.read_csv("user_reviews.csv")
dataframe = dataframe.drop(['User','Item'] , axis = 1)
dataframe = dataframe.iloc[:,:].values

ratings_list = []
review_list = []
for info in dataframe :
    ratings_list.append(info[0])
    review_list.append(info[1])

data_list = []
for i in range(len(review_list)) :
    inputs = tokenizer(review_list[i] ,  return_tensors="pt")
    data_list.append([inputs, ratings_list[i]])

    
training_args = TrainingArguments(output_dir="fine_tunerr")


model.classifier = nn.Identity()
model.add_module("classifier1", nn.Linear(768, 400))  # Add your first linear layer
model.add_module("classifier2", nn.Linear(400, 64))  # Add your second linear layer for classification
model.add_module("classifier3", nn.Linear(64, 5))

classifier1_params = model.classifier1.parameters()
classifier2_params = model.classifier2.parameters()
classifier3_params = model.classifier3.parameters()

for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False

device = 'cuda'
model.to(device)


optimizer_params = list(classifier1_params) + list(classifier2_params) + list(classifier3_params)


optim = AdamW(optimizer_params, lr=1e-4)
loss_f= nn.CrossEntropyLoss()

epochs = 30
tot_len = len(data_list)
m = nn.Softmax(dim=1)

new_data_list = data_list

max_accuracy = 0

for epoch in range(epochs):
    train_loss = 0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    random.shuffle(new_data_list)
    data_list = new_data_list
    # print(data_list[0])
    test_list = data_list[len(data_list)-150:]
    data_list = data_list[:len(data_list)-150]
    # print(data_list[0])
    for i in range(len(data_list)):
        # if count >=1 :
        #     break
        input = data_list[i][0]
        # print(input)
        rating = data_list[i][1]
        optim.zero_grad()
        outputs = model(**input.to('cuda'))
        output1 = model.classifier1(outputs.logits)
        output2 = model.classifier2(output1)
        output3 = model.classifier3(output2)
        pos = torch.argmax(output3)
        new_output = torch.zeros_like(output3).to(device)
        new_output[0][rating-1] = 1
        loss = loss_f(m(output3) , new_output)

        train_loss += loss.clone().detach()
        if pos == rating-1 :
            train_acc += 1
        loss.backward()
        optim.step()
        # count += 1

    for info in test_list :
        input = info[0]
        rating = info[1]
        outputs = model(**input.to('cuda'))
        output1 = model.classifier1(outputs.logits)
        output2 = model.classifier2(output1)
        output3 = model.classifier3(output2)
        pos = torch.argmax(output3)
        new_output = torch.zeros_like(output3).to(device)
        new_output[0][rating-1] = 1
        loss = loss_f(m(output3) , new_output)
        test_loss += loss.clone().detach()
        if pos == rating-1 :
            test_acc += 1

    print(f'training loss for epoch = {epoch+1}: ',train_loss/(tot_len-150))
    print(f'training accuracy for epoch = {epoch+1}: ',train_acc/(tot_len-150))
    print(f'testing loss for epoch = {epoch+1}: ',test_loss/150)
    print(f'testing accuracy for epoch = {epoch+1}: ',test_acc/150)

    if max_accuracy < 2*train_acc/(3*(tot_len-150))+test_acc/450 :
        max_accuracy = 2*train_acc/(3*(tot_len-150))+test_acc/450
        print()
        print("model weights changed at epoch ",epoch+1)
        print()
        torch.save(model.state_dict(), "./model/sentiment_bert.pt")

    print("---------------------------")





