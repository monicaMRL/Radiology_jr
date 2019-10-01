import os
from config_variables import AB_PATH
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from Dataset_module import VQAClassificationDataset
from VQA_classification_models import VQAClassifier
from RAD_CAE_model import RadiologyCAE

raw_data_path = os.path.join(AB_PATH, "unformatted_data/qa_textdata_final.json")
image_folder = "/home/monica/Research/Insight/data_generation/images"

classification_dataset = VQAClassificationDataset(image_folder, raw_data_path)

input_dim = 149896  # --> [200 * 27 * 27] + [4096]
output_dim = classification_dataset.number_classes  # --> 338 (Answer vocabulary size)

vqa_model = VQAClassifier(input_dim, output_dim)

if vqa_model.use_cuda:
    vqa_model.cuda()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = torch.optim.Adam(vqa_model.parameters(), lr=learning_rate)

batch_size = 1
n_iters = 60000
num_epochs = n_iters / (len(classification_dataset) / batch_size)
num_epochs = int(num_epochs)

classification_dataloader = DataLoader(classification_dataset, batch_size=batch_size, shuffle=True)

loss_list = list()
epoch_number = 1
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (img, q, q_len, a) in enumerate(classification_dataloader):
        if torch.cuda.is_available():
            img_vecs = Variable(img.cuda())
            labels = Variable(a.cuda())
        else:
            img_vecs = Variable(img)
            labels = Variable(a)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = vqa_model(img_vecs, q)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        loss_list.append(loss.item())

    print(epoch_number, epoch_loss)
    epoch_number += 1

plt.plot(loss_list)
