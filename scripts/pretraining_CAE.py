from data_utilities.Dataset_module import RadiologyImages
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from rj_models.RAD_CAE import RadiologyCAE
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

train_dataset = RadiologyImages(image_folder="/home/monica/Research/Insight/Project/PreTrain/", image_ext='.png')

batch_size = 1
n_iters = 12000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = RadiologyCAE()

if torch.cuda.is_available():
    print("Using GPU")
    model.cuda()

learning_rate = 0.001
criterion = nn.MSELoss()

optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

iter = 0
loss_list = list()
last_op = None
last_img = None
done = False
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        print(images.size())
        if torch.cuda.is_available():
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, images)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        print(loss.item())
        loss_list.append(loss.item())
        last_op = outputs
        last_img = images

        if loss <= 0.001:
            done = True
            break
    if done:
        break

plt.plot(loss_list)
trans_pil = transforms.ToPILImage()

decode_image = last_op[2].data.cpu()
actual_image = last_img[2].data.cpu()

decode_image = trans_pil(decode_image)
actual_image = trans_pil(actual_image)

plt.show()
decode_image.show()
actual_image.show()

# model_path = "/home/monica/Research/Insight/learned_models/image_model3"
# torch.save(model, model_path)