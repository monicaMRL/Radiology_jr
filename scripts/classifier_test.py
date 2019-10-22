import os
from config.global_path import AB_PATH
from torch.utils.data.dataloader import DataLoader
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from data_utilities.Dataset_module import VQAClassificationDataset
from rj_models.VQA_classification import VQAClassifier
from rj_models.RAD_CAE import RadiologyCAE

raw_data_path = os.path.join(AB_PATH, "unformatted_data/qa_textdata_final.json")
image_folder = "/home/monica/Research/Insight/data_generation/images"
model_path = "/home/monica/Research/Insight/learned_models/vqamodel_300"

classification_dataset = VQAClassificationDataset(image_folder, raw_data_path, save_answer_labels=False)
classification_dataloader = DataLoader(classification_dataset, batch_size=1, shuffle=True)
answer_dict = classification_dataset.label2answer_dict

input_dim = 149896  # --> [200 * 27 * 27] + [4096]
output_dim = classification_dataset.number_classes  # --> 338 (Answer vocabulary size)

vqa_model = VQAClassifier(input_dim, output_dim)
vqa_model.load_state_dict(torch.load(model_path))

if vqa_model.use_cuda:
    vqa_model.cuda()

trans_pil = transforms.ToPILImage()

for i, (img, q, q_len, a) in enumerate(classification_dataloader):
    if torch.cuda.is_available():
        img_vecs = Variable(img.cuda())
        labels = Variable(a.cuda())
    else:
        img_vecs = Variable(img)
        labels = Variable(a)

    # Forward pass to get output/logits
    outputs = vqa_model(img_vecs, q)

    # Get predictions from the maximum value
    _, predicted = torch.max(outputs.data, 1)
    real_label = a
    predicted_label = predicted.cpu().numpy()[0]
    print predicted_label, real_label
    answer = answer_dict.keys()[answer_dict.values().index(predicted_label)]

    print("Question: ", q)
    print("Answer: ", answer)
