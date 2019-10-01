from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from torchvision import transforms
import os
import json

from data_generation.classifier_structuring import answer_vocabulary, convert2labels
from config_variables import AB_PATH


class VQAClassificationDataset(Dataset):
    """
    Data set class for classifcation problem. It can be wrapped using functions from Dataloader to
    iterate over data in batches.
    """
    def __init__(self, image_folder, raw_data_file):
        """

        :param image_folder: path to folder with all images
        :type image_folder: str

        :param raw_data_file: path to raw question answer data
        :type raw_data_file: str

        """
        self.image_folder = image_folder
        self.qa_list = json.load(open(raw_data_file))

        self.answer_labels = convert2labels(answer_vocabulary(raw_data_file))
        self._n_classes = len(self.answer_labels.keys())

        self.height = 128
        self.width = 128

    def __getitem__(self, item):
        """
        Given an index return a data sample from data set.

        :param item: index at which data is to be retrieved
        :return: image tensor, question, length of question and label of answer
        """
        qa_dict = self.qa_list[item]

        image_name = qa_dict["image_name"]
        image = Image.open(os.path.join(self.image_folder, image_name))

        grayscale_transform = transforms.Grayscale(num_output_channels=1)
        gray_image = grayscale_transform(image)

        resize_transform = transforms.Resize((self.width, self.height))
        resized_img = resize_transform(gray_image)

        pil2tensor_transform = transforms.ToTensor()
        image_tensor = pil2tensor_transform(resized_img)

        question = qa_dict["question"]

        answer = str(qa_dict["answer"]).lower().split(" ")[0]

        answer_label = self.answer_labels[answer]

        return image_tensor, question, len(question), answer_label

    def __len__(self):
        """
        :return: length of complete data set
        """
        return len(self.qa_list)

    @property
    def number_classes(self):
        return self._n_classes


if __name__ == "__main__":
    """
    Class testing + Example 
    """
    raw_data_path = os.path.join(AB_PATH, "unformatted_data/qa_textdata_final.json")
    image_folder = "/home/monica/Research/Insight/data_generation/images"

    classification_dataset = VQAClassificationDataset(image_folder, raw_data_path)
    classification_dataloader = DataLoader(classification_dataset, batch_size=3, shuffle=True)

    for i, (img, q, q_len, a) in enumerate(classification_dataloader):
        print("question", q)
        print("length of question", q_len)
        print("answer", a)
        print("Image", img.shape)