import kivy

kivy.require('1.9.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
import os
import torch
from torch.autograd import Variable

from config.global_path import AB_PATH
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from data_utilities.Dataset_module import VQAClassificationDataset
from rj_models.VQA_classification import VQAClassifier
from rj_models.RAD_CAE import RadiologyCAE

from PIL import Image

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

# You can create your kv code in the Python file
Builder.load_file('./interface.kv')


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)


class ScreenTwo(Screen):
    text_input_ans = ObjectProperty(None)

    def set_text(self, ip_text):
        self.text_input_ans.text = ip_text


class ScreenOne(Screen):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def __int__(self):
        super(ScreenOne, self).__init__()
        print("Init is being called")

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        self.img.source = filename[0]
        self.img.reload()

        pil_img = Image.open(self.img.source)

        grayscale_transform = transforms.Grayscale(num_output_channels=1)
        gray_image = grayscale_transform(pil_img)

        resize_transform = transforms.Resize((128, 128))
        resized_img = resize_transform(gray_image)

        pil2tensor_transform = transforms.ToTensor()
        self.image_tensor = pil2tensor_transform(resized_img)

        self.image_tensor = self.image_tensor.reshape(1, 1, 128, 128)

        self.dismiss_popup()
        return os.path.join(path, filename[0])

    def save(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            stream.write(self.text_input.text)

        self.dismiss_popup()

    def show_answers(self):
        q1 = self.left_ip.text
        q2 = self.center_ip.text
        q3 = self.right_ip.text

        if torch.cuda.is_available():
            img_vecs = Variable(self.image_tensor.cuda())
        else:
            img_vecs = Variable(self.image_tensor)

        if not q1 == '':
            # Forward pass to get output/logits
            outputs = vqa_model(img_vecs, [q1])

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = predicted.cpu().numpy()[0]
            answer = answer_dict.keys()[answer_dict.values().index(predicted_label)]

            self.ans_left_ip.text = answer
        else:
            self.ans_left_ip.text = ''

        if not q2 == '':
            # Forward pass to get output/logits
            outputs = vqa_model(img_vecs, [q2])

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = predicted.cpu().numpy()[0]
            answer = answer_dict.keys()[answer_dict.values().index(predicted_label)]

            self.ans_center_ip.text = answer
        else:
            self.ans_center_ip.text = ''

        if not q3 == '':
            # Forward pass to get output/logits
            outputs = vqa_model(img_vecs, [q3])

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = predicted.cpu().numpy()[0]
            answer = answer_dict.keys()[answer_dict.values().index(predicted_label)]

            self.ans_right_ip.text = answer
        else:
            self.ans_right_ip.text = ''


# The ScreenManager controls moving between screens
screen_manager = ScreenManager()

# Add the screens to the manager and then supply a name
# that is used to switch screens
screen_manager.add_widget(ScreenOne(name="screen_one"))
screen_manager.add_widget(ScreenTwo(name="screen_two"))


class RadiologyJr(App):

    def build(self):
        return screen_manager


sample_app = RadiologyJr()
sample_app.run()