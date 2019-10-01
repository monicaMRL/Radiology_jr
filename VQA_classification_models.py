"""
@Author: Monica Patel
"""

import torch.nn as nn
from torch.autograd import Variable
import torch

from pretrained_models.models import InferSent
from RAD_CAE_model import RadiologyCAE


class SaveFeatures():
    """
    @Author: Fabio M. Graetz (https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030)

    Class to put hook on convolution auto encoder layers to extract the embeddings at particular layer.
    """
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output, requires_grad=True).cuda()

    def close(self):
        self.hook.remove()


class VQAClassifier(nn.Module):
    """
    Model class for Visual Question answering Classifier
    """
    model_version = 1
    SENT_MODEL_PATH = "/home/monica/Research/Insight/Pretrained_models/infersent%s.pkl" % model_version
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}

    W2V_PATH = '/home/monica/Research/Insight/Pretrained_models/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
    CAE_MODEL_PATH = "/home/monica/Research/Insight/learned_models/image_model3"

    def __init__(self, input_dim, output_dim, use_cuda=True):
        """

        :param input_dim: input dimention of linear layer, this is length of embeddings obtained from image
        plus embedding obtained from sentence ecoder.

        :param output_dim: Output dimension of linear layer. This is equal to number of classes in the dataset

        """
        super(VQAClassifier, self).__init__()

        # ----------------- Load sentence encoder---------------------------------------
        self.sentence_encoder = InferSent(VQAClassifier.params_model)
        self.sentence_encoder.load_state_dict(torch.load(VQAClassifier.SENT_MODEL_PATH))
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.sentence_encoder = self.sentence_encoder.cuda()

        self.sentence_encoder.set_w2v_path(VQAClassifier.W2V_PATH)

        self.sentence_encoder.build_vocab_k_words(K=100000)
        # ------------------ Wights of sentence encoder are no updated during the learning process ----
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False

        # ---------------------- Load Convolution autoencoder -----------------------------------
        self.cae_model = torch.load(VQAClassifier.CAE_MODEL_PATH)
        self.image_activations = SaveFeatures(list(self.cae_model.children())[6])

        # Weights of convolution auto encoder are fixed during the learning process.
        for param in self.cae_model.parameters():
            param.requires_grad = False

        # Linear layer whose weights are updated during training.
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, image, question):
        """

        :param image: image tensor [batch_size, channel, width, height]
        :type image: torch.Tensor

        :param question: question associated with the image

        :return: softmax distribution over classes.
        """
        img_out = self.cae_model(image)
        image_encodings = self.image_activations.features

        text_encoding = self.sentence_encoder.encode(question, bsize=128, tokenize=False, verbose=False)
        text_encoding = torch.from_numpy(text_encoding)
        if self.use_cuda:
            text_encoding = text_encoding.cuda()

        joint_encoding = self.join_embeddings(image_encodings, text_encoding)

        out = self.fc(joint_encoding)

        return out

    def join_embeddings(self, img_encode, que_encode):
        """

        :param img_encode: CAE features of the image. [batch_size, channel, kernel width, kernel height]
        :param que_encode: encodings obtained from sentence encoder

        :return: concatinated embedding of image and text [batch size, length of encoding]
        """
        reshaped_img_tensor = img_encode.view(-1, img_encode.shape[1] * img_encode.shape[2] * img_encode.shape[3])
        joint_embedding = torch.cat((reshaped_img_tensor, que_encode), dim=1)

        return joint_embedding
