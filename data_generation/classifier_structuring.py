"""
Scripts converts the scraped and processed data in to required VQA format for classifier.

Format: list of dictionaries
        {
            image_name: str
            question: str
            answer_label: int
        }
"""
import json
import unittest


def answer_vocabulary(raw_file_name):
    """
    Gathers all the unique words form answer text

    :param raw_file_name: absolute path to the file containing unformatted data in the form
            {
                image_name: str
                question: str
                answer_label: str
            }
    :type raw_file_name: str

    :return: list of unique words combining all answer's first word (Class Labels)

    """
    qa_list = json.load(open(raw_file_name))
    all_words = list()

    for qa_dict in qa_list:
        answer = str(qa_dict["answer"]).lower()
        all_words += [answer.split(" ")[0]]

    unique_words = list(set(all_words))

    print("Answer vocabulary size consisting only first word is: {}".format(len(unique_words)))

    return unique_words


def convert2labels(unique_words, destination_file=None, save2file=False):
    """
    Converts text name of classes to unique integer value and writes the dictionary to json file

    :param unique_words: list of unique labels.
    :type unique_words: list
    :param destination_file: full path to file to store the label to integer value dictionary, None by default
    :type destination_file: str
    :param save2file: Bool value to indicate if dictionary needs to be saved to the file. False by default.
    :type save2file: bool

    :return: dictionary of labels as keys and integer representation as value

    """
    answer_labels = dict()

    for i in range(len(unique_words)):
        answer_labels[unique_words[i]] = i

    if save2file:
        unittest.TestCase.assertIsNotNone(destination_file)
        json.dump(answer_labels, open(destination_file, "w"), indent=4)

    return answer_labels


def restructure_file(raw_file_name, destination_file, label_file):
    """
    Generates new structured file of the data (Not used when Dataset class is made of the data.)

    :param raw_file_name: full path to file with unstructured data
    :type raw_file_name: str
    :param destination_file: full path to file to store new structured data
    :type destination_file: str
    :param label_file: full path to file to store word labels to integer dictionary
    :type label_file: str

    :return: None
    """
    unique_words = answer_vocabulary(raw_file_name)
    answer_labels = convert2labels(unique_words, label_file)

    qa_list = json.load(open(raw_file_name))
    new_qa_list = list()

    for qa_dict in qa_list:
        answer = str(qa_dict["answer"]).lower()
        first_word = answer.split(" ")[0]

        new_qa_dict = {
            "image_name": qa_dict["image_name"],
            "question": qa_dict["question"],
            "answer_label": answer_labels[first_word]
        }

        new_qa_list.append(new_qa_dict)

    json.dump(new_qa_list, open(destination_file, "w"), indent=4)