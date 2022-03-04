import os
import datasets
import numpy as np

from iz.crossfit.fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class YahooAnswersTopics(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "yahoo_answers_topics"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"Society & Culture",
            1:"Science & Mathematics",
            2:"Health",
            3:"Education & Reference",
            4:"Computers & Internet",
            5:"Sports",
            6:"Business & Finance",
            7:"Entertainment & Music",
            8:"Family & Relationships",
            9:"Politics & Government",
        }

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "test")

        return train_lines, test_lines
        
    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            input_text = "question_title: " + datapoint["question_title"] + " [SEP] question_content: " + datapoint["question_content"] + " [SEP] best_answer: " + datapoint["best_answer"]
            lines.append((input_text, self.label[datapoint["topic"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('yahoo_answers_topics')

def main():
    dataset = YahooAnswersTopics()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()