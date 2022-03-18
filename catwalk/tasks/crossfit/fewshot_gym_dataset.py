import os
import numpy as np

class FewshotGymDataset():
    def write_to_tsv(self, lst, out_file):
        with open(out_file, "w") as fout:
            for line in lst:
                fout.write("{}\t{}\n".format(line[0], line[1]))

class FewshotGymClassificationDataset(FewshotGymDataset):

    def get_train_test_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "validation")
        return train_lines, test_lines

    def generate_k_shot_data(self, k, seed, path=None):
        """
        generate a k-shot (k) dataset using random seed (seed)
        return train, dev, test
        """

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, test_lines = self.get_train_test_lines(dataset)

        # shuffle the data
        np.random.seed(seed)
        np.random.shuffle(train_lines)

        # Get label list for balanced sampling
        label_list = {}
        for line in train_lines:
            label = line[-1]
            if label not in label_list:
                label_list[label] = [line]
            else:
                label_list[label].append(line)

        # make train, dev, test data
        k_shot_train = []
        for label in label_list:
            for line in label_list[label][:k]:
                k_shot_train.append(line)

        k_shot_dev = []
        for label in label_list:
            for line in label_list[label][k:2*k]:
                k_shot_dev.append(line)

        k_shot_test = test_lines

        # save to path
        if path:
            os.makedirs(os.path.join(path, self.hf_identifier), exist_ok=True)
            prefix = os.path.join(path, self.hf_identifier, "{}_{}_{}".format(self.hf_identifier, k, seed))
            self.write_to_tsv(k_shot_train, prefix + "_train.tsv")
            self.write_to_tsv(k_shot_dev, prefix + "_dev.tsv")
            self.write_to_tsv(k_shot_test, prefix + "_test.tsv")

        return k_shot_train, k_shot_dev, k_shot_test

class FewshotGymTextToTextDataset(FewshotGymDataset):

    def get_train_test_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "validation")
        return train_lines, test_lines

    def generate_k_shot_data(self, k, seed, path=None):
        """
        generate a k-shot (k) dataset using random seed (seed)
        return train, dev, test
        """

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, test_lines = self.get_train_test_lines(dataset)

        # shuffle the data
        np.random.seed(seed)
        np.random.shuffle(train_lines)

        # make train, dev, test data
        k_shot_train = []
        for line in train_lines[:k]:
            k_shot_train.append(line)

        k_shot_dev = []
        for line in train_lines[k:2*k]:
            k_shot_dev.append(line)

        k_shot_test = test_lines

        # save to path
        if path:
            os.makedirs(os.path.join(path, self.hf_identifier), exist_ok=True)
            prefix = os.path.join(path, self.hf_identifier, "{}_{}_{}".format(self.hf_identifier, k, seed))
            self.write_to_tsv(k_shot_train, prefix + "_train.tsv")
            self.write_to_tsv(k_shot_dev, prefix + "_dev.tsv")
            self.write_to_tsv(k_shot_test, prefix + "_test.tsv")

        return k_shot_train, k_shot_dev, k_shot_test