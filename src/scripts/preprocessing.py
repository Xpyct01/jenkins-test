from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pathlib


class Preprocessing:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.iris_data = None
        self.base_dir = "processing"

    def extract_data(self):
        self.iris_data = load_iris()

    def transform_data(self):
        data = pd.DataFrame(data=np.c_[self.iris_data['data'], self.iris_data['target']],
                            columns=self.iris_data['feature_names'] + ['target'])
        self.train_data, self.test_data = train_test_split(data, test_size=.3, random_state=17)

    def load_data(self):
        train_data_path = f"{self.base_dir}/data/train/"
        pathlib.Path(train_data_path).mkdir(parents=True, exist_ok=True)
        self.train_data.to_csv(f"{train_data_path}/train.csv")

        test_data_path = f"{self.base_dir}/data/test/"
        pathlib.Path(test_data_path).mkdir(parents=True, exist_ok=True)
        self.test_data.to_csv(f"{test_data_path}/test.csv")

    def preprocess_data(self):
        self.extract_data()
        self.transform_data()
        self.load_data()


if __name__ == "__main__":
    Preprocessing().preprocess_data()
