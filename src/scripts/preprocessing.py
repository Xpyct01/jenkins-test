from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pathlib


if __name__ == "__main__":
    base_dir = "processing"

    iris_data = load_iris()

    data = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                        columns=iris_data['feature_names'] + ['target'])
    train_data, test_data = train_test_split(data, test_size=.3, random_state=17)

    train_data_path = f"{base_dir}/data/train/"
    pathlib.Path(train_data_path).mkdir(parents=True, exist_ok=True)
    train_data.to_csv(f"{train_data_path}/train.csv")

    test_data_path = f"{base_dir}/data/test/"
    pathlib.Path(test_data_path).mkdir(parents=True, exist_ok=True)
    test_data.to_csv(f"{test_data_path}/test.csv")
