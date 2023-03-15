from sklearn.svm import SVC
import pandas as pd
import joblib
import pathlib


class TrainingModel:
    def __init__(self):
        self.train_labels = None
        self.train_features = None
        self.model = None
        self.base_dir = "processing"

    def get_data(self):
        data = pd.read_csv(f"{self.base_dir}/data/train/train.csv")
        self.train_features = data.drop(['target'], axis=1)
        self.train_labels = data['target']

    def train_model(self):
        self.model = SVC(random_state=17)
        self.model.fit(self.train_features, self.train_labels)

    def save_model(self):
        model_path = f"{self.base_dir}/model/"
        docker_path = "../container/"
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, f"{model_path}/model.gz")
        joblib.dump(self.model, f"{docker_path}/model.gz")

    def get_model(self):
        self.get_data()
        self.train_model()
        self.save_model()


if __name__ == "__main__":
    TrainingModel().get_model()
