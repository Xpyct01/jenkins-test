from sklearn.metrics import accuracy_score
import json
import joblib
import pathlib
import pandas as pd


class Evaluate:
    def __init__(self):
        self.report_dict = None
        self.test_labels = None
        self.test_features = None
        self.model = None
        self.base_dir = "processing"

    def load_model(self):
        model_path = f"{self.base_dir}/model/model.gz"
        self.model = joblib.load(model_path)

    def load_data(self):
        test_data_path = f"{self.base_dir}/data/test/test.csv"
        test_data = pd.read_csv(test_data_path)

        self.test_features = test_data.drop(['target'], axis=1)
        self.test_labels = test_data['target']

    def evaluate_model(self):
        predictions = self.model.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, predictions)
        self.report_dict = {
            "accuracy": accuracy
        }

    def save_evaluation(self):
        evaluation_path = f"{self.base_dir}/evaluation/"
        pathlib.Path(evaluation_path).mkdir(parents=True, exist_ok=True)
        with open(f"{evaluation_path}/evaluation.json", "w") as f:
            f.write(json.dumps(self.report_dict))

    def evaluation_process(self):
        self.load_model()
        self.load_data()
        self.evaluate_model()
        self.save_evaluation()


if __name__ == "__main__":
    Evaluate().evaluation_process()
