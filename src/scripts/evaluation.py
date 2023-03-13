from sklearn.metrics import accuracy_score
import json
import joblib
import pathlib
import pandas as pd


if __name__ == "__main__":
    base_dir = "processing"

    model_path = f"{base_dir}/model/model.gz"
    model = joblib.load(model_path)

    test_data_path = f"{base_dir}/data/test/test.csv"
    test_data = pd.read_csv(test_data_path)

    test_features = test_data.drop(['target'], axis=1)
    test_labels = test_data['target']

    predictions = model.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    report_dict = {
        "accuracy":  accuracy
    }

    evaluation_path = f"{base_dir}/evaluation/"
    pathlib.Path(evaluation_path).mkdir(parents=True, exist_ok=True)
    with open(f"{evaluation_path}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))
