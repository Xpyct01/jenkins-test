from sklearn.svm import SVC
import pandas as pd
import joblib
import pathlib


if __name__ == "__main__":
    base_dir = "processing"

    train_data = pd.read_csv(f"{base_dir}/data/train/train.csv")
    train_features = train_data.drop(['target'], axis=1)
    train_labels = train_data['target']

    model = SVC(random_state=17)
    model.fit(train_features, train_labels)

    model_path = f"{base_dir}/model/"
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, f"{model_path}/model.gz")
