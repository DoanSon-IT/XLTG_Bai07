# training.py
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from data.load_data import load_and_split_data

os.makedirs("Du_lieu_so/models", exist_ok=True)

def train_and_save_model(model, model_path):
    X_train, _, y_train, _, _ = load_and_split_data()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

if __name__ == "__main__":
    # Khởi tạo các mô hình
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC()
    ann = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000)

    # Huấn luyện và lưu các mô hình
    train_and_save_model(knn, "Du_lieu_so/models/knn_model.pkl")
    train_and_save_model(svm, "Du_lieu_so/models/svm_model.pkl")
    train_and_save_model(ann, "Du_lieu_so/models/ann_model.pkl")
    
    print("Models trained and saved successfully!")
