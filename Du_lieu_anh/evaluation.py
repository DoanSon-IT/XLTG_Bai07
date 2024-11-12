import joblib
import time
import numpy as np
from sklearn.metrics import accuracy_score
import preprocessing  # File xử lý dữ liệu

def evaluate_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    accuracy = accuracy_score(y_test, y_pred)
    elapsed_time = end_time - start_time
    return accuracy, elapsed_time

if __name__ == "__main__":
    # Đọc dữ liệu test từ preprocessing.py
    X_test, y_test = preprocessing.load_images_and_labels("Data/")
    X_test = X_test.reshape(len(X_test), -1)  # Chuyển đổi hình ảnh thành vector
    
    # Đường dẫn các mô hình
    knn_model_path = "knn_model.pkl"
    svm_model_path = "svm_model.pkl"
    ann_model_path = "ann_model.pkl"

    # Đánh giá các mô hình
    knn_accuracy, knn_time = evaluate_model(knn_model_path, X_test, y_test)
    svm_accuracy, svm_time = evaluate_model(svm_model_path, X_test, y_test)
    ann_accuracy, ann_time = evaluate_model(ann_model_path, X_test, y_test)

    # Ghi kết quả vào README.md
    with open("README.md", "w") as f:
        f.write("# Model Evaluation Results\n")
        f.write("## KNN Model\n")
        f.write(f"- Accuracy: {knn_accuracy:.2f}\n")
        f.write(f"- Time: {knn_time:.4f} seconds\n\n")
        
        f.write("## SVM Model\n")
        f.write(f"- Accuracy: {svm_accuracy:.2f}\n")
        f.write(f"- Time: {svm_time:.4f} seconds\n\n")
        
        f.write("## ANN Model\n")
        f.write(f"- Accuracy: {ann_accuracy:.2f}\n")
        f.write(f"- Time: {ann_time:.4f} seconds\n")

    print("Evaluation results saved to README.md")
