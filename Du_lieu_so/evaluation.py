# evaluation.py
import time
import joblib
from data.load_data import load_and_split_data
from sklearn.metrics import accuracy_score

def evaluate_model(model_path):
    _, X_test, _, y_test, _ = load_and_split_data()
    model = joblib.load(model_path)
    start_time = time.time()
    y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, elapsed_time

if __name__ == "__main__":
    knn_accuracy, knn_time = evaluate_model("Du_lieu_so/models/knn_model.pkl")
    svm_accuracy, svm_time = evaluate_model("Du_lieu_so/models/svm_model.pkl")
    ann_accuracy, ann_time = evaluate_model("Du_lieu_so/models/ann_model.pkl")

    # Ghi kết quả vào README.md
    with open("Du_lieu_so/README.md", "w") as f:
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
