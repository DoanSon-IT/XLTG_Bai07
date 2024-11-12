import joblib
import pandas as pd

label_mapping = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Load các mô hình
    knn_model = joblib.load("Du_lieu_so/models/knn_model.pkl")
    svm_model = joblib.load("Du_lieu_so/models/svm_model.pkl")
    ann_model = joblib.load("Du_lieu_so/models/ann_model.pkl")

    # Chuẩn bị dữ liệu đầu vào dưới dạng DataFrame
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

    # Dự đoán với các mô hình
    knn_prediction = label_mapping[knn_model.predict(input_data)[0]]
    svm_prediction = label_mapping[svm_model.predict(input_data)[0]]
    ann_prediction = label_mapping[ann_model.predict(input_data)[0]]

    return knn_prediction, svm_prediction, ann_prediction

if __name__ == "__main__":
    # Nhập các thông số hoa từ người dùng
    sepal_length = float(input("Sepal Length (cm): "))
    sepal_width = float(input("Sepal Width (cm): "))
    petal_length = float(input("Petal Length (cm): "))
    petal_width = float(input("Petal Width (cm): "))

    # Thực hiện dự đoán và in kết quả
    knn_prediction, svm_prediction, ann_prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    print(f"KNN Prediction: {knn_prediction}")
    print(f"SVM Prediction: {svm_prediction}")
    print(f"ANN Prediction: {ann_prediction}")
