import os
import cv2
import joblib
import preprocessing

def predict_image(image_path, model_path):
    model = joblib.load(model_path)
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (255, 255))
    img_reshaped = img_resized.reshape(1, -1)
    label = model.predict(img_reshaped)
    return label[0], img_resized

if __name__ == "__main__":
    image_path = "check_var_4.jpg"
    
    # Tạo thư mục output nếu chưa có
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Đường dẫn các mô hình
    knn_model_path = "knn_model.pkl"
    svm_model_path = "svm_model.pkl"
    ann_model_path = "ann_model.pkl"

    # Dự đoán bằng mô hình KNN
    knn_label, knn_img = predict_image(image_path, knn_model_path)
    cv2.putText(knn_img, f"KNN: {knn_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_folder, "knn_prediction.jpg"), knn_img)

    # Dự đoán bằng mô hình SVM
    svm_label, svm_img = predict_image(image_path, svm_model_path)
    cv2.putText(svm_img, f"SVM: {svm_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_folder, "svm_prediction.jpg"), svm_img)

    # Dự đoán bằng mô hình ANN
    ann_label, ann_img = predict_image(image_path, ann_model_path)
    cv2.putText(ann_img, f"ANN: {ann_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_folder, "ann_prediction.jpg"), ann_img)

    print("Saved predictions to 'output' folder as: 'knn_prediction.jpg', 'svm_prediction.jpg', 'ann_prediction.jpg'")
