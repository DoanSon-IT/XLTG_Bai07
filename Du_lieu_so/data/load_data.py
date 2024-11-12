# data/load_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_split_data():
    # Đọc dữ liệu từ Iris.csv
    df = pd.read_csv("Du_lieu_so/data/Iris.csv")
    # Lấy các thuộc tính và nhãn
    X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    y = df["Species"]
    
    # Mã hóa nhãn thành số
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Chia dữ liệu thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoder
