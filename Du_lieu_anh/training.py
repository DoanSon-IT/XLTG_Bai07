from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib
import preprocessing

def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    joblib.dump(knn, 'knn_model.pkl')
    return knn

def train_svm(X_train, y_train):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    joblib.dump(svm, 'svm_model.pkl')
    return svm

def train_ann(X_train, y_train):
    ann = MLPClassifier(hidden_layer_sizes=(255,), max_iter=300)
    ann.fit(X_train, y_train)
    joblib.dump(ann, 'ann_model.pkl')
    return ann

if __name__ == "__main__":
    images, labels = preprocessing.load_images_and_labels("Data/")
    X_train, X_test, y_train, y_test = train_test_split(images.reshape(len(images), -1), labels, test_size=0.2, random_state=42)

    knn_model = train_knn(X_train, y_train)
    svm_model = train_svm(X_train, y_train)
    ann_model = train_ann(X_train, y_train)
