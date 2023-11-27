import h5py
from KNearestNeighbors import KNearestNeighbors

with h5py.File("usps.h5", 'r') as hf:
    train = hf.get('train')
    X_train = train.get('data')[:]  # <class 'numpy.ndarray'> (7291, 256)
    y_train = train.get('target')[:]  # <class 'numpy.ndarray'> (7291,)
    test = hf.get('test')
    X_test = test.get('data')[:]  # <class 'numpy.ndarray'> (2007, 256)
    y_test = test.get('target')[:]  # <class 'numpy.ndarray'> (2007,)

clf = KNearestNeighbors(X_train, y_train, n_neighbors=21)
score = clf.score(X_test, y_test)
print(f"KNN(K=21) 在 USPS 上的精确度为 {score * 100:.2f}%")  # 91.93%

clf.n_neighbors = 7
score = clf.score(X_test, y_test)
print(f"KNN(K=7) 在 USPS 上的精确度为 {score * 100:.2f}%")  # 94.17%

clf.n_neighbors = 5
score = clf.score(X_test, y_test)
print(f"KNN(K=5) 在 USPS 上的精确度为 {score * 100:.2f}%")  # 94.47%

clf.n_neighbors = 1
score = clf.score(X_test, y_test)
print(f"KNN(K=1) 在 USPS 上的精确度为 {score * 100:.2f}%")  # 94.37%
