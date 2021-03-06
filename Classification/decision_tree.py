import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
    
    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test=X[test_idx,:], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:, 1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='test set')

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:,[2,3]]
    y=iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)
    tree.fit(X_train, y_train)

    X_combined=np.vstack((X_train,X_test))
    y_combined=np.hstack((y_train,y_test))

    plot_decision_regions(X_combined,y_combined,classifier=tree,test_idx=range(105,150))
    plt.suptitle('Decision Tree Classifier')
    plt.xlabel('Petal Length [cm]')
    plt.ylabel('Petal Width [cm]')
    plt.legend(loc='upper left')
    plt.title('Iris Flower Classification')
    plt.show()

    print("Accuracy: {}".format(100*tree.score(X_test, y_test)))
