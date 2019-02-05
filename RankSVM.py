import itertools
import numpy as np
from scipy import stats
import pylab as pl
from sklearn import svm, linear_model, cross_validation
from numpy.linalg import norm
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import fbeta_score, make_scorer
from math import sqrt

if __name__=='__main__':
    
    weights=[]
    accuracy=[]
    error=[]
    
    for i in range(20):
        
        X= genfromtxt('train_data.csv', delimiter=',')
        y= genfromtxt('ego_rank_label2.csv', delimiter=',')
        
        blocks = np.array([0,1] * int(X.shape[0]/2))
    
        # split into train and test set
        cv = cross_validation.StratifiedShuffleSplit(y, test_size=.3)
        train, test = iter(cv).__next__()
        X_train, y_train, b_train = X[train], y[train], blocks[train]
        X_test, y_test, b_test = X[test], y[test], blocks[test]
        
        # form all pairwise combinations
        comb = itertools.combinations(range(X_train.shape[0]), 2)
        k = 0
        Xp, yp, diff = [], [], []
        for (i, j) in comb:
            if y_train[i] == y_train[j] \
                or blocks[train][i] != blocks[train][j]:
                # skip if same target or different group
                continue
            Xp.append(X_train[i] - X_train[j])
            diff.append(y_train[i] - y_train[j])
            yp.append(np.sign(diff[-1]))
            # output balanced classes
            if yp[-1] != (-1) ** k:
                yp[-1] *= -1
                Xp[-1] *= -1
                diff[-1] *= -1
            k += 1
        Xp, yp, diff = map(np.asanyarray, (Xp, yp, diff))
        
        
        pl.scatter(Xp[:, 0], Xp[:, 1], c=diff, s=60, marker='o', cmap=pl.cm.Blues)
        x_space = np.linspace(-10, 10)
    
        clf = svm.SVC(kernel='linear', C=1.0)
        clf.fit(Xp, yp)
        coef = clf.coef_.ravel() / norm(clf.coef_)
        y_pred= clf.predict(Xp)
       
        
        weights.append(coef)
        accuracy.append(clf.score(Xp, yp))
        error.append(mean_squared_error(yp, y_pred))
    
    degree=[]
    closeness=[]
    betweenness=[]
    eigenvector=[]
    
    for i in range(len(weights)):
        
        degree.append(weights[i][0])
        closeness.append(weights[i][1])
        betweenness.append(weights[i][2])
        eigenvector.append(weights[i][3])
        
  
    print('global ego ranking:') 
    print('degree_centrality weight=', np.mean(degree))
    print('closeness_centrality weight=',np.mean(closeness))
    print('betweenness_centrality weight=',np.mean(betweenness))
    print('eigenvector_centrality weight=',np.mean(eigenvector))
    
   
    print('degree_centrality std=', np.std(degree))
    print('closeness_centrality stdt=',np.std(closeness))
    print('betweenness_centrality std=',np.std(betweenness))
    print('eigenvector_centrality std=',np.std(eigenvector))
    
    
    print('Accuracy=',np.mean(accuracy))
    print('Accuracy standard deviation: ',np.std(accuracy))
    
    print('Mean Squared Error=',np.mean(error))
    print('Root Mean Squared Error=',sqrt(np.mean(error)))


