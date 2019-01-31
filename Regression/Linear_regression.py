import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
#print(df.head())

X=df[['RM']].values
y=df['MEDV'].values

lr=LinearRegression()
lr.fit(X,y)
print('Slope: %.3f' % lr.coef_[0])
print('Intercept: %.3f' % lr.intercept_)
print("Accuracy: {}".format(100*lr.score(X,y)))

# plotting the regression line
def lin_regplot(X,y,model):
    """ Helper function to plot scatterplot of training samples and add regression line """
    plt.scatter(X,y,c='m',edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None

lin_regplot(X,y,lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in  $1000s [MEDV] (standardized)')
plt.show()
