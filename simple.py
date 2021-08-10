import pandas as pd
df = pd.read_csv("Iris.csv")
X = df.drop(‘Species’, 'Id', axis=1)
y = df[‘Species’]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)
from sklearn import model_selection
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
rfc_probs= rfc.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_auc_score 
roc_value = roc_auc_score(y_test, rfc_probs)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring=’roc_auc’)
print(rfc_cv_score) #prints all of them
print(rfc_cv_score.mean()) #prints average of all of them
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = [‘auto’, ‘sqrt’]
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
random_grid = {
 ‘n_estimators’: n_estimators,
 ‘max_features’: max_features,
 ‘max_depth’: max_depth
 }
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rfc_random.fit(X_train, y_train)
print(rfc_random.best_params_)
rfc = RandomForestClassifier(n_estimators=600, max_depth=300, max_features='sqrt')
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
print(rfc_cv_score) 
print(rfc_cv_score.mean())
