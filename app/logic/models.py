from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def train_random_forest(preprocessor, X_train, y_train):
    rf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(preprocessor, X_train, y_train):
    xgb = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])
    xgb.fit(X_train, y_train)
    return xgb
