```python
from sklearn.ensemble import RandomForestClassifier
```

```python
rf_model = RandomForestClassifier(n_estimators=300, max_depth=None, max_samples=None, max_features=0.1)
```

```python
X_train = train_partition[all_features].replace('?', 0)
y_train = train_partition['delay']

#rf_model.fit(X_train, y_train)
rf_model.fit(X_resampled, y_resampled)
```

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score

X_train = train_partition[all_features].replace('?', 0)
y_train = train_partition['delay']

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_rf_model = random_search.best_estimator_

X_test = test_partition[all_features].replace('?', 0)
y_test = test_partition['delay']

accuracy = accuracy_score(y_test, best_rf_model.predict(X_test))
print(f"Test Accuracy (tuned model): {accuracy:.4f}")

y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC (tuned model): {roc_auc:.4f}")
```

```python
#Ez bugolik
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score

param_grid = {
    'n_estimators': [200, 250, 300],
    'max_features': ['log2', 0.1],
    'max_depth': [30, 40, None],
    'max_samples': [None, 0.1, 0.2, 0.5],
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    verbose=2,
    n_jobs=-1
)

#grid_search.fit(X_t, y_t)
grid_search.fit(X_t, y_t)

best_rf_grid_model = grid_search.best_estimator_

print("Best parameters found by Grid Search:")
print(grid_search.best_params_)

X_test = test_partition[all_features].replace('?', 0)
y_test = test_partition['delay']

accuracy_grid = accuracy_score(y_test, best_rf_grid_model.predict(X_test))
print(f"Test Accuracy (tuned by Grid Search): {accuracy_grid:.4f}")

y_pred_proba_grid = best_rf_grid_model.predict_proba(X_test)[:, 1]
roc_auc_grid = roc_auc_score(y_test, y_pred_proba_grid)
print(f"ROC AUC (tuned by Grid Search): {roc_auc_grid:.4f}")
```

```python
from sklearn.metrics import roc_auc_score, accuracy_score

# Ensure 'place' columns in test_partition are remapped
features2_test = [col for col in test_partition.columns if col.endswith("place")]
test_partition[features2_test] = test_partition[features2_test].apply(lambda x: lookup(x).numpy())

# Select features for X_test after remapping and ensure column order matches X_resampled
X_test = test_partition[X_resampled.columns].replace('?', 0)
y_test = test_partition['delay']

accuracy = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc:.4f}")
```

```python
from sklearn.metrics import roc_auc_score, accuracy_score


X_test = test_partition[all_features].replace('?', 0)
y_test = test_partition['delay']

accuracy = accuracy_score(y_test, best_rf_model.predict(X_test))
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc:.4f}")
```

```python
# Print the best parameters found by the random search
print(random_search.best_params_)
```
