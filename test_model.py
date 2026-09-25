import sys
import sklearn
import joblib

print("Python:", sys.executable)
print("sklearn version:", sklearn.__version__)
print("sklearn path:", sklearn.__file__)

model = joblib.load("titanic_logistic_model1.pkl")

print("\nModel loaded successfully")
print("Model type:", type(model))
print("Classes:", model.classes_)
print("Has multi_class:", hasattr(model, "multi_class"))

print("\nTrying predict_proba...")
print(model.predict_proba([[1, 1, 25, 0, 0, 80, 0, 2]]))