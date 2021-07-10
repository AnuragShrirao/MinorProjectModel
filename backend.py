from joblib import dump, load

def predict(x,y,z,k):
    model = load("model.ml")
    # print(model.predict([[x,y,z,k]]))
    return model.predict([[x,y,z,k]])[0]