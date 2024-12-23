import joblib

kmenas_loaded=joblib.load('kmeans.pkl')
input_data=[[0,1]]
prediction=kmenas_loaded.predict(input_data)
print(prediction)