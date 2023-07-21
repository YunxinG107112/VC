import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'sepal length':3.5, 'sepal width':1.2, 'petal length':2.4,'petal width':0.2})

print(r.json())
