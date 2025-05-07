import requests

def test_api_status_code():
    response = requests.get("http://localhost:8000/docs")
    assert response.status_code == 200
	
def test_api_status_code_train():
    response = requests.post("http://localhost:8000/training/")
    assert response.status_code == 200
	
def test_api_status_code_predict():
    response = requests.post("http://localhost:8000/prediction/")
    assert response.status_code == 200