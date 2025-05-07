import requests

def test_api_status_code():
    response = requests.get("http://localhost:8000/docs")
    assert response.status_code == 200
	
def test_api_status_code_train():
    response = requests.post("http://localhost:8000/training/", json={})
    assert response.status_code == 200
	
def test_api_status_code_predict():
    url = "http://localhost:8000/prediction/"
    with open("test_image.jpg", "rb") as f:
        files = {"file": ("test_image.jpg", f, "image/jpeg")}
        data = {
            "id": 1,
            "designation": "Test",
            "description": "Test desc",
            "productid": 123,
            "imageid": 456
        }
        response = requests.post(url, files=files, data=data)
        assert response.status_code == 200
    assert response.status_code == 200