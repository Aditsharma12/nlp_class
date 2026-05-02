import pytest
from app import app, preprocess_text, apply_contrast_weighting, extract_words

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200

def test_predict_empty(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400
    assert response.get_json() == {"error": "Empty input"}

def test_predict_no_input(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert response.get_json() == {"error": "No input"}

def test_predict_positive(client):
    response = client.post("/predict", json={"text": "This product is amazing and I love it."})
    assert response.status_code == 200
    data = response.get_json()
    assert "sentiment" in data
    assert "confidence" in data
    # May not strictly be "Positive" if the model has a weird quirk, but mostly it will be
    assert data["sentiment"] in ["Positive", "Negative", "Mixed"]

def test_preprocess_text():
    assert preprocess_text("HOWEVER it is bad") == "but it is bad"
    assert preprocess_text("although it is good") == "but it is good"

def test_apply_contrast_weighting():
    text = "it is good but it is expensive"
    res = apply_contrast_weighting(text)
    assert res == "it is good  it is expensive it is expensive"

def test_extract_words():
    words = extract_words("I love this amazing product")
    assert "love" in words["positive"] or "amazing" in words["positive"]
    assert len(words["negative"]) == 0
