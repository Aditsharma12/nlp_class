# NLP Sentiment Analysis Engine

A highly optimized dual-pipeline Sentiment Analysis web application. This engine evaluates text to determine whether the sentiment is **Positive**, **Negative**, or **Mixed** by combining Deep Learning with Structural NLP rules.

## 🏗️ System Architecture

Our application runs a hybrid system composed of a Deep Learning (TensorFlow) model for global semantic understanding, and a VADER-based rule heuristic for local clause understanding.

```mermaid
flowchart TD
    User([User Website]) --> |POST /predict| FlaskAPI[Flask API: app.py]
    
    subgraph "Backend Engine"
        FlaskAPI --> Preprocessor[Text Preprocessor]
        
        Preprocessor --> |TF-IDF Features| DeepLearningPath
        Preprocessor --> |Raw Text| RulePath
        
        subgraph DeepLearningPath [Deep Learning Pipeline]
            TFIDF[TF-IDF Vectorizer\n_vectorizer.pkl_] --> ONNXInference[ONNX Inference Engine\n_model.onnx_]
            ONNXInference --> |Softmax Probabilities| PredDL[DL Prediction]
        end
        
        subgraph RulePath [Structural NLP]
            VADER[NLTK VADER Lexicon] --> Words[Word Extraction]
            VADER --> Clauses[Clause Splitting]
            Words --> |Specific terms| Rules[Rule Constraints]
            Clauses --> |Contrasts e.g. 'but'| Rules
            Rules --> PredRules[Heuristic Flags]
        end
        
        PredDL --> Combiner{Logic Combiner}
        PredRules --> Combiner
    end
    
    Combiner --> JSONOut([JSON Response])
    JSONOut --> User
```

## 🧠 Deep Learning Pipeline (Training)

The model is trained dynamically using `train.py`. The transition to TensorFlow Keras prevents overfitting by enforcing regularization and early stopping criteria.

```mermaid
flowchart LR
    A[Raw Dataset\nAmazon Polarity] --> B(Text Preprocessing & Weighting)
    B --> C[TF-IDF Vectorizer\n10k Features]
    
    subgraph Keras[TF / Keras Architecture]
        D(Dense 256 + L2) --> E(Batch Norm)
        E --> F(ReLU)
        F --> G(Dropout 0.4)
        G --> H(Dense 128 + L2)
        H --> I(...)
        I --> J(Softmax\n2 Neurons)
    end
    
    C --> Keras
    
    Keras --> |SavedModel| T2O[tf2onnx CLI]
    T2O --> ONNX[(model.onnx)]
    C --> |pickle| PKL[(vectorizer.pkl)]
```

## 📦 File Structure

- **`app.py`**: The production Flask application serving endpoints and logic.
- **`train.py`**: The standalone training script for generating the AI artifacts.
- **`model.onnx`**: The compiled execution graph of our TensorFlow model.
- **`vectorizer.pkl`**: The fitted mathematical dictionary used for transforming strings into tensors.
- **`requirements.txt`**: Python dependencies.
- **`templates/`**: Holds the frontend application files (`index.html`).
- **`test_app.py`**: Test suite for the application using `pytest`.
- **`Dockerfile`**: Defines the container image for running the app in isolated environments.
- **`Makefile`**: Handy shortcuts for common tasks (e.g. testing, building).
- **`.github/workflows/`**: Continuous Integration (CI) configuration for GitHub Actions.

## ⚙️ MLOps & Testing

This project employs MLOps best practices:
- **CI/CD Pipeline**: GitHub actions are configured to automatically run code checks and tests.
- **Testing**: The test suite covers both unit testing and API endpoint testing. Use `pytest` to run tests locally:
  ```bash
  pytest test_app.py -v
  ```

## 🐳 Docker Deployment

To build and run the application within a Docker container:
```bash
docker build -t nlp_sentiment_app .
docker run -p 5000:5000 nlp_sentiment_app
```

## 🛠️ Make Commands

Use `make` for quick execution of recurring tasks:
- `make run`: Starts the Flask app.
- `make test`: Runs `pytest`.
- `make build`: Builds the Docker image.
- `make docker-run`: Runs the container locally.

## 🚀 Quick Start

1. Ensure Python 3.10 is installed (TensorFlow requirement).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Boot the API server:
   ```bash
   python app.py
   ```
4. Access the UI via `http://localhost:5000`
