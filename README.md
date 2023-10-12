## EEG-to-Text FastAPI Application

This project provides an API for translating EEG data into text using a trained model. The application is built using FastAPI and can be deployed both locally and within a Docker container.

### Example curl for testing purposes

curl -X POST -F "file=@datasets/saved_data/input_embeddings_1.json" http://127.0.0.1:8000/predict/


### Project Structure

```
fastapi-eeg-app/
│
├── Dockerfile
├── Requirements.txt
│
├── api.py                # FastAPI application and endpoints
├── inference.py          # Functions for model loading and data inference
├── brain_translator_model.py  # Model architecture (BrainTranslator class)
└── eeg_preprocessing.py  # EEG data preprocessing functions
```

### Setup and Local Testing

1. **Environment Setup**:
   Ensure you have Python 3.8 or newer installed. Then, navigate to the project directory and install the necessary dependencies:
   ```bash
   pip install -r Requirements.txt
   ```

2. **Run the FastAPI Application**:
   Start the FastAPI server using the following command:
   ```bash
   uvicorn api:app --reload
   ```
   The application will be accessible at `http://127.0.0.1:8000`.

3. **Test the API**:
   To test the prediction endpoint, use `curl` or any API testing tool like Postman. Here's a `curl` command to test with a sample EEG data file in CSV format:
   ```bash
   curl -X 'POST' -F 'file=@path_to_your_eeg_data.csv' http://127.0.0.1:8000/predict/
   ```

### Using Docker (Optional)

If you prefer to run the application within a Docker container, follow these steps:

1. **Build the Docker Image**:
   Navigate to the project directory and build the Docker image:
   ```bash
   docker build -t fastapi-eeg-app .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 8000:8000 fastapi-eeg-app
   ```

3. **Test the API**:
   Just as in the local testing, use `curl` or any API testing tool to send a request to the `/predict/` endpoint:
   ```bash
   curl -X 'POST' -F 'file=@path_to_your_eeg_data.csv' http://127.0.0.1:8000/predict/
   ```

### Note

Ensure that all necessary files (`api.py`, `inference.py`, `brain_translator_model.py`, `eeg_preprocessing.py`, and any model checkpoint or other necessary files) are in the specified directory structure for the application to function correctly.

---

You can save the above content as `README.md` in your project's root directory. This README provides a clear overview and instructions for users or developers to set up and test the application. If you need any further adjustments or additions, please let me know!