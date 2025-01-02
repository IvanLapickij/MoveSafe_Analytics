import os
from inference import get_model

# Replace 'YOUR_API_KEY_HERE' with your actual API key or retrieve it from an environment variable
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', 'tvZVhjN9hMWkURbVo84w')
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

# Print to confirm the model is loaded successfully
print("Model loaded:", PLAYER_DETECTION_MODEL)
