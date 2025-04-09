# inference_models/base.py
class BaseModel:
    def predict(self, frame):
        raise NotImplementedError("Must implement predict()")
