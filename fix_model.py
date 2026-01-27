import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
import json

print(f"TensorFlow Version: {tf.__version__}")

# 1. Define Compatibility Layer
InputLayer = tf.keras.layers.InputLayer
class FixedInputLayer(InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('batch_shape', None)
        config.pop('optional', None)
        return cls(**config)

# 2. Try Loading
try:
    print("Attempting load with custom objects...")
    model = load_model("emotion_model.h5", compile=False, custom_objects={'InputLayer': FixedInputLayer})
    print("SUCCESS! Model loaded in memory.")
    
    # 3. Save as new compatible file
    print("Saving compatible version as 'compatible_model.h5'...")
    model.save("compatible_model.h5")
    print("Done.")
    
except Exception as e:
    print("FAILED:")
    print(e)
