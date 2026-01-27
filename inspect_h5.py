import h5py
import json

print("--- Inspecting emotion_model.h5 ---")
try:
    with h5py.File("emotion_model.h5", "r") as f:
        if 'model_config' in f.attrs:
            print("\n--- Model Architecture ---")
            config_str = f.attrs['model_config']
            config = json.loads(config_str)
            
            # Navigate to layers
            layers = []
            if 'config' in config and 'layers' in config['config']:
                layers = config['config']['layers']
            elif 'layers' in config:
                layers = config['layers'] # Older Keras format
            
            for i, layer in enumerate(layers):
                ln = layer['class_name']
                cfg = layer['config']
                details = ""
                if ln == 'Conv2D':
                     details = f"filters={cfg.get('filters')}, kernel={cfg.get('kernel_size')}, activation={cfg.get('activation')}"
                elif ln == 'Dropout':
                     details = f"rate={cfg.get('rate')}"
                elif ln == 'Dense':
                     details = f"units={cfg.get('units')}, activation={cfg.get('activation')}"
                
                print(f"{i}: {ln} | {details}")
        else:
             print("No model_config attribute found on root.")

except Exception as e:
    print(f"Error reading h5: {e}")
