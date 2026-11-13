import json
import os
import glob

def update_config_with_args(config, args):
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

def load_saved_data(path):
    dirs = os.listdir(path)
    json_files = glob.glob(os.path.join(path, '*.json')) 
    instructions = []
    for json_file in json_files:
        with open(json_file, 'r+') as f:
            for line in f:  
                data = json.loads(line)
                instructions.append(data['instruction'])
                break
    return instructions

def make_json_serializable(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'dtype'):  # numpy types
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return make_json_serializable(obj.tolist())
        else:
            return str(obj)
    else:
        return obj