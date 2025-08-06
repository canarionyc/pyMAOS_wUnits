# Function to read scaling parameters from configuration file
def get_scaling_from_config(config_file_path):
    # Default scaling values
    default_scaling = {
        "axial_load": 100,
        "normal_load": 100,
        "point_load": 1,
        "axial": 2,
        "shear": 2,
        "moment": 0.1,
        "rotation": 5000,
        "displacement": 100,
    }
    import json
    try:
        with open(config_file_path, 'r') as f:
            config_scaling = json.load(f)
            print(f"Loaded scaling configuration from {config_file_path}")

            # Update default scaling with values from config file
            for key, value in config_scaling.items():
                if key in default_scaling:
                    default_scaling[key] = value
                else:
                    print(f"Warning: Unknown scaling parameter '{key}' in config file")
    except FileNotFoundError:
        print(f"Scaling configuration file not found: {config_file_path}")
        print("Using default scaling values")
    except json.JSONDecodeError:
        print(f"Error parsing scaling configuration file: {config_file_path}")
        print("Using default scaling values")
    except Exception as e:
        print(f"Error reading scaling configuration: {str(e)}")
        print("Using default scaling values")

    return default_scaling