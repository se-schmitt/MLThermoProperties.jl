import torch
import json
import os

def convert_state_dict_to_json(bin_path, json_path):
    """
    Loads a PyTorch state_dict from a .bin file and saves it to a .json file.
    Tensors are converted to lists.
    """
    if not os.path.exists(bin_path):
        print(f"Error: File {bin_path} not found.")
        return

    print(f"Loading {bin_path}...")
    # Load the state_dict (map_location='cpu' ensures it works even if saved on GPU)
    try:
        state_dict = torch.load(bin_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"torch.load failed: {e}. Trying json.load...")
        try:
            with open(bin_path, 'r') as f:
                state_dict = json.load(f)
        except Exception as e_json:
            print(f"json.load also failed: {e_json}")
            return
    
    # If the file contains a full model instead of just state_dict, extract state_dict
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    
    json_dict = {}
    
    print("Converting tensors to lists...")
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Convert tensor to numpy array, then to list
            json_dict[key] = value.cpu().numpy().tolist()
        else:
            json_dict[key] = value
            
    print(f"Saving to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=4) 
    
    print("Done.")

# Example usage:
if __name__ == "__main__":
    # Adjust paths as needed
    convert_state_dict_to_json("pytorch_model.bin", "pytorch_model.json")