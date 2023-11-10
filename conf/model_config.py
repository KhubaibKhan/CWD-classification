# %%
import yaml


# yaml file path
yaml_path = "../models/models.yaml"

model_dict = {}

with open(yaml_path, 'r') as f:
    temp_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_dict.update(temp_dict)

# %%