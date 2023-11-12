# %%
import yaml


# yaml file path
yaml_path = "models/models.yaml"

model_dict = {}

with open(yaml_path, 'r') as f:
    temp_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_dict.update(temp_dict)

# convert keys to values and vice versa
model_dict = {v: k for k, v in model_dict.items()}
# %%