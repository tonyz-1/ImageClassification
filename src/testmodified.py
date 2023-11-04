import functions
import json

#Config dict
config_file = "./modelinit.json"
with open(config_file) as handle:
    config = json.loads(handle.read())

#Model init
model_config = config["model_modified"]
encoder_file = model_config["encoder"]
device = model_config["device"]
model = functions.init_modified_model(encoder_file, device)


#Dataloader
hp_config = config["hyperparameters"]
batch_size = hp_config["batch_size"]
augmented = hp_config["augmented"]
dataset = hp_config["dataset"]
train_dl, test_dl = functions.init_dataloaders(batch_size, augmented, dataset=dataset)


#Errors
top_1_error = functions.top_k_error(1, model, test_dl, device)
top_5_error = functions.top_k_error(5, model, test_dl, device)

#Output
print(f"Top 1 error: {top_1_error}%")
print(f"Top 5 error: {top_5_error}%")