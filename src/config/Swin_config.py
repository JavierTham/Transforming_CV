config = {
    "learning_rate": 1e-04,
    "epochs": 20,
    "batch_size": 512,
    "num_workers": 4,
    "pretrained": True
}

# for DataLoader
params = {
	'batch_size': config['batch_size'],
	'shuffle': True,
	'num_workers': config['num_workers'],
	'pin_memory': True
}