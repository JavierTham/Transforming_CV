config = {
    "learning_rate": 1e-04,
    "epochs": 20,
    "batch_size": 32,
    "num_workers": 4,
    # "depth": 6, 
    # "heads": 6,
    # "dim": 512,
    # "mlp_dim": 256,
    "pretrained": True
}

# for DataLoader
params = {
	'batch_size': config['batch_size'],
	'shuffle': True,
	'num_workers': config['num_workers'],
	'pin_memory': True
}