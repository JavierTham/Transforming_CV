config = {
    "learning_rate": 1e-03,
    "epochs": 10,
    "batch_size": 16,
    "sequence_len": 50,
    "num_workers": 4,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 1, 
    "drop out:": True
}

params = {
	'batch_size': config['batch_size'],
	'shuffle': True,
	'num_workers': config['num_workers'],
	'pin_memory': True
}