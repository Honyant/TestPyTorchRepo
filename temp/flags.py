import os
FLAGS = {}
FLAGS['data_dir'] = "/tmp/cifar"
FLAGS['batch_size'] = 128
FLAGS['num_workers'] = 4
FLAGS['learning_rate'] = 0.02
FLAGS['momentum'] = 0.9
FLAGS['num_epochs'] = 200
FLAGS['num_cores'] = 8 if os.environ.get('TPU_NAME', None) else 1
FLAGS['log_steps'] = 20
FLAGS['metrics_debug'] = False