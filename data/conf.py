
DATASET_CONFIGS = {
    'thyroid': {
        'data_dim': 6,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 5,
        'learning_rate': 0.05,
        'batch_size': 256,
        'epochs': 200,
    },

    'mammography': {
        'data_dim': 6,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 5,
        'learning_rate': 0.05,
        'batch_size': 512,
        'epochs': 200,
    },

    'pima': {
        'data_dim': 8,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 6,
        'learning_rate': 0.05,
        'batch_size': 256,
        'epochs': 200,
    },

    'breastw': {
        'data_dim': 9,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 6,
        'learning_rate': 0.05,
        'batch_size': 256,
        'epochs': 200,
    },

    'shuttle': {
        'data_dim': 9,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 6,
        'learning_rate': 0.05,
        'batch_size': 512,
        'epochs': 200,
    },

    'glass': {
        'data_dim': 7,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 6,
        'learning_rate': 0.05,
        'batch_size': 128,
        'epochs': 200,
    },

    'wine': {
        'data_dim': 13,
        'hidden_dim': 96,
        'z_dim': 48,
        'mask_num': 8,
        'learning_rate': 0.05,
        'batch_size': 256,
        'epochs': 200,
    },

    'pendigits': {
        'data_dim': 16,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 10,
        'learning_rate': 0.05,
        'batch_size': 512,
        'epochs': 200,
    },

    'cardio': {
        'data_dim': 21,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 10,
        'learning_rate': 0.05,
        'batch_size': 512,
        'epochs': 200,
    },

    'cardiotocography': {
        'data_dim': 21,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 10,
        'learning_rate': 0.05,
        'batch_size': 512,
        'epochs': 200,
    },

    'fraud': {
        'data_dim': 29,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 12,
        'learning_rate': 0.05,
        'batch_size': 512,
        'epochs': 200,
    },

    'wbc': {
        'data_dim': 30,
        'hidden_dim': 128,  # 256'dan düşürüldü
        'z_dim': 64,        # 128'den düşürüldü
        'mask_num': 15,
        'learning_rate': 0.05,
        'batch_size': 512,
        'epochs': 200,
    },

    'ionosphere': {
        'data_dim': 32,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 12,     # 10'dan artırıldı
        'learning_rate': 0.05,
        'batch_size': 256,
        'epochs': 200,
    },

    'satellite': {
        'data_dim': 36,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 12,
        'learning_rate': 0.05,
        'batch_size': 512,
        'epochs': 200,
    },

    'satimage-2': {
        'data_dim': 36,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 12,
        'learning_rate': 0.05,
        'batch_size': 512,
        'epochs': 200,
    },

    'campaign': {
        'data_dim': 62,
        'hidden_dim': 256,
        'z_dim': 128,
        'mask_num': 20,
        'learning_rate': 0.03,
        'batch_size': 512,
        'epochs': 150,
    },

    'optdigits': {
        'data_dim': 64,
        'hidden_dim': 256,
        'z_dim': 128,
        'mask_num': 20,
        'learning_rate': 0.03,
        'batch_size': 512,
        'epochs': 150,
    },

    'nslkdd': {
        'data_dim': 122,
        'hidden_dim': 512,
        'z_dim': 256,
        'mask_num': 30,
        'learning_rate': 0.01,
        'batch_size': 512,
        'epochs': 150,
    },

    'arrhythmia': {
        'data_dim': 274,
        'hidden_dim': 512,
        'z_dim': 256,
        'mask_num': 40,
        'learning_rate': 0.01,
        'batch_size': 512,
        'epochs': 150,
    },

    'census': {
        'data_dim': 299,
        'hidden_dim': 512,
        'z_dim': 256,
        'mask_num': 50,
        'learning_rate': 0.01,
        'batch_size': 512,
        'epochs': 150,
    },
}

DATASET_CATEGORIES = {
    'small': ['thyroid', 'mammography', 'pima', 'breastw', 'shuttle', 'glass', 'wine', 'pendigits'],
    'medium': ['cardio', 'cardiotocography', 'fraud', 'wbc', 'ionosphere', 'satellite', 'satimage-2', 'campaign',
               'optdigits'],
    'large': ['census'],
}
DATASET_CATEGORIES['all'] = (
        DATASET_CATEGORIES['small'] +
        DATASET_CATEGORIES['medium'] +
        DATASET_CATEGORIES['large']
)