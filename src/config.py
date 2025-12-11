# Model Settings
TEACHER_MODEL = "bert-base-uncased"
STUDENT_MODEL = "bert-base-uncased"
MAX_SEQ_LEN = 64
POOLING = "mean"

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
EPOCHS = 5
PATIENCE = 5
SEED = 42

# QR-EOS Specifics
QR_K = 16
LAMBDA_COS = 1.0
LAMBDA_QR = 1.0
LAMBDA_MSE = 1.0

# Dataset
TRAIN_LIMIT = 10000
EVAL_LIMIT = 1500
