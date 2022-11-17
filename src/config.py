import transformers

DEVICE = "cuda"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 100
BERT_PATH = "../models/MLM"
MODEL_PATH = "../models/BERTNSP"
TRAINING_FILE = "../input/q_a.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
