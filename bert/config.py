import transformers

DEVICE = "cuda"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 100
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "model/model.bin"
TRAINING_FILE = "../qa.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
