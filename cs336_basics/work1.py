# Byte-pair encoding (BPE) tokenizer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests')))

from test_train_bpe import test_train_bpe


if __name__ == "__main__":
    test_train_bpe()
