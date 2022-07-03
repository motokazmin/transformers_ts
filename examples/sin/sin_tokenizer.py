import numpy as np
import random
from transformers_ts.word_level_tokenizer import  RangeTokenizer

class SinTokenizer(RangeTokenizer):
    def __init__(self, delta = 0.01, vmin = -1.0, vmax = 1.0, model_max_length = 512):
        super().__init__(delta, vmin, vmax, model_max_length)
            
    def make_data_lines(self, num_trains = 10000):
            textfile = open("train_data.txt", "w")

            for _ in range(num_trains):
                r = np.pi*random.uniform(0.0, 1.0)
                vals = np.sin(np.linspace(-np.pi + r, np.pi + r, self.model_max_length))
                textfile.write(' '.join(self.convert_vals_to_tokens(vals)) + '\n')

            textfile.close()
            
    def make_masked_data(self, num_masks):
        r = np.pi*random.uniform(0.0, 1.0)
        vals = np.sin(np.linspace(-np.pi + r, np.pi + r, self.model_max_length))
        tokens = self.convert_vals_to_tokens(vals)
        idx = sorted(random.sample(range(0, len(vals)), num_masks))
        saved_tokens = [tokens[i] for i in idx]

        tokens_with_mask = [token if i not in(idx) else '<mask>' for i, token in zip(range(0, len(tokens)), tokens)]

        return ' '.join(tokens_with_mask), ' '.join(tokens), vals[idx]
      
    def get_valid_interval(self):
      return 2.0