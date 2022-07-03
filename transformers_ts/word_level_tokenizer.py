from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import RobertaTokenizerFast

class KrvTokenizer:
    def save_tokenizer(self, tokenizer, model_max_length, tc = "RobertaTokenizer"):
        stm = '{"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "sep_token": "</s>",\
                "pad_token": "<pad>", "cls_token": "<s>", "mask_token": {"content": "<mask>", "single_word": false,\
                "lstrip": true, "rstrip": false, "normalized": false}}'
        tc = '{"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>", "add_prefix_space": false,\
                "errors": "replace", "sep_token": "</s>", "cls_token": "<s>", "pad_token": "<pad>",\
                "mask_token": "<mask>", "model_max_length": %d, "special_tokens_map_file": null,\
                "name_or_path": "krv_tokenizer", "tokenizer_class": \"%s\"}' % (model_max_length,tc)

        tokenizer.save('krv_tokenizer/tokenizer.json')
        vocab = tokenizer.get_vocab()

        textfile = open("krv_tokenizer/vocab.json", "w")
        textfile.write(str(vocab))
        textfile.close()

        textfile = open("krv_tokenizer/merges.txt", "w")
        textfile.write('#version: 0.2 - Trained by `huggingface/tokenizers`')
        textfile.close()

        textfile = open("krv_tokenizer/special_tokens_map.json", "w")
        textfile.write(stm)
        textfile.close()

        textfile = open("krv_tokenizer/tokenizer_config.json", "w")
        textfile.write(tc)
        textfile.close()
        
    def make_tokenizer(self, model_max_length = 512):
        tokenizer = Tokenizer(WordLevel())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"])
        tokenizer.train(["corpus.txt"], trainer)
        self.save_tokenizer(tokenizer, model_max_length)
        self.tokenizer = RobertaTokenizerFast.from_pretrained('./krv_tokenizer')


    def convert_ids_to_tokens(self, target_indexes):
        return self.tokenizer.convert_ids_to_tokens(target_indexes)
      
    def get_vocab(self):
      	return self.tokenizer.get_vocab()

    def make_corpus(self, ln):
        print(f'The vocabulary size is {ln}')

        s = ''
        for i in range(ln):
            s += str(i) + ' '

        textfile = open("corpus.txt", "w")
        textfile.write(s)
        textfile.close()

class RangeTokenizer(KrvTokenizer):
    def __init__(self, delta, vmin, vmax, model_max_length):
        self.model_max_length = model_max_length
        self.vmin = vmin
        self.vmax = vmax
        self.delta = delta
        super().make_corpus(int((vmax - vmin)/delta))
        super().make_tokenizer(model_max_length)
        self.values = self.convert_tokens_to_vals(self.tokenizer.get_vocab())
        
    def convert_vals_to_tokens(self, vals):
        return [str(round((v - self.vmin)/self.delta)) for v in vals]

    def convert_tokens_to_vals(self, tokens):
        return [int(w)*self.delta + self.vmin if w.isnumeric() else -1000000.0 for w in tokens]