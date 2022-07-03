from transformers import Trainer, TrainingArguments
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
import torch
from transformers_ts.loss_func import KrvCrossEntropyLossEx, KrvCrossEntropyLoss

class KrvTrainer(Trainer):
    def __init__(self, **kwargs):
        self.tok = kwargs.pop('tok', None)
        self.device = kwargs.pop('device', None)
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        logits = model(**inputs).logits
        loss = KrvCrossEntropyLossEx(logits.view(-1, logits.shape[2]), labels.view(-1), self.tok, self.device)
        
        return loss        
      
def prepare_trainer(vocab_size, tok, num_train_epochs=500, save_steps = 10000, from_pretrained = None):
  config = RobertaConfig(
      vocab_size=len(tok.get_vocab()),
      max_position_embeddings=tok.model_max_length + 2,
      num_attention_heads=6,
      num_hidden_layers=3,
      type_vocab_size=2,
      attention_probs_dropout_prob = 0.1,
  )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if from_pretrained == None:
  	model = RobertaForMaskedLM(config=config).to(device)
  else:
    model = RobertaForMaskedLM.from_pretrained(from_pretrained).to(device)

  data_collator = DataCollatorForLanguageModeling(
      tokenizer=tok.tokenizer, mlm=True, mlm_probability=0.15
  )

  dataset = LineByLineTextDataset(
      tokenizer=tok.tokenizer,
      file_path="./train_data.txt",
      block_size=tok.model_max_length,
  )

  training_args = TrainingArguments(
      output_dir="./roberta_ts",
      overwrite_output_dir=True,
      num_train_epochs=num_train_epochs,
      per_gpu_train_batch_size=40,
      save_steps=save_steps,
      save_total_limit=20,
      prediction_loss_only=True,
      #learning_rate=2e-4,
      greater_is_better=False
  )

  trainer = KrvTrainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=dataset,
      tok = tok,
      device = device
  )
  
  return trainer