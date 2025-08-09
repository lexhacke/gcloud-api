import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

class DirectPipeline:
  """
  DirectPipeline is a wrapper class for model and tokeniser to allow for streamlined inference
  """
  def __init__(self, 
               model=AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2"),
               tokenizer=AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")):
    self.model = model
    self.tokenizer = tokenizer
    self.max_tokens = 512
    for param in self.model.parameters():
      param.requires_grad = False

  def __call__(self, question, context, no_none=False):
    """
    no_none can be set to True if an answer should be forced, i.e. a None (No Answer) return is blocked
    """
    x = self.tokenizer(question,
                  context,
                  truncation=True,
                  padding='max_length',
                  max_length=self.max_tokens,
                  return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
    raw_output = self.model(**x)
    start_logits = raw_output['start_logits'][0].cpu()
    end_logits = raw_output['end_logits'][0].cpu()
    span = (start_logits.argmax().numpy(),end_logits.argmax().numpy())
    if span[1] == 0 and no_none:
      print("Generating alternative answer")
      span = (start_logits[1:].argmax().numpy(),end_logits[1:].argmax().numpy())
    answer = self.tokenizer.decode(x['input_ids'][0][span[0]:span[1]+1]) if span[1] > 0 else None #tokenizer.decode(x['input_ids'][0][alternate_answer[0]:alternate_answer[1]+1])
    return answer