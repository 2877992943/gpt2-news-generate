#-*- coding: UTF-8 -*-



"""https://github.com/huggingface/transformers/tree/master/examples#language-model-training"""



import transformers


import torch
from transformers import  GPT2Tokenizer ,GPT2Config, GPT2LMHeadModel




class GPT2Tokenizer_inherit(GPT2Tokenizer):
    def _tokenize(self, text):
        """ Tokenize a string. """
        return list(text.strip())


tokenizer=GPT2Tokenizer_inherit(vocab_file='./download_gpt2/gpt2-vocab-cn.json',merges_file='./download_gpt2/merge.txt')
tokens = tokenizer.tokenize("这是一个什么")
print ('')
#torch.set_grad_enabled(False)

# Store the model we want to use

MODEL_NAME='gpt2'



print("Tokens: {}".format(tokens))

# This is not sufficient for the model, as it requires integers as input,
# not a problem, let's convert tokens to ids.
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Tokens id: {}".format(tokens_ids))

# Add the required special tokens
tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)

# We need to convert to a Deep Learning framework specific format, let's use PyTorch for now.
tokens_pt = torch.tensor([tokens_ids])
print("Tokens PyTorch: {}".format(tokens_pt))

# Now we're ready to go through BERT with out input
#outputs, pooled = model(tokens_pt)
#print("Tokenw   ise output: {}, Pooled output: {}".format(outputs.shape, pooled.shape))