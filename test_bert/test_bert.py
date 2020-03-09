
"""https://github.com/huggingface/transformers/tree/master/examples#language-model-training"""



import transformers


import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer,GPT2Tokenizer

torch.set_grad_enabled(False)

# Store the model we want to use
MODEL_NAME = "bert-base-cased"
MODEL_NAME='bert-base-chinese'
download='./download/'
# We need to create the model and tokenizer
#model = AutoModel.from_pretrained(MODEL_NAME)
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)



tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=download,
                                          cache_dir=download
                                          )

####
# Tokens comes from a process that splits the input into sub-entities with interesting linguistic properties.
tokens = tokenizer.tokenize("这是一个什么")
single_seg_input = tokenizer.encode_plus("这是一个什么")

multi_seg_input = tokenizer.encode_plus("这是一个什么", "那又是一个什么")

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