SPECIAL_TOKENS = {
    't5': {
        'eos': '</s>',
    },
    'xlm-r': {
        'bos': '<s>',
        'eos': '</s>',
    },
    'mistral': {
        'bos': '<s>',
        'eos': '</s>',
    },
    'llama': {
        'bos': '<|begin_of_text|>',
        'eos': '<|end_of_text|>',
        'pad': '<|finetune_right_pad_id|>',
        'mask': "<|reserved_special_token_0|>",
    },
    'nvidia/NV-Embed-v2': {
        'bos': '<s>',
        'eos': '</s>',
    }
}