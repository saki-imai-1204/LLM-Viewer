def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
    return getattr(model_params, "num_attention_heads")

def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]

def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")


def get_intermediate_size(model_params):
    return 6400
    # because intermediate_size=4×n_embd=4×1600=6400


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")

def post_process(model_params,args):
    hiddensize=get_hidden_size(model_params)
    vocab_size=get_vocab_size(model_params)
    layers=[]
    for stage in ["prefill", "decode"]:
        layers.append({
            'name': 'lm_head',
            'stage':stage,
            'OPs':args['batchsize']*hiddensize*vocab_size*1,
            'load_weight':hiddensize*vocab_size *args['w_byte'],
            'load_act':hiddensize*args['a_byte'],
            'store_act':vocab_size*args['a_byte'],
        })
    return layers

def get_linear_layers(model_params):
    hidden_size = get_hidden_size(model_params)
    intermediate_size = get_intermediate_size(model_params)
    key_value_heads = get_num_key_value_heads(model_params)
    attention_heads = get_num_attention_heads(model_params)
    return {
        "q_proj": [hidden_size, hidden_size],
        "k_proj": [hidden_size, hidden_size * key_value_heads / attention_heads],
        "v_proj": [hidden_size, hidden_size * key_value_heads / attention_heads],
        "out_proj": [hidden_size, hidden_size],
        "gate_proj": [hidden_size, intermediate_size],
        "up_proj": [hidden_size, intermediate_size],
        "down_proj": [intermediate_size, hidden_size],
    }


transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "qk_matmul": ["q_proj", "k_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
    "out_proj": ["sv_matmul"],
    "attn_add": ["input", "out_proj"],
    "mlp_norm": ["attn_add"],
    "up_proj": ["mlp_norm"],
    "mlp_act": ["up_proj"],
    "down_proj": ["mlp_act"],
    "mlp_add": ["attn_add", "down_proj"],
    "output": ["mlp_add"],
}