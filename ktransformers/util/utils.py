#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import torch
from torch import nn
import itertools
import time
import enum
from ktransformers.util.custom_gguf import translate_name_to_gguf
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.operators import base_operator
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.util.textstream import TextStreamer
from ktransformers.operators.flashinfer_wrapper import MLAWrapperSingleton

warm_uped = False

def get_compute_capability(device:torch.device = None):
    if torch.cuda.is_available():
        if device is None:
            num_gpus = torch.cuda.device_count()
            min_compute_capability_major = 100
            for gpu_id in range(num_gpus):
                gpu_props = torch.cuda.get_device_properties(gpu_id)
                min_compute_capability_major = min(min_compute_capability_major, gpu_props.major)
            return min_compute_capability_major
        else:
            return torch.cuda.get_device_properties(device)

def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        if hasattr(cur_mod, s):
            cur_mod = getattr(cur_mod, s)
        else: # nn.ModuleList or nn.ModuleList
            cur_mod=cur_mod[int(s)]
    if hasattr(cur_mod, tokens[-1]):
        setattr(cur_mod, tokens[-1], module)
    else: # nn.ModuleList or nn.ModuleList
        cur_mod[int(tokens[-1])] = module

def set_param(module: nn.Module, name: str, weights: torch.Tensor):
    
    param=nn.parameter.Parameter(weights, requires_grad=False)
    if isinstance(module, nn.Linear) and len(weights.shape)==1:
        param.unsqueeze_(0)
    setattr(module, name, param)

def get_device(gguf_module_key:str, device_map:dict):
    if gguf_module_key in device_map:
        return device_map[gguf_module_key]["generate_device"]
    else:
        return "cuda"

def get_all_used_cuda_device(device_map:dict):
    all_device_list = set()
    for key in device_map:
        all_device_list.add(device_map[key]["generate_device"]) if "generate_device" in device_map[key] else None
        all_device_list.add(device_map[key]["prefill_device"]) if "prefill_device" in device_map[key] else None
    if "cpu" in all_device_list:
        all_device_list.remove("cpu")
    all_device_list = list(all_device_list)
    return all_device_list

def load_cur_state_dict(module: nn.Module, gguf_loader: GGUFLoader, prefix: str = ""):
    prefix = prefix.replace("orig_module.", "")
    persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
    local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    for name, param in local_state.items():
        key = prefix + name
        translated_key = translate_name_to_gguf(key)
        
        # TODO: Merge all loader.
        # I know this is ugly but lets do it for now.
        if gguf_loader.safetensor_loader is not None:
            load_dequantized_tensor = gguf_loader.safetensor_loader.load_dequantized_tensor
            tensor_file_map = gguf_loader.safetensor_loader.tensor_file_map
        else:
            load_dequantized_tensor = gguf_loader.load_gguf_tensor
            tensor_file_map = gguf_loader.tensor_file_map
        
        if translated_key in tensor_file_map:
            target_dtype = torch.get_default_dtype()
            device = get_device(translated_key[:translated_key.rfind(".")], gguf_loader.tensor_device_map)
            print(f"loading {translated_key} to {device}")
            torch.cuda.empty_cache()
            weights = load_dequantized_tensor(translated_key, device=device).to(dtype=target_dtype)
            set_param(module, name, weights)
            del weights
        else:
            #print(load_config.tensor_file_map.keys())
            raise Exception(f"can't find {translated_key} in GGUF file!")
        
def load_weights(module:nn.Module, gguf_loader:GGUFLoader, prefix=''):
    #print(f"recursively loading weights {prefix}")
    if not isinstance(module, base_operator.BaseInjectedModule):
        load_cur_state_dict(module, gguf_loader, prefix)
        for name, child in module._modules.items():
            load_weights(child, gguf_loader, prefix+name+".")
    else:
        module.load()

def prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000, use_cuda_graph: bool = True,
                         mode = 'normal', force_think: bool = False, chunk_prefill_size = 16384, use_flashinfer_mla = False,
                         num_heads = None, head_dim_ckv = None, head_dim_kpe = None, q_head_dim = None,
                         do_sample = None, fixed_decode_tokens = None, min_new_tokens = None,
                         temperature = None, top_p = None, top_k = None):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape
    device_map = model.gguf_loader.tensor_device_map
    torch_device = get_device('blk.0.self_attn', device_map)
    torch_device = "cuda:0" if torch_device == "cuda" else torch_device
    inputs = inputs.to(torch_device)
    all_cuda_device = get_all_used_cuda_device(device_map)

    tokens = []
    
    if do_sample is None:
        do_sample = os.environ.get("HYBRIMOE_DO_SAMPLE", "1").strip().lower() in {"1", "true", "yes", "on"}
    if fixed_decode_tokens is None:
        fixed_decode_tokens = os.environ.get("HYBRIMOE_FIXED_DECODE_TOKENS", "0").strip().lower() in {"1", "true", "yes", "on"}
    if min_new_tokens is None:
        min_new_tokens = int(os.environ.get("HYBRIMOE_MIN_NEW_TOKENS", "0"))
    min_new_tokens = max(0, min(int(min_new_tokens), int(max_new_tokens)))
    if temperature is None and os.environ.get("HYBRIMOE_TEMPERATURE"):
        temperature = float(os.environ["HYBRIMOE_TEMPERATURE"])
    if top_p is None and os.environ.get("HYBRIMOE_TOP_P"):
        top_p = float(os.environ["HYBRIMOE_TOP_P"])
    if top_k is None and os.environ.get("HYBRIMOE_TOP_K"):
        top_k = int(os.environ["HYBRIMOE_TOP_K"])
    safe_sampling = os.environ.get("HYBRIMOE_SAFE_SAMPLING", "1").strip().lower() in {"1", "true", "yes", "on"}

    eos_token_ids = set()
    for candidate in (getattr(tokenizer, "eos_token_id", None), getattr(model.generation_config, "eos_token_id", None)):
        if isinstance(candidate, (list, tuple, set)):
            eos_token_ids.update(int(item) for item in candidate if item is not None)
        elif candidate is not None:
            eos_token_ids.add(int(candidate))
    try:
        im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(im_end_ids) == 1:
            eos_token_ids.add(int(im_end_ids[0]))
    except Exception:
        pass

    def is_eos_token(token) -> bool:
        token_id = int(token[0].item()) if hasattr(token, "shape") and token.numel() else int(token)
        if token_id in eos_token_ids:
            return True
        try:
            return tokenizer.decode([token_id]) == "<|im_end|>"
        except Exception:
            return False

    def select_next_token(next_token_scores, generation_config, *, suppress_eos: bool = False):
        scores = next_token_scores
        if safe_sampling:
            scores = torch.nan_to_num(scores.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
        if suppress_eos and eos_token_ids:
            scores = scores.clone()
            vocab_size = scores.shape[-1]
            for eos_id in eos_token_ids:
                if 0 <= eos_id < vocab_size:
                    scores[:, eos_id] = -1e9
        if generation_config.do_sample:
            probs = nn.functional.softmax(scores, dim=-1)
            if safe_sampling:
                probs_ok = (
                    torch.isfinite(probs).all()
                    and (probs >= 0).all()
                    and (probs.sum(dim=-1) > 0).all()
                )
                if not bool(probs_ok.item()):
                    return torch.argmax(scores, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(1)
        return torch.argmax(scores, dim=-1)

    def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph: bool = True, suppress_eos: bool = False):
        if cuda_graph_runner is None:
            use_cuda_graph = False
        if use_cuda_graph:
            logits = cuda_graph_runner(cur_token, position_ids, cache_position)
        else:
            # custom_stream = torch.cuda.Stream()
            torch.cuda.set_device(torch_device)
            inputs_embeds = model.model.embed_tokens(cur_token.to("cpu")).to(torch_device)
            # with torch.cuda.stream(custom_stream):
            logits=model(inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        return_dict=False, use_cache=True)[0]
        if past_key_values != None:
            past_key_values.change_seq_length(1)
        for device in all_cuda_device:
            torch.cuda.synchronize(device)
        #print(logits)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        next_token = select_next_token(next_token_scores, generation_config, suppress_eos=suppress_eos)
        return next_token
    
    # TODO: use CUDA Graph for chunk prefill, may get small improvement
    def chunk_prefill(inputs, cache_position, past_key_values):
        if mode == "long_context":
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu"))
        else:
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu")).to(torch_device)
        if use_flashinfer_mla:
            MLAWrapperSingleton.update_buffer(past_key_values.max_pages)
            MLAWrapperSingleton.need_plan_all()
            
        logits = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
        
        return logits
    
    torch.cuda.set_device(torch_device)
    with torch.no_grad():
        
        stream = TextStreamer(tokenizer)
        if mode != 'long_context':
            past_key_values = StaticCache(
                config = model.config, max_batch_size = 1, max_cache_len = seq_length + max_new_tokens, device = device_map, dtype = model.dtype
            )
        else:
            past_key_values = None
        
        generation_config, model_kwargs = model._prepare_generation_config(
            None, do_sample=bool(do_sample)
            # change this to modify generate config
            #top_k=5, top_p=0.85, temperature=0.1
        )
        generation_config.do_sample = bool(do_sample)
        if temperature is not None:
            generation_config.temperature = float(temperature)
        if top_p is not None:
            generation_config.top_p = float(top_p)
        if top_k is not None:
            generation_config.top_k = int(top_k)
        try: # transformers==4.43
            logits_warper = (
                model._get_logits_warper(generation_config,device=inputs.device)
            )
        except: 
            logits_warper = (
                model._get_logits_warper(generation_config)
            )

        cache_position = torch.arange(seq_length, device=torch_device, dtype=torch.int32)
        generated_ids = torch.zeros(
            batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=torch_device
        )
        generated_ids[:, cache_position] = inputs.to(torch_device).to(torch.int)
        start_time = time.time()

        chunk_start = 0
        while chunk_start < seq_length:
            chunk_end = min(chunk_start + chunk_prefill_size, seq_length)
            if past_key_values != None:
                past_key_values.cur_idx=cache_position[chunk_start:chunk_end]
            logits = chunk_prefill(inputs[:, chunk_start:chunk_end], cache_position[chunk_start:chunk_end], past_key_values)
            chunk_start += chunk_prefill_size

        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        next_token = select_next_token(
            next_token_scores,
            generation_config,
            suppress_eos=bool((fixed_decode_tokens and max_new_tokens > 1) or min_new_tokens > 1),
        )

        first_token_time = time.time() - start_time
        
        if use_flashinfer_mla:
            MLAWrapperSingleton.reset_buffer()

        prefill_count = seq_length
        prefill_time = first_token_time
        if force_think:
            print("<think>")
        print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, seq_length] = next_token
        tokens.append(int(next_token))
        inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([seq_length], device=torch_device, dtype=torch.int32)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1
        
        cuda_graph_runner = None
            
        start_time = time.time()
        for i in range(1, max_new_tokens):
            if use_flashinfer_mla:
                MLAWrapperSingleton.plan_all(None,None,None,position_ids.squeeze(1)+1,
                                             num_heads, head_dim_ckv, head_dim_kpe, past_key_values.page_size,
                                             model.model.layers[0].self_attn.softmax_scale, torch.bfloat16, torch.bfloat16)
            global warm_uped
            if use_cuda_graph and ( (warm_uped == True and int(i) == 1) or (warm_uped == False and int(i) == 2) ):
                warm_uped = True
                cuda_graph_runner = CUDAGraphRunner()
                cuda_graph_runner.capture(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, torch_device, return_dict=False, use_cache=True)
            next_token = decode_one_tokens(
                cuda_graph_runner,
                next_token.unsqueeze(0),
                position_ids,
                cache_position,
                past_key_values,
                logits_warper,
                generation_config,
                use_cuda_graph,
                suppress_eos=bool((fixed_decode_tokens and i < max_new_tokens - 1) or i < min_new_tokens - 1),
            ).to(torch_device)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(int(next_token))
            seq_length += 1
            
            if not fixed_decode_tokens and len(tokens) >= min_new_tokens and is_eos_token(next_token):
                print(stream.end(), end="", flush=True)
                break
            else:
                print(stream.put(next_token.item()), end="", flush=True)
            cache_position += 1
            position_ids = cache_position.unsqueeze(0)
        

    total_time = time.time() - start_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time

    print("")

    print(f"prompt eval count:    {prefill_count} token(s)")
    print(f"prompt eval duration: {prefill_time}s")
    print(f"prompt eval rate:     {prefill_count/prefill_time} tokens/s")
    print(f"eval count:           {tokens_generated} token(s)")
    print(f"eval duration:        {total_time}s")
    print(f"eval rate:            {tokens_per_second} tokens/s")

    return tokens, prefill_time, tokens_generated, total_time

class InferenceState(enum.Enum):
    UNLOAD = 0
    PREFILL = 1
    GENERATE = 2
    RESTORE = 3
