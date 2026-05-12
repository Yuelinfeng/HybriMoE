"""
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import os
import platform
import sys
import argparse
import time
from pathlib import Path

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
import fire
from ktransformers.operators.experts import KExpertsCache
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import prefill_and_generate, get_compute_capability
from ktransformers.util.trace_context import trace_context
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from ktransformers.util.vendors import device_manager, get_device, to_device, GPUVendor

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
)
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}


def _load_prompt_stream(prompt_stream: str | None, prompt_limit: int = 0) -> list[dict]:
    if prompt_stream is None:
        return []
    records = []
    with open(prompt_stream, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            payload = json.loads(line)
            content = payload.get("prompt") or payload.get("content")
            if not content:
                raise ValueError(f"Prompt stream record {idx} has no prompt/content field.")
            payload.setdefault("prompt_id", f"prompt_{idx:06d}")
            payload.setdefault("sequence_index", idx)
            records.append(payload)
            if prompt_limit and len(records) >= prompt_limit:
                break
    if not records:
        raise ValueError(f"Prompt stream contains no usable records: {prompt_stream}")
    return records


def _build_input_tensor(tokenizer, content: str, force_think: bool):
    messages = [{"role": "user", "content": content}]
    input_tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if force_think:
        token_thinks = torch.tensor(
            [tokenizer.encode("<think>\\n", add_special_tokens=False)],
            device=input_tensor.device,
        )
        input_tensor = torch.cat([input_tensor, token_thinks], dim=1)
    return input_tensor


def _run_generation(
    *,
    model,
    tokenizer,
    config,
    input_tensor,
    max_new_tokens: int,
    use_cuda_graph: bool,
    mode: str,
    force_think: bool,
    chunk_prefill_size: int,
    do_sample: bool | None,
    fixed_decode_tokens: bool | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
):
    if mode == "long_context":
        assert Config().long_context_config["max_seq_len"] > input_tensor.shape[1] + max_new_tokens, (
            "please change max_seq_len in  ~/.ktransformers/config.yaml"
        )

    if (
        platform.system() != "Windows"
        and (config.architectures[0] == "DeepseekV2ForCausalLM" or config.architectures[0] == "DeepseekV3ForCausalLM")
        and flashinfer_enabled
        and get_compute_capability() >= 8
        and device_manager.gpu_vendor == GPUVendor.NVIDIA
    ):
        return prefill_and_generate(
            model,
            tokenizer,
            input_tensor.cuda(),
            max_new_tokens,
            use_cuda_graph,
            mode=mode,
            force_think=force_think,
            chunk_prefill_size=chunk_prefill_size,
            use_flashinfer_mla=True,
            num_heads=config.num_attention_heads,
            head_dim_ckv=config.kv_lora_rank,
            head_dim_kpe=config.qk_rope_head_dim,
            q_head_dim=config.qk_rope_head_dim + config.qk_nope_head_dim,
            do_sample=do_sample,
            fixed_decode_tokens=fixed_decode_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    return prefill_and_generate(
        model,
        tokenizer,
        input_tensor.cuda(),
        max_new_tokens,
        use_cuda_graph,
        mode=mode,
        force_think=force_think,
        chunk_prefill_size=chunk_prefill_size,
        do_sample=do_sample,
        fixed_decode_tokens=fixed_decode_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def _stream_metadata(
    record: dict,
    index: int,
    input_tokens: int,
    max_new_tokens: int,
    *,
    do_sample: bool | None,
    fixed_decode_tokens: bool | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> dict:
    return {
        "prompt_id": record.get("prompt_id", f"prompt_{index:06d}"),
        "prompt_index": index,
        "prompt_category": record.get("category", "unknown"),
        "workload": record.get("workload", "single"),
        "workload_phase": record.get("phase", record.get("workload_phase", "single")),
        "phase_index": record.get("phase_index"),
        "stream_id": record.get("stream_id"),
        "seed": record.get("seed"),
        "input_tokens": input_tokens,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "fixed_decode_tokens": fixed_decode_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }


def _set_generation_seed(record: dict, index: int) -> None:
    raw_seed = record.get("generation_seed", record.get("seed"))
    if raw_seed is None:
        return
    seed = int(raw_seed) * 1000003 + int(index)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def local_chat(
    model_path: str | None = None,
    optimize_config_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 1000,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = False,
    prompt_file : str | None = None,
    mode: str = "normal",
    force_think: bool = False,
    chunk_prefill_size: int = 8192,
    load_size: int = None,
    prefetch_size: int = 0,
    prompt_stream: str | None = None,
    prompt_limit: int = 0,
    stream_output: str | None = None,
    do_sample: bool | None = None,
    fixed_decode_tokens: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    override_stream_max_new_tokens: bool = False,
):

    torch.set_grad_enabled(False)

    Config().cpu_infer = cpu_infer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if (
                "Qwen2Moe" in config.architectures[0]
            ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"

            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )

    if load_size is None:
        load_size = input("please input the load size:")
    load_size = [load_size] * config.num_hidden_layers
    config.load_size = load_size
    config.prefetch_size = prefetch_size
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
    
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"generation config can't auto create, make default. Message: {e}")
        gen_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            do_sample=True
        )
        model.generation_config = gen_config
    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    logging.basicConfig(level=logging.INFO)

    system = platform.system()
    if system == "Windows":
        os.system("cls")
    else:
        os.system("clear")

    # while True:
    # content = "请详细阐述广义相对论中的时空弯曲概念,并结合爱因斯坦场方程解释质量如何影响时空几何结构,进一步分析黑洞的形成机制及其边界事件视界的物理意义,同时探讨引力波的存在及其在LIGO实验中的探测原理,最后讨论宇宙学中的暗物质与暗能量问题,解释它们如何通过引力效应影响宇宙大尺度结构的演化,并分析当前宇宙加速膨胀现象与爱因斯坦最初引入的宇宙学常数之间的关系,以及现代观测数据对标准宇宙学模型的验证与挑战"
    prompt_records = _load_prompt_stream(prompt_stream, prompt_limit)
    if prompt_records:
        output_fh = None
        if stream_output:
            Path(stream_output).parent.mkdir(parents=True, exist_ok=True)
            output_fh = open(stream_output, "a", encoding="utf-8")
        try:
            for index, record in enumerate(prompt_records):
                content = str(record.get("prompt") or record.get("content"))
                prompt_max_new_tokens = int(
                    max_new_tokens if override_stream_max_new_tokens else record.get("max_new_tokens", max_new_tokens)
                )
                input_tensor = _build_input_tensor(tokenizer, content, force_think)
                metadata = _stream_metadata(
                    record,
                    index,
                    int(input_tensor.shape[1]),
                    prompt_max_new_tokens,
                    do_sample=do_sample,
                    fixed_decode_tokens=fixed_decode_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                print(
                    f"\n[hybrimoe-stream] {index + 1}/{len(prompt_records)} "
                    f"prompt_id={metadata['prompt_id']} category={metadata['prompt_category']} "
                    f"workload={metadata['workload']} phase={metadata['workload_phase']}",
                    flush=True,
                )
                wall_start = time.time()
                _set_generation_seed(record, index)
                with trace_context(metadata):
                    tokens, prefill_time, tokens_generated, decode_time = _run_generation(
                        model=model,
                        tokenizer=tokenizer,
                        config=config,
                        input_tensor=input_tensor,
                        max_new_tokens=prompt_max_new_tokens,
                        use_cuda_graph=use_cuda_graph,
                        mode=mode,
                        force_think=force_think,
                        chunk_prefill_size=chunk_prefill_size,
                        do_sample=do_sample,
                        fixed_decode_tokens=fixed_decode_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                if output_fh:
                    output_fh.write(
                        json.dumps(
                            {
                                **metadata,
                                "wall_time_s": time.time() - wall_start,
                                "prefill_time_s": prefill_time,
                                "decode_time_s": decode_time,
                                "tokens_generated": tokens_generated,
                                "generated_text": tokenizer.decode(tokens, skip_special_tokens=True),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    output_fh.flush()
        finally:
            if output_fh is not None:
                output_fh.close()
        return

    content = open(prompt_file, "r", encoding="utf-8").read() if prompt_file else "hi"
    if os.path.isfile(content):
        content = open(content, "r", encoding="utf-8").read()

    input_tensor = _build_input_tensor(tokenizer, content, force_think)
    metadata = {
        "prompt_id": "single_prompt",
        "prompt_index": 0,
        "prompt_category": "single",
        "workload": "single",
        "workload_phase": "single",
        "input_tokens": int(input_tensor.shape[1]),
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "fixed_decode_tokens": fixed_decode_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    with trace_context(metadata):
        _run_generation(
            model=model,
            tokenizer=tokenizer,
            config=config,
            input_tensor=input_tensor,
            max_new_tokens=max_new_tokens,
            use_cuda_graph=use_cuda_graph,
            mode=mode,
            force_think=force_think,
            chunk_prefill_size=chunk_prefill_size,
            do_sample=do_sample,
            fixed_decode_tokens=fixed_decode_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--optimize_config_path", type=str, default=None)
    parser.add_argument("--gguf_path", type=str, default=None)
    parser.add_argument("--cache_size", type=int, default=None)
    parser.add_argument("--prefetch_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--prompt_stream", type=str, default=None)
    parser.add_argument("--prompt_limit", type=int, default=0)
    parser.add_argument("--stream_output", type=str, default=None)
    parser.add_argument("--do_sample", type=int, choices=[0, 1], default=None)
    parser.add_argument("--fixed_decode_tokens", type=int, choices=[0, 1], default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--override_stream_max_new_tokens", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    local_chat(
        model_path=args.model_path,
        optimize_config_path=args.optimize_config_path,
        gguf_path=args.gguf_path,
        max_new_tokens=args.max_new_tokens,
        prompt_file=args.prompt_file,
        load_size=args.cache_size,
        prefetch_size=args.prefetch_size,
        prompt_stream=args.prompt_stream,
        prompt_limit=args.prompt_limit,
        stream_output=args.stream_output,
        do_sample=None if args.do_sample is None else bool(args.do_sample),
        fixed_decode_tokens=None if args.fixed_decode_tokens is None else bool(args.fixed_decode_tokens),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        override_stream_max_new_tokens=bool(args.override_stream_max_new_tokens),
    )
