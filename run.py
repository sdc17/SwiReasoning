import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from generation_utils import (
    set_seed,
    get_math_symbols_ids,
    generate_cot,
    generate_swir,
)
from grader import answer_match
import concurrent.futures
import multiprocessing


def grade_batch_task(batch_data):
    dataset_name = batch_data["dataset_name"]
    questions = batch_data["questions"]
    golds = batch_data["golds"]
    preds = batch_data["preds"]
    generated_ids_list = batch_data["generated_ids_list"]
    prompt_len = batch_data["prompt_len"]
    tokenizer = batch_data["tokenizer"]
    model_name = batch_data["model_name"]
    eot_id = batch_data["eot_id"]
    
    grading_inputs = []
    batch_details_skeleton = []
    
    for idx in range(len(questions)):
        gold = golds[idx]
        question = questions[idx]
        pred = preds[idx]
        
        output_ids = generated_ids_list[idx][prompt_len:]
        try:
            if eot_id in output_ids:
                index = len(output_ids) - output_ids[::-1].index(eot_id)
            else:
                index = 0
        except ValueError:
            index = 0
            
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        answer_content = pred[len(thinking_content):]
        
        grading_inputs.append((dataset_name, answer_content, gold))
        
        gold_to_save = {k: v for k, v in gold.items() if k != "verification_info"} if isinstance(gold, dict) else gold
        batch_details_skeleton.append({
            "question": question,
            "gold": gold_to_save,
            "thinking": thinking_content,
            "answer_content": answer_content,
            "full_pred": pred 
        })

    
    grading_results = []
    PER_TASK_TIMEOUT = 60
    try:
        with multiprocessing.Pool(processes=8, maxtasksperchild=1) as pool:
            async_results = []
            for args in grading_inputs:
                res = pool.apply_async(answer_match, args)
                async_results.append(res)
            
            for i, res in enumerate(async_results):
                try:
                    result = res.get(timeout=PER_TASK_TIMEOUT)
                    grading_results.append(result)
                except multiprocessing.TimeoutError:
                    print(f"  [WARNING] Task {i} timed out (>{PER_TASK_TIMEOUT}s). Skipping.")
                    grading_results.append((False, "Timeout"))
                except Exception as e:
                    print(f"  [WARNING] Task {i} error: {e}")
                    grading_results.append((False, f"Error: {str(e)}"))

    except Exception as e:
        print(f"Pool critical failure: {e}")
        remaining = len(questions) - len(grading_results)
        grading_results.extend([(False, "Pool Crash")] * remaining)

    batch_correct = 0
    batch_total = 0
    batch_details = []
    batch_total_lens = []
    batch_correct_lens = []
    batch_wrong_lens = []

    if len(grading_results) < len(questions): ###
        grading_results.extend([(False, "Missing")] * (len(questions) - len(grading_results)))
    
    for idx, (is_correct, prediction) in enumerate(grading_results):
        skel = batch_details_skeleton[idx]
        
        batch_correct += int(is_correct)
        batch_total += 1
        
        batch_details.append({
            "question": skel["question"],
            "gold": skel["gold"],
            "prediction": prediction,
            "correct": is_correct,
            "thinking": skel["thinking"],
            "answer_content": skel["answer_content"],
        })
        
        output_token_ids = tokenizer.encode(skel["full_pred"], add_special_tokens=False)
        t_len = len(output_token_ids)
        batch_total_lens.append(t_len)
        if is_correct:
            batch_correct_lens.append(t_len)
        else:
            batch_wrong_lens.append(t_len)
            
    return {
        "correct": batch_correct,
        "total": batch_total,
        "details": batch_details,
        "total_lens": batch_total_lens,
        "correct_lens": batch_correct_lens,
        "wrong_lens": batch_wrong_lens
    }


def main(args):
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    model_name = args.model_name
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    n_samples = args.n_samples
    method = args.method
    alpha = args.alpha
    max_switch_count = args.max_switch_count

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    eot_id = tokenizer.convert_tokens_to_ids("</think>")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        eot_ids = tokenizer.encode("</think>", add_special_tokens=False)
        if len(eot_ids) != 1:
            raise ValueError(f"Cannot resolve </think> to a single token id: {eot_ids}")
        eot_id = eot_ids[0]

    is_async_mode = (dataset_name == "livecodebench") ### Coding only
    if is_async_mode:
        eval_tokenizer = AutoTokenizer.from_pretrained(model_name)
        eval_tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"": local_rank}
    )
    
    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
    elif dataset_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    elif dataset_name == "aime_2024":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    elif dataset_name == "aime_2025":
        dataset = load_dataset("yentinglin/aime_2025", split="train")
    elif dataset_name == "gpqa_diamond":
        dataset = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
    elif dataset_name == "2wikimqa":
        dataset = load_dataset("THUDM/LongBench", "2wikimqa", split="test", trust_remote_code=True)
    elif dataset_name == "commonsenseqa":
        dataset = load_dataset("tau/commonsense_qa", "default", split="validation")
    elif dataset_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
    elif dataset_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp","sanitized", split="test")
    elif dataset_name == "leetcode_contest":
        dataset = load_dataset("TechxGenus/LeetCode-Contest", split="train")
    elif dataset_name == "livecodebench":
        dataset = load_dataset("PrimeIntellect/LiveCodeBench-v5", split="train")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))
    total_len = len(dataset)
    chunk_size = (total_len + world_size - 1) // world_size
    start = local_rank * chunk_size
    end = min(start + chunk_size, total_len)
    dataset = dataset.select(range(start, end))
    
    correct = 0
    total = 0
    details = []
    total_token_lens = []
    correct_token_lens = []
    wrong_token_lens = []

    math_symbols_ids = get_math_symbols_ids(tokenizer)
    math_ids_tensor = torch.tensor(list(math_symbols_ids), device=model.device)
    
    thread_executor = None ### Coding only
    futures = []
    if is_async_mode:
        thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        print(f"[Rank {local_rank}] Async grading pipeline ENABLED for {dataset_name}.")

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        if args.dataset_name == "gsm8k":
            questions = batch["question"]
            golds = [str(a).split("####")[-1].strip() for a in batch["answer"]]
        elif args.dataset_name == "math500":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "aime_2024":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "aime_2025":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "gpqa_diamond":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["solution"]]
        elif args.dataset_name == "2wikimqa":
            questions = [
                f"Question: {q}\n\nContext:\n{ctx}"
                for q, ctx in zip(batch["input"], batch["context"])
            ]
            golds = [ans_list for ans_list in batch["answers"]]
        elif args.dataset_name == "commonsenseqa":
            questions = batch["question"]
            golds = [
                {
                    "id": id_,
                    "question": question,
                    "question_concept": question_concept,
                    "choices": choices, 
                    "answerKey": answer_key,
                }
                for id_, question, question_concept, choices, answer_key in zip(
                    batch["id"],
                    batch["question"],
                    batch["question_concept"],
                    batch["choices"],
                    batch["answerKey"],
                )
            ]
        elif args.dataset_name == "humaneval":
            questions = batch["prompt"]
            golds = [
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "canonical_solution": canonical_solution,
                    "test": test,
                    "entry_point": entry_point,
                }
                for task_id, prompt, canonical_solution, test, entry_point in zip(
                    batch["task_id"],
                    batch["prompt"],
                    batch["canonical_solution"],
                    batch["test"],
                    batch["entry_point"],
                )
            ]
        elif args.dataset_name == "mbpp":
            questions = batch["prompt"]
            golds = [
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "code": code,
                    "test_imports": test_imports,
                    "test_list": test_list,
                }
                for task_id, prompt, code, test_imports, test_list in zip(
                    batch["task_id"],
                    batch["prompt"],
                    batch["code"],
                    batch["test_imports"],
                    batch["test_list"],
                )
            ]
        elif args.dataset_name == "leetcode_contest":
            questions = batch["prompt_sft"]
            golds = [
                {
                    "task_id": task_id,
                    "url": url,
                    "title": title,
                    "meta": meta,
                    "prompt": prompt,          
                    "prompt_sft": prompt_sft,
                    "test": test,
                }
                for task_id, url, title, meta, prompt, prompt_sft, test in zip(
                    batch["task_id"],
                    batch["url"],
                    batch["title"],
                    batch["meta"],
                    batch["prompt"],
                    batch["prompt_sft"],
                    batch["test"],
                )
            ]
        elif args.dataset_name == "livecodebench":
            questions = batch["prompt"]
            golds = [
                {
                    "problem_id": problem_id,
                    "task_type": task_type,
                    "prompt": prompt,
                    "verification_info": verification_info,
                }
                for problem_id, task_type, prompt, verification_info in zip(
                    batch["problem_id"],
                    batch["task_type"],
                    batch["prompt"],
                    batch["verification_info"],
                )
            ]

        if args.dataset_name == "2wikimqa":
            prompts = []
            for q in questions:
                prompts.append(
                    f"{q}\n\n"
                    "Please answer the question based on the context.\n"
                    f"Please reason step by step, and put your final answer within \\boxed{{}}.\n"
                    "Your final answer should be a short phrase.\n"
                )
        elif args.dataset_name == "commonsenseqa":
            prompts = []
            for q, gold in zip(questions, golds):
                labels = gold["choices"]["label"]
                texts = gold["choices"]["text"]
                option_lines = [f"({lab}) {txt}" for lab, txt in zip(labels, texts)]
                options_str = "\n".join(option_lines)
                prompts.append(
                    "You are an expert at commonsense reasoning.\n"
                    "You will be given a question and five answer choices (A, B, C, D, E).\n"
                    "Please reason step by step, then give only the letter of the correct option as your final answer.\n"
                    f"Put your final answer letter inside \\boxed{{}}.\n\n"
                    f"Question: {q}\n\n"
                    f"Choices:\n{options_str}\n"
                )
        elif args.dataset_name == "humaneval":
            prompts = []
            for q in questions:
                prompts.append(
                    "You are an expert Python programmer.\n"
                    "Complete the following Python function so that it passes the tests.\n"
                    "Please reason step by step.\n"
                    "Write only valid Python code and do not include any explanations.\n\n"
                    f"{q}"
                )
        elif args.dataset_name == "mbpp":
            prompts = []
            for q, gold in zip(questions, golds):
                tests_preview = "\n".join(gold["test_list"])
                prompts.append(
                    "You are an expert Python programmer.\n"
                    "Here is your task:\n"
                    f"{q}\n\n"
                    "Your code should pass the following tests:\n"
                    f"{tests_preview}\n\n"
                    "Please reason step by step.\n"
                    "Write only Python code and do not include any explanations.\n"
                )
        elif args.dataset_name == "leetcode_contest":
            prompts = []
            for q in questions:
                prompts.append(
                    "You are an expert competitive programmer.\n"
                    "Solve the following LeetCode contest problem in Python.\n"
                    "Write a class named `Solution` implementing the required method(s).\n"
                    "Please reason step by step.\n"
                    "Output only valid Python code, without any explanations or backticks.\n\n"
                    f"{q}"
                )
        elif args.dataset_name == "livecodebench":
            prompts = []
            for q in questions:
                prompts.append(
                    q
                    + "\n\n"
                    "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n"
                    "Please reason step by step.\n"
                )

        else:
            prompts = [
                f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                for q in questions
            ]
        messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for messages in messages_batch
        ]
        model_inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
    
        with torch.no_grad():
            if method == "cot":
                # generated_ids = model.generate( 
                #     **model_inputs,
                #     **gen_kwargs,
                # )
                generated_ids = generate_cot( # better memory efficiency 
                    model,
                    tokenizer,
                    **model_inputs,   
                    **gen_kwargs,   
                )
            elif method == "cot_greedy":
                gen_kwargs["do_sample"] = False
                # generated_ids = model.generate( 
                #     **model_inputs,
                #     **gen_kwargs,
                # )
                generated_ids = generate_cot( # better memory efficiency
                    model,
                    tokenizer,
                    **model_inputs,   
                    **gen_kwargs,   
                )
            elif method == "swir":
                model_inputs["alpha_0"] = alpha
                model_inputs["max_switch_count"] = max_switch_count
                model_inputs["math_ids_tensor"] = math_ids_tensor
                model_inputs["convergence_words"] = "</think>" if "Qwen" in model_name else "\n\n</think>\n\n"
                generated_ids = generate_swir(
                    model,
                    tokenizer,
                    **model_inputs,   
                    **gen_kwargs,   
                )
        
        prompt_len = model_inputs["input_ids"].shape[1]

        if is_async_mode: ### Coding only
            generated_ids_cpu_list = generated_ids.cpu().tolist()
            
            current_preds = [
                tokenizer.decode(generated_ids_cpu_list[idx][prompt_len:], skip_special_tokens=True)
                for idx in range(len(questions))
            ]
            
            batch_data = {
                "dataset_name": dataset_name,
                "questions": questions,
                "golds": golds,
                "preds": current_preds,
                "generated_ids_list": generated_ids_cpu_list, 
                "prompt_len": prompt_len,
                "tokenizer": eval_tokenizer, 
                "model_name": model_name,
                "eot_id": eot_id
            }
            future = thread_executor.submit(grade_batch_task, batch_data)
            futures.append(future)
        else:
            preds = [
                tokenizer.decode(generated_ids[idx][prompt_len:], skip_special_tokens=True)
                for idx in range(len(questions))
            ]
        
            for idx in range(len(questions)):
                gold = golds[idx]
                question = questions[idx]
                pred = preds[idx]
                output_ids = generated_ids[idx][prompt_len:].tolist()
                try:
                    index = len(output_ids) - output_ids[::-1].index(eot_id)
                except ValueError:
                    index = 0
                thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
                answer_content = pred[len(thinking_content):]
                is_correct, prediction = answer_match(dataset_name, answer_content, gold)
                correct += int(is_correct)
                total += 1
                if args.dataset_name == "livecodebench": ### Too large to save verification_info
                    gold = {k: v for k, v in gold.items() if k != "verification_info"}
                details.append({
                    "question": question,
                    "gold": gold,
                    "prediction": prediction,
                    "correct": is_correct,
                    "thinking": thinking_content,
                    "answer_content": answer_content,
                })
                if total % 20 == 0:
                    print(f"Processed {total} examples, Accuracy: {correct/total:.2%}")
                    
                output_token_ids = tokenizer.encode(pred, add_special_tokens=False)
                total_token_len = len(output_token_ids)
                total_token_lens.append(total_token_len)
                if is_correct:
                    correct_token_lens.append(total_token_len)
                else:
                    wrong_token_lens.append(total_token_len)

    if is_async_mode: ### Coding only
        print(f"[Rank {local_rank}] Generation finished. Waiting for background grading threads...")
        thread_executor.shutdown(wait=True)
        
        for future in futures:
            res = future.result()
            correct += res["correct"]
            total += res["total"]
            details.extend(res["details"])
            total_token_lens.extend(res["total_lens"])
            correct_token_lens.extend(res["correct_lens"])
            wrong_token_lens.extend(res["wrong_lens"])
    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total:.2%}")
    
    avg = lambda l: float(sum(l)) / len(l) if l else 0.0
    length_stats = {
        "max_new_tokens": max_new_tokens,
        "avg_total_token_len": avg(total_token_lens),
        "correct_avg_total_token_len": avg(correct_token_lens),
        "wrong_avg_total_token_len": avg(wrong_token_lens),
    }
    
    result = {
        "accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "length_stats": length_stats,
        "details": details
    }
    
    os.makedirs("logs", exist_ok=True)
    model_name = model_name.split("/")[-1]
    log_path = f"logs/{model_name}_{dataset_name}_{method}_{max_new_tokens}_rank{local_rank}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Rank {local_rank}] log written: {log_path}")


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--dataset_name', type=str, default="gsm8k")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=None) 
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--do_sample", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_new_tokens', type=int, default=38912)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--method", type=str, default="swir", choices=["swir", "cot", "cot_greedy"])
    parser.add_argument('--alpha', type=float, default=1.0) # swir-specific
    parser.add_argument('--max_switch_count', type=int, default=None) # swir-specific
    args = parser.parse_args()
    main(args)
