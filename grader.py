from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
from latex2sympy2_extended import NormalizationConfig
import signal
import re

import json
import io
import sys
import ast
import inspect
import math
import bisect
import heapq
import typing
import itertools
import functools
import collections
import random
import string
import operator
import queue
import copy
from collections import deque, Counter, defaultdict
from contextlib import redirect_stdout
import resource

# Math graders
def gsm8k_grader(solution_str: str, ground_truth: str) -> bool:
    # if not ground_truth.startswith("$"):
    #     ground_truth = f"${ground_truth}$"
    gold = parse(
        ground_truth,
        extraction_config=[ExprExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"
    else:
        return verify(gold, answer), str(answer)


def math500_grader(solution_str: str, ground_truth: str) -> bool:
    if not ground_truth.startswith("$"):
        ground_truth = f"${ground_truth}$"
    gold = parse(
        ground_truth,
        extraction_config=[LatexExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"
    else:
        return verify(gold, answer), str(answer)


def aime_grader(solution_str: str, ground_truth: str) -> bool:
    # if not ground_truth.startswith("$"):
    #     ground_truth = f"${ground_truth}$"
    gold = parse(
        ground_truth,
        extraction_config=[ExprExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"
    else:
        return verify(gold, answer), str(answer)
            

# STEM grader
def gpqa_grader(solution_str: str, ground_truth: str) -> bool:
    if not ground_truth.startswith("$"):
        ground_truth = f"${ground_truth}$"
    gold = parse(
        ground_truth,
        extraction_config=[LatexExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"
    else:
        return verify(gold, answer), str(answer)


# Multi-hop graders
def _normalize_qa_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.strip("\"'")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_str_answer(s: str):
    if s is None:
        return ""
    text = s.strip()

    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx != -1:
        start = idx + len(marker)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        candidate = text[start : j - 1].strip()

        if candidate.startswith(r"\text{") and candidate.endswith("}"):
            inner = candidate[len(r"\text{") : -1].strip()
            candidate = inner

        return candidate

    m = re.search(r"[Aa]nswer(?: is|:)\s*([^\n\.]+)", text)
    if m:
        return m.group(1).strip()

    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and ln.strip() not in {"$", "$$", "\\[", "\\]"}
    ]
    if not lines:
        return ""
    cand = lines[-1]
    cand = cand.strip(" .\"'")
    return cand


_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def _extract_number_value(norm_str):
    if not norm_str:
        return None

    tokens = norm_str.split()

    for tok in tokens:
        m = re.search(r"\d+", tok)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass

    for tok in tokens:
        if tok in _NUM_WORDS:
            return _NUM_WORDS[tok]
        if tok in _ORDINAL_WORDS:
            return _ORDINAL_WORDS[tok]

    return None


def is_token_subseq(short_norm, long_norm, max_extra_tokens=None):
    short_tokens = short_norm.split()
    long_tokens = long_norm.split()

    if not short_tokens or len(short_tokens) > len(long_tokens):
        return False

    if max_extra_tokens is not None:
        if len(long_tokens) - len(short_tokens) > max_extra_tokens:
            return False

    for i in range(len(long_tokens) - len(short_tokens) + 1):
        if long_tokens[i : i + len(short_tokens)] == short_tokens:
            return True
    return False


def multihop_grader(solution_str: str, ground_truth):

    pred_ans = _extract_str_answer(solution_str)
    if not pred_ans:
        return False, pred_ans

    pred_norm = _normalize_qa_text(pred_ans)
    if not pred_norm:
        return False, pred_ans

    if isinstance(ground_truth, (list, tuple)):
        gold_list = list(ground_truth)
    else:
        gold_list = [ground_truth]

    MAX_EXTRA_TOKENS_FOR_SUPERSET = 3

    pred_num = _extract_number_value(pred_norm)

    for g in gold_list:
        gold_norm = _normalize_qa_text(str(g))
        if not gold_norm:
            continue

        if pred_norm == gold_norm:
            return True, pred_ans

        gold_num = _extract_number_value(gold_norm)
        if pred_num is not None and gold_num is not None and pred_num == gold_num:
            return True, pred_ans

        if is_token_subseq(gold_norm, pred_norm, max_extra_tokens=MAX_EXTRA_TOKENS_FOR_SUPERSET):
            return True, pred_ans

        if is_token_subseq(pred_norm, gold_norm):
            return True, pred_ans

    return False, pred_ans


def commonsenseqa_grader(solution_str: str, sample):

    raw = _extract_str_answer(solution_str)
    if raw is None:
        raw = ""
    raw_upper = raw.upper().strip()

    labels = list(sample["choices"]["label"])
    texts = list(sample["choices"]["text"])
    gold_label = sample["answerKey"]

    letters = re.findall(r"[A-E]", raw_upper)
    pred_label = None
    if len(letters) == 1:
        pred_label = letters[0]

    if pred_label is not None:
        is_correct = (pred_label == gold_label)
        return is_correct, pred_label

    pred_norm = _normalize_qa_text(raw)
    norm_texts = [_normalize_qa_text(t) for t in texts]

    if pred_norm in norm_texts:
        idx = norm_texts.index(pred_norm)
        pred_label = labels[idx]
        is_correct = (pred_label == gold_label)
        return is_correct, f"{pred_label}: {texts[idx]}"

    return False, raw


# Coding graders
class GraderTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise GraderTimeoutError()


def _extract_python_code(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()

    def looks_like_code(body: str) -> bool:
        for line in body.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("import ", "from ", "def ", "class ", "@")):
                return True
        return False

    fences = list(
        re.finditer(
            r"```(?:python)?\s*(.*?)```",
            s,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    if fences:
        bodies = [m.group(1) for m in fences]
        code_bodies = [b for b in bodies if looks_like_code(b)]
        if code_bodies:
            return code_bodies[-1].strip()
        return bodies[-1].strip()

    lines = s.splitlines()
    blocks = []
    current = []
    in_block = False

    for line in lines:
        stripped = line.lstrip()
        is_code_line = False
        if stripped.startswith(("import ", "from ", "def ", "class ", "@", "#")):
            is_code_line = True
        elif line.startswith((" ", "\t")) and stripped != "":
            is_code_line = True

        if not in_block:
            if is_code_line:
                in_block = True
                current.append(line)
        else:
            if is_code_line or stripped == "":
                current.append(line)
            else:
                blocks.append("\n".join(current))
                current = []
                in_block = False

    if in_block and current:
        blocks.append("\n".join(current))

    if blocks:
        code_blocks = [b for b in blocks if looks_like_code(b)]
        if code_blocks:
            return code_blocks[-1].strip()
        return blocks[-1].strip()

    return s


def humaneval_grader(solution_str: str, sample, timeout: float = 3.0):
    completion = _extract_python_code(solution_str)

    prompt = sample["prompt"]
    test_code = sample["test"]
    entry_point = sample["entry_point"]

    full_code = prompt + "\n" + completion + "\n\n" + test_code

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)

    try:
        global_ns = {}
        exec(full_code, global_ns)
        candidate = global_ns[entry_point]
        check_fn = global_ns["check"]
        check_fn(candidate)
        passed = True
    except GraderTimeoutError:
        passed = False
    except Exception:
        passed = False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)

    return passed, completion


def mbpp_grader(solution_str: str, sample, timeout: float = 10.0):
    completion = _extract_python_code(solution_str)

    if "test_setup_code" in sample:
        setup_code = sample.get("test_setup_code", "") or ""
        tests = list(sample.get("test_list", [])) + list(
            sample.get("challenge_test_list", []) or []
        )
        prelude = setup_code
    else:
        imports = "\n".join(sample.get("test_imports", []))
        tests = list(sample.get("test_list", []))
        prelude = imports

    if prelude.strip():
        full_code = prelude + "\n\n" + completion
    else:
        full_code = completion

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)

    try:
        global_ns = {}
        exec(full_code, global_ns)
        for t in tests:
            exec(t, global_ns)
        passed = True
    except GraderTimeoutError:
        passed = False
    except Exception:
        passed = False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)

    return passed, completion


def leetcode_grader(solution_str: str, sample, timeout: float = 10.0):
    completion = _extract_python_code(solution_str)
    test_code = sample.get("test", "")

    full_code = completion + "\n\n" + test_code

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)

    try:
        global_ns = {}
        exec(full_code, global_ns)
        passed = True
    except GraderTimeoutError:
        passed = False
    except Exception:
        passed = False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)

    return passed, completion


def set_memory_limit(max_mem_gb=16):
    try:
        max_mem = max_mem_gb * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, resource.RLIMIT_INFINITY))
    except Exception:
        pass


def run_livecode_one_case(code_str, inp_str, expected_str, timeout):
    
    def smart_equal(prediction_obj, expected_s):
        try:
            expected_obj = json.loads(expected_s)
            if isinstance(expected_obj, float) or (isinstance(expected_obj, list) and expected_obj and isinstance(expected_obj[0], float)):
                 if prediction_obj == expected_obj: return True
            if prediction_obj == expected_obj: return True
        except: pass
        
        if isinstance(prediction_obj, (list, dict, bool, int, float)):
            pred_s = json.dumps(prediction_obj)
        else:
            pred_s = str(prediction_obj)
            
        o_tokens = pred_s.strip().replace('[', ' [ ').replace(']', ' ] ').replace(',', ' ').split()
        e_tokens = expected_s.strip().replace('[', ' [ ').replace(']', ' ] ').replace(',', ' ').split()
        
        if len(o_tokens) != len(e_tokens): return False
        for o, e in zip(o_tokens, e_tokens):
            if o == e: continue
            try:
                if not math.isclose(float(o), float(e), rel_tol=1e-4, abs_tol=1e-4): return False
            except ValueError: return False
        return True

    def run_leetcode(global_ns, inp):
        try:
            exec(code_str, global_ns)
            if "Solution" not in global_ns: return None, None
            SolClass = global_ns["Solution"]
            instance = SolClass()
            methods = [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]
            if not methods: return None, None
            entry_point = getattr(instance, methods[0])
            
            parsed_args_candidates = []
            try:
                clean_inp = re.sub(r'\b[a-zA-Z_]\w*\s*=\s*', '', inp)
                if ',' in clean_inp:
                    tuple_inp = f"({clean_inp})"
                    parsed_args_candidates.append(ast.literal_eval(tuple_inp))
                else:
                    parsed_args_candidates.append([ast.literal_eval(clean_inp)])
            except: pass
            try:
                json_inp = json.loads(inp)
                if isinstance(json_inp, list): parsed_args_candidates.append(json_inp)
                else: parsed_args_candidates.append([json_inp])
            except: pass
            try:
                lines_args = []
                for line in inp.strip().splitlines():
                    line = line.strip()
                    if not line: continue
                    if '=' in line: line = line.split('=', 1)[1].strip()
                    try: lines_args.append(json.loads(line))
                    except: 
                        try: lines_args.append(ast.literal_eval(line))
                        except: lines_args.append(line)
                if lines_args: parsed_args_candidates.append(lines_args)
            except: pass
            parsed_args_candidates.append([inp])

            sig = inspect.signature(entry_point)
            required_params = len(sig.parameters)
            last_err = None
            for args_candidate in parsed_args_candidates:
                try:
                    if isinstance(args_candidate, tuple): args_candidate = list(args_candidate)
                    if len(args_candidate) == required_params:
                        return entry_point(*args_candidate), "success"
                    if required_params == 1:
                        return entry_point(args_candidate), "success"
                except Exception as e:
                    last_err = e
                    continue
            return None, last_err
        except Exception as e:
            return None, e

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    
    try:
        set_memory_limit(max_mem_gb=16)
        sys.setrecursionlimit(10**6)
        class ListNode:
            def __init__(self, val=0, next=None): self.val, self.next = val, next
            def __repr__(self): return f"ListNode({self.val})"
        class TreeNode:
            def __init__(self, val=0, left=None, right=None): self.val, self.left, self.right = val, left, right
            def __repr__(self): return f"TreeNode({self.val})"

        global_ns = {
            "__name__": "__main__", "__builtins__": __builtins__,
            "sys": sys, "math": math, "bisect": bisect, "heapq": heapq,
            "itertools": itertools, "functools": functools, "collections": collections,
            "re": re, "random": random, "string": string, "operator": operator,
            "queue": queue, "copy": copy, "typing": typing, "json": json,
            "List": typing.List, "Dict": typing.Dict, "Tuple": typing.Tuple, "Set": typing.Set,
            "Optional": typing.Optional, "Union": typing.Union, "Any": typing.Any,
            "Deque": typing.Deque, "Iterable": typing.Iterable, "Iterator": typing.Iterator,
            "deque": collections.deque, "defaultdict": collections.defaultdict, "Counter": collections.Counter,
            "lru_cache": functools.lru_cache, "cache": functools.cache, "reduce": functools.reduce,
            "gcd": math.gcd, "inf": math.inf, "nan": math.nan, "ceil": math.ceil, "floor": math.floor, "sqrt": math.sqrt,
            "bisect_left": bisect.bisect_left, "bisect_right": bisect.bisect_right,
            "heappush": heapq.heappush, "heappop": heapq.heappop, "heapify": heapq.heapify,
            "ListNode": ListNode, "TreeNode": TreeNode
        }
        if hasattr(math, 'lcm'): global_ns['lcm'] = math.lcm
        else: global_ns['lcm'] = lambda a, b: abs(a*b) // math.gcd(a, b)

        lc_res, status = run_leetcode(dict(global_ns), inp_str)
        if status == "success":
            return smart_equal(lc_res, expected_str)

        stdin_backup = sys.stdin
        stdout_backup = sys.stdout
        real_inp = inp_str if inp_str.endswith('\n') else inp_str + '\n'
        sys.stdin = io.StringIO(real_inp)
        out_buf = io.StringIO()
        sys.stdout = out_buf
        
        try:
            c_obj = compile(code_str, "<string>", "exec")
            exec(c_obj, global_ns)
            output = out_buf.getvalue()
            return smart_equal(output, expected_str)
        except SystemExit:
            output = out_buf.getvalue()
            return smart_equal(output, expected_str)
        except Exception:
            return False
        finally:
            sys.stdin = stdin_backup
            sys.stdout = stdout_backup

    except GraderTimeoutError:
        return False
    except Exception:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def livecodebench_grader(solution_str: str, sample, timeout: float = 6.0):
    completion = _extract_python_code(solution_str)
    raw_info = sample.get("verification_info", "") or ""
    try: info = json.loads(raw_info)
    except: 
        try: info = json.loads(ast.literal_eval(raw_info))
        except: return False, completion
    ground_truth = info.get("ground_truth", [])
    if isinstance(ground_truth, str):
        try: ground_truth = json.loads(ground_truth)
        except: 
            try: ground_truth = ast.literal_eval(ground_truth)
            except: ground_truth = []
    if not ground_truth: return False, completion

    try: compile(completion, "<string>", "exec")
    except SyntaxError: return False, completion

    for case in ground_truth:
        inp = str(case.get("input", ""))
        expected = str(case.get("output", ""))
        
        passed = run_livecode_one_case(completion, inp, expected, timeout)
        if not passed:
            return False, completion

    return True, completion

def answer_match(dataset_name, pred, gold):
    if dataset_name == "gsm8k":
        return gsm8k_grader(pred, gold)
    elif dataset_name == "math500":
        return math500_grader(pred, gold)
    elif "aime" in dataset_name:
        return aime_grader(pred, gold)
    elif dataset_name == "gpqa_diamond":
        return gpqa_grader(pred, gold)
    elif dataset_name == "2wikimqa":
        return multihop_grader(pred, gold)
    elif dataset_name == "commonsenseqa":
        return commonsenseqa_grader(pred, gold)
    elif dataset_name == "humaneval":
        return humaneval_grader(pred, gold)
    elif dataset_name == "mbpp":
        return mbpp_grader(pred, gold)
    elif dataset_name == "leetcode_contest":
        return leetcode_grader(pred, gold)
    elif dataset_name == "livecodebench":
        return livecodebench_grader(pred, gold)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    

def answer_extraction(pred):
    return gsm8k_grader(pred, None)
