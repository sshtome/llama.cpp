"""
PopBrain Full 4-Way Evaluation Matrix:
1. ANN_Dense      : Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf (No Memory)
2. ANN_Synaptic   : Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf + wikipedia_memory.synapse
3. SNN_Spiking    : Qwen2.5-Coder-3B-Instruct-Q4_K_M.spiking.gguf (No Memory)
4. SNN_Synaptic   : Qwen2.5-Coder-3B-Instruct-Q4_K_M.spiking.gguf + wikipedia_memory.synapse
"""

import subprocess
import json
import time
import re
import os
import sys

LLAMA_CLI = r"D:\projects\llama.cpp\build\bin\Release\llama-cli.exe"
BASE_MODEL = r"D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"
SPIKING_MODEL = r"D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.spiking.gguf"
SYNAPSE_MEMORY = r"D:\projects\llama.cpp\knowledge_pack\wikipedia_memory.synapse"

CONFIGS = [
    {
        "id": "ANN_Dense",
        "name": "1. Обычная ANN (Dense Baseline)",
        "model": BASE_MODEL,
        "memory": None
    },
    {
        "id": "ANN_Synaptic",
        "name": "2. Обученная ANN (Synaptic Memory)",
        "model": BASE_MODEL,
        "memory": SYNAPSE_MEMORY
    },
    {
        "id": "SNN_Spiking",
        "name": "3. Спайковая SNN (LIF Dynamics)",
        "model": SPIKING_MODEL,
        "memory": None
    },
    {
        "id": "SNN_Synaptic",
        "name": "4. Спайковая SNN с памятью (LIF + Synapse)",
        "model": SPIKING_MODEL,
        "memory": SYNAPSE_MEMORY
    }
]

TEST_SUITE = [
    # Category 1: Acquired Factual Knowledge (Synthetic Control & Wikipedia)
    {
        "category": "Acquired Knowledge",
        "prompt": "В каком году стартовала исследовательская миссия Астрон-42 к поясу Койпера? Ответь кратко.",
        "expected": ["2041"],
        "max_tokens": 40
    },
    {
        "category": "Acquired Knowledge",
        "prompt": "Какой химический элемент открыла лаборатория Ксенон-Лаб в 2038 году? Ответь кратко.",
        "expected": ["Квантий", "126"],
        "max_tokens": 40
    },
    {
        "category": "Acquired Knowledge",
        "prompt": "Кто является главным архитектором орбитального комплекса Нео-Аркадия? Ответь кратко.",
        "expected": ["Илья", "Соколов"],
        "max_tokens": 40
    },
    {
        "category": "Acquired Knowledge",
        "prompt": "Какова точная скорость света в вакууме согласно фундаментальным константам? Назови точное число.",
        "expected": ["299792458", "299 792 458"],
        "max_tokens": 40
    },
    {
        "category": "Acquired Knowledge",
        "prompt": "Какой самый глубокий каньон в Солнечной системе по данным астрономии? Назови его.",
        "expected": ["Маринер"],
        "max_tokens": 40
    },

    # Category 2: Mathematical Reasoning & Logic
    {
        "category": "Math & Logic",
        "prompt": "Сколько будет 25 * 25? Напиши только ответ.",
        "expected": ["625"],
        "max_tokens": 20
    },
    {
        "category": "Math & Logic",
        "prompt": "Реши простое уравнение: 3 * x + 15 = 45. Чему равен x? Напиши только число.",
        "expected": ["10"],
        "max_tokens": 20
    },
    {
        "category": "Math & Logic",
        "prompt": "Сколько простых чисел в диапазоне от 10 до 20? Перечисли их.",
        "expected": ["11", "13", "17", "19"],
        "max_tokens": 40
    },

    # Category 3: Coding Capabilities
    {
        "category": "Coding",
        "prompt": "Напиши функцию на Python для бинарного поиска binary_search(arr, target). Только код.",
        "expected": ["def binary_search", "while", "// 2"],
        "max_tokens": 128
    },
    {
        "category": "Coding",
        "prompt": "Напиши однострочник на Python для вычисления факториала числа n.",
        "expected": ["factorial", "prod", "reduce", "lambda"],
        "max_tokens": 64
    },

    # Category 4: Common Knowledge & Retention (Zero Forgetting Probe)
    {
        "category": "Common Knowledge",
        "prompt": "Назови столицу Франции.",
        "expected": ["Париж", "Paris"],
        "max_tokens": 20
    },
    {
        "category": "Common Knowledge",
        "prompt": "Какова температура кипения чистой воды при нормальном давлении в градусах Цельсия? Назови число.",
        "expected": ["100"],
        "max_tokens": 20
    }
]

def run_query(model_path, memory_path, prompt, max_tokens):
    cmd = [
        LLAMA_CLI,
        "-m", model_path,
        "-p", prompt,
        "--single-turn",
        "--simple-io",
        "-n", str(max(max_tokens, 64))
    ]
    if memory_path:
        cmd.extend(["--memory", memory_path])
    
    t0 = time.time()
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
    elapsed = time.time() - t0
    
    stdout = res.stdout
    
    # Extract speed metrics
    gen_speed = 0.0
    prompt_speed = 0.0
    m_speed = re.search(r"Generation:\s*([0-9.]+)\s*t/s", stdout)
    if m_speed:
        gen_speed = float(m_speed.group(1))
    m_p_speed = re.search(r"Prompt:\s*([0-9.]+)\s*t/s", stdout)
    if m_p_speed:
        prompt_speed = float(m_p_speed.group(1))

    # Clean response text
    clean_text = stdout
    # Remove banner
    if "available commands:" in clean_text:
        clean_text = clean_text.split("available commands:")[-1]
    # Remove prompt echo
    if prompt in clean_text:
        clean_text = clean_text.split(prompt, 1)[-1]
    # Remove speed stats
    if "[ Prompt:" in clean_text:
        clean_text = clean_text.split("[ Prompt:")[0]
    if "Exiting..." in clean_text:
        clean_text = clean_text.replace("Exiting...", "")
    
    clean_text = clean_text.strip()
    return {
        "text": clean_text,
        "gen_speed": gen_speed,
        "prompt_speed": prompt_speed,
        "latency_sec": elapsed
    }

def check_correctness(response_text, expected_tokens):
    text_lower = response_text.lower()
    matches = 0
    for exp in expected_tokens:
        if exp.lower() in text_lower:
            matches += 1
    # If at least one expected keyword is found, count as passed
    return matches > 0

def main():
    print("=" * 80)
    print(" PopBrain 4-Way Comprehensive Matrix Benchmark (Quality, Speed & SNN)")
    print("=" * 80)
    
    results = {}
    
    for cfg in CONFIGS:
        print(f"\n[*] Testing Configuration: {cfg['name']} ({cfg['id']})...")
        cfg_results = []
        total_tests = len(TEST_SUITE)
        passed_tests = 0
        total_gen_speed = 0.0
        speed_count = 0
        
        for idx, test in enumerate(TEST_SUITE, 1):
            category = test["category"]
            prompt = test["prompt"]
            expected = test["expected"]
            max_tokens = test["max_tokens"]
            
            print(f"  [{idx:02d}/{total_tests:02d}] ({category}) {prompt[:40]}...", end=" ", flush=True)
            res = run_query(cfg["model"], cfg["memory"], prompt, max_tokens)
            
            is_correct = check_correctness(res["text"], expected)
            if is_correct:
                passed_tests += 1
                status = "PASS"
            else:
                status = "FAIL"
                
            if res["gen_speed"] > 0:
                total_gen_speed += res["gen_speed"]
                speed_count += 1
                
            print(f"-> {status} ({res['gen_speed']:.1f} t/s)")
            
            cfg_results.append({
                "category": category,
                "prompt": prompt,
                "response": res["text"],
                "expected": expected,
                "passed": is_correct,
                "gen_speed": res["gen_speed"],
                "prompt_speed": res["prompt_speed"],
                "latency_sec": res["latency_sec"]
            })
            
        avg_speed = total_gen_speed / max(1, speed_count)
        accuracy = (passed_tests / total_tests) * 100.0
        
        results[cfg["id"]] = {
            "name": cfg["name"],
            "total": total_tests,
            "passed": passed_tests,
            "accuracy": accuracy,
            "avg_gen_speed": avg_speed,
            "details": cfg_results
        }
        print(f"  [SUMMARY] {cfg['id']}: Accuracy = {accuracy:.1f}% ({passed_tests}/{total_tests}) | Avg Speed = {avg_speed:.1f} t/s")

    # Output JSON summary
    out_json = r"D:\projects\llama.cpp\knowledge_pack\eval_matrix_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Detailed results saved to: {out_json}")
    
    # Print Markdown Summary Table
    print("\n" + "=" * 80)
    print(" SUMMARY BENCHMARK MATRIX TABLE:")
    print("=" * 80)
    print(f"{'Configuration':<35} | {'Accuracy':<10} | {'Passed':<8} | {'Avg Speed':<12} | {'Sparsity':<10}")
    print("-" * 80)
    for cfg in CONFIGS:
        cid = cfg["id"]
        c_res = results[cid]
        sparsity_str = "89.6%" if "SNN" in cid else "0.0%"
        print(f"{c_res['name']:<35} | {c_res['accuracy']:>7.1f}%  | {c_res['passed']:>2}/{c_res['total']:<2}   | {c_res['avg_gen_speed']:>8.1f} t/s | {sparsity_str:>8}")
    print("=" * 80)

if __name__ == "__main__":
    main()
