# MathAgent

MathAgent is a system for generating and evaluating applied mathematics multiple-choice questions with LLMs. It is designed as an LLMA-style benchmark where models solve numerical math problems using reasoning, generated code, or a tool-augmented agent loop.

The current workflow has been used with:

- `claude-opus-4-8`
- `gpt-5.5`
- `Qwen/Qwen2.5-7B-Instruct`

## What This Project Does

MathAgent creates benchmark questions from several applied mathematics domains, stores the correct answers, and evaluates language models on those questions.

The benchmark covers:

- Stochastic reactor probability problems
- ODE finite-time blow-up problems
- Dominant-balance integral approximations
- Nondimensionalization problems
- Laplace-method asymptotic approximation problems

Each question has answer options `A`, `B`, `C`, `D`, and optionally `E`.

Option `E` means `None of the Above`. In the tool-augmented agent workflow, option `E` is hidden from the model during solving. If the agent cannot submit a valid answer before the step limit, the evaluator assigns `E`.

## Project Structure

```text
MathAgent/
├── generator.py
├── code_evaluator.py
├── local_code_evaluator.py
├── no_tool_batch_evaluator.py
├── agent_runner.py
├── tools.py
├── code_runner.py
├── summarize_results.py
├── claude_code_batch_evaluator.py
├── solvers/
├── questions/
├── answers/
└── results/
```

Important files:

- `generator.py` generates the benchmark questions and answer key.
- `solvers/` contains the math solvers and question generators for each domain.
- `code_evaluator.py` evaluates API-based models such as Claude and GPT.
- `local_code_evaluator.py` evaluates local HuggingFace models such as Qwen.
- `no_tool_batch_evaluator.py` runs no-tool batch evaluations for Claude and GPT.
- `agent_runner.py` runs the tool-augmented MathAgent loop.
- `tools.py` defines the text-based tools used by the agent.
- `code_runner.py` safely executes generated Python code in a subprocess.
- `results/` stores model evaluation outputs.

## Setup

Install the Python packages used by the generators, solvers, and evaluators:

```bash
pip install anthropic openai transformers torch scipy sympy bitsandbytes accelerate
```

For API models, set the relevant API key:

```bash
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
```

## Generate Questions

To generate a benchmark with questions from all domains:

```bash
python generator.py --stoch 50 --blowup 50 --dom 50 --nondim 50 --laplace 50 --include-e
```

This writes:

```text
questions/all_questions_with_E.json
answers/all_answers_with_E.json
```

## Run API Model Evaluation

Use `code_evaluator.py` for API-based models.

Tool-augmented evaluation:

```bash
python code_evaluator.py --model claude-opus-4-8 --mode tool_augmented --max-tokens 2048
```

```bash
python code_evaluator.py --model gpt-5.5 --mode tool_augmented --max-tokens 2048
```

The evaluator automatically chooses the Anthropic or OpenAI client from the model name.

## Run No-Tool Batch Evaluation

Use `no_tool_batch_evaluator.py` when you want Claude or GPT to answer directly through a batch API job. This workflow was used to reduce API calling cost, since batch APIs are typically about half the cost of real-time requests. In this mode, the model sees the full multiple-choice question and picks `A`, `B`, `C`, `D`, or `E` without running code.

For Claude:

```bash
python no_tool_batch_evaluator.py submit --model claude-opus-4-8
python no_tool_batch_evaluator.py status --model claude-opus-4-8
python no_tool_batch_evaluator.py fetch  --model claude-opus-4-8
```

For GPT:

```bash
python no_tool_batch_evaluator.py submit --model gpt-5.5
python no_tool_batch_evaluator.py status --model gpt-5.5
python no_tool_batch_evaluator.py fetch  --model gpt-5.5
```

The script keeps a separate state file for each model, so submitted batch jobs can be checked and fetched later. The `fetch` step downloads the model outputs, extracts the answer letter, scores against `answers/all_answers_with_E.json`, and saves a result file under `results/`.

## Run Local Model Evaluation

Use `local_code_evaluator.py` for local HuggingFace models.

Example with Qwen:

```bash
python local_code_evaluator.py --model Qwen/Qwen2.5-7B-Instruct --mode tool_augmented
```

The local evaluator uses 4-bit loading by default. To disable 4-bit loading:

```bash
python local_code_evaluator.py --model Qwen/Qwen2.5-7B-Instruct --mode tool_augmented --no-4bit
```

## Evaluation Modes

MathAgent supports three evaluation modes.

### No Tool Access

The model answers directly from reasoning.

```bash
python code_evaluator.py --model claude-opus-4-8 --mode no_tool_access
```

### Code Generation

The model writes Python code, the evaluator executes it, and the model chooses the closest option.

```bash
python code_evaluator.py --model claude-opus-4-8 --mode code_generation --max-tokens 2048
```

### Tool Augmented

The model runs inside the MathAgent loop. It must call `run_python` for numerical work, observe the output, and then call `submit_answer`.

```bash
python code_evaluator.py --model claude-opus-4-8 --mode tool_augmented --max-tokens 2048
```

This is the main workflow used for the current MathAgent experiments.

## Summarize Results

After running an evaluation, summarize a result file with:

```bash
python summarize_results.py results/results_gpt-5-5_tool_augmented.json
```

Result files include per-question details such as:

- Question ID
- Domain
- Model answer
- Expected answer
- Correct/incorrect status
- Tool steps and executions for agent runs

## Notes

Generated model code is executed locally through `code_runner.py`, which uses a subprocess and timeout so failed or slow code does not stop the main evaluator.

The benchmark is intended for comparing how well LLMs can solve numerical applied mathematics problems when they are allowed to use code as a tool.
