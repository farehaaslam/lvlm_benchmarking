# Gemma 4 12B on Video-MMMU: benchmark record

This folder holds the code, logs and method used to evaluate **`google/gemma-4-12B-it`** (BF16, unquantized) on
**Video-MMMU** (all 300 videos, 900 questions). The goal of this document is that a reader can answer any question
about *how* the numbers were produced, without having to trust the author, and can check every claim against the
code, the logs, or the raw per-question outputs.

The whole benchmark was produced by **one script, [`vllm_eval.py`](vllm_eval.py)**, using the prompt and scoring
code in [`videommmu_official.py`](videommmu_official.py). Other scripts in this folder were **not** used to produce
the reported results (see [Files](#1-files-in-this-folder)).

---

## Contents

0. [Results](#0-results)
1. [Files in this folder](#1-files-in-this-folder)
2. [Model](#2-model)
3. [Dataset](#3-dataset)
4. [What the model is given (input construction)](#4-what-the-model-is-given-input-construction)
5. [Generation settings](#5-generation-settings)
6. [Thinking, the thinking budget, and what gets scored](#6-thinking-the-thinking-budget-and-what-gets-scored)
7. [Scoring](#7-scoring)
8. [How the run was executed (full history)](#8-how-the-run-was-executed-full-history)
9. [Environment](#9-environment)
10. [Output files](#10-output-files)
11. [How to reproduce or audit](#11-how-to-reproduce-or-audit)
12. [Limitations and comparability: read before comparing numbers](#12-limitations-and-comparability-read-before-comparing-numbers)
13. [FAQ for reviewers](#13-faq-for-reviewers)

---

## 0. Results

RESULTS_PLACEHOLDER

---

## 1. Files in this folder

| File | Role in the reported results |
|---|---|
| [`vllm_eval.py`](vllm_eval.py) | **The benchmark runner.** Loads the questions, builds each prompt, runs Gemma 4 through vLLM, splits the reasoning from the answer, scores it, and writes one CSV per video. |
| [`videommmu_official.py`](videommmu_official.py) | **Prompt and scoring logic**, copied from lmms-eval's `video_mmmu` task (provenance in [§7](#7-scoring)). Used by `vllm_eval.py`. |
| [`native_sampling.py`](native_sampling.py) | Earlier Hugging Face `generate` runner (one question at a time, too slow for a full run). **Its generation path was not used.** `vllm_eval.py` imports only its helper functions: `TRACKS`, `build_video_index`, `load_image`, `load_questions`, `natural_key`, and `summarize` (the final aggregation and re-scoring). |
| [`score.py`](score.py) | Standalone re-scorer, safe to run during a run. Its default `OUT_DIR` points to the *native* results folder, so pass `--out_dir ../results/videommmu_gemma4_12b_vllm` to use it on these results. It applies the same `vm.score` as `summarize`. |
| [`sampling.py`](sampling.py) | Oldest prototype: different data format, its own regex scorer, greedy decoding, 512 tokens. **Not used at all.** Kept only for history. |
| `smoke.log` | Log of the 3-video smoke test, whose outputs are part of the final results (see [§8](#8-how-the-run-was-executed-full-history)). |
| `logs/old/vllm_{0,1}_run1.log` | Logs of the first full-run attempt (stopped after one batch; its outputs are part of the final results). |
| `logs/vllm_{0,1}.log` | Logs of the run that completed the benchmark. |

Results are written to `../results/videommmu_gemma4_12b_vllm/`.

> `results/` and `video_mmmu_gemma/logs/*.log` are in `.gitignore`. They are **not** in git, so share them
> separately with anyone who audits the run.

---

## 2. Model

| Item | Value |
|---|---|
| Model | `google/gemma-4-12B-it` (instruction-tuned) from the Hugging Face Hub |
| Exact weights | Hub snapshot **`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`** (the `main` ref at download time; one `model.safetensors`, 22.28 GiB) |
| Precision | **BF16**, unquantized (`dtype="bfloat16"`, `quantization=None` in the engine log) |
| Architecture as loaded by vLLM | `Gemma4UnifiedForConditionalGeneration` |
| Chat template | The model's own `chat_template.jinja` from the same snapshot (no custom template) |
| Fine-tuning / adapters / prompt tuning | **None** |

---

## 3. Dataset

### Source and version

- **`lmms-lab/VideoMMMU`** on the Hugging Face Hub (the official release used by lmms-eval), dataset revision
  **`d1c35ac933123d79e877b7f1b9506afb0309cf1b`** (from `dataset/videommmu/.cache/huggingface/download/*.metadata`).
- Questions come from the three official parquet files: `Perception/`, `Comprehension/`, and `Adaptation/test-00000-of-00001.parquet`.
- Videos come from the official zips (`Art`, `Business`, `Engineering`, `Humanities`, `Medicine`, `Science`).

### What is evaluated

| Track | Questions | Question types | `qa_type` breakdown |
|---|---|---|---|
| Perception | 300 | 300 multiple-choice | OCR 277, ASR 23 |
| Comprehension | 300 | 300 multiple-choice | Concept Comprehension 171, Problem-solving Comprehension 129 |
| Adaptation | 300 | 279 multiple-choice, 20 open, 1 labelled `"None"` (see quirks) | Problem-solving Adaptation 159, Case Study Analysis 141 |
| **Total** | **900** | | |

- **Every question is evaluated**, with no filtering and no sub-sampling. Every video id has all three tracks (the
  loader aborts otherwise), and all 300 ids have a matching video file (`0 missing` in every log).
- Multiple-choice questions have between 2 and 14 options; most have 10.
- By id prefix: 212 `validation_`, 45 `new_`, 40 `test_`, 3 `dev_`. These are the dataset's own ids, and all are part of the official 900.

### Which video file is used for each question

`build_video_index` walks `dataset/videommmu/` and maps each file **stem** to its path. The folder holds 647 video
files, but only the correct 300 can match a question id:

- `question_only/<id>_image.mp4` (300 files, the dataset's question-only ablation videos) have an `_image` suffix, so they **never match** a question id.
- `__MACOSX/**/._<id>.mp4` (46 macOS resource-fork files from the zips) start with `._`, so they **never match** either.
- Verified: all 300 ids map into the six domain folders (Engineering 113, Business 44, Science 44, Medicine 43,
  Humanities 35, Art 21). No duplicate stems were found; the script prints a warning if any exist.

**The same video file is used for all three questions of a video.** Video-MMMU's own convention: the last frame of
each video shows the quiz question for that video.

### Dataset quirks (inherited from the official data; handled exactly as the official code handles them)

1. **`validation_Accounting_13` (Adaptation)** has `question_type == "None"` (the string), although it has 10 options
   and gold answer `"C"`. The official code branches only on `question_type == "multiple-choice"`, so this question
   gets the **open-ended prompt with no options shown**, and its answer is judged with the open-ended matcher
   against the gold value `"C"`. We did not patch the data, so it is handled the same way as in lmms-eval.
2. **Open-ended questions with letter answers.** `validation_Physics_21` (gold `"A"`) and
   `validation_Basic_Medical_Science_10` (gold `"C"`) are labelled `open`. The official open-ended matcher looks for
   `" a"` / `"a "` as a substring of the parsed candidates, a known weakness of the upstream scorer. This was kept unchanged.
3. **`validation_Math_15`** has a gold answer that is a malformed list stored as a string (`"['24/7', '3.429', '3.43]"`).
   It is matched as a plain string, as upstream does.
4. **293 of 300 Adaptation questions contain the placeholder `<image 1>`** in the question text. The official prompt
   leaves it in, and so do we.

---

## 4. What the model is given (input construction)

Each question is one single-turn chat request: **one user message**, no system prompt text, and no few-shot examples.
The content parts are in this order:

```
[ video ]  →  [ question text (official prompt) ]  →  [ question image ]   (image: Adaptation only)
```

### 4.1 Video

- Passed to vLLM as `{"type": "video_url", "video_url": {"url": "file:///…/<id>.mp4"}}`. vLLM decodes it with OpenCV
  (`media_io_kwargs={"video": {"num_frames": 32, "video_backend": "opencv"}}`).
- **Frame selection:** vLLM's `VideoBackend.compute_frames_index_to_sample` picks
  `np.linspace(0, total_frames - 1, 32, dtype=int)`: 32 evenly spaced frames that **include the first and the
  last frame**. No fps cap is set. Videos with fewer than 32 frames would use every frame.
- **32 frames is Gemma 4's own video setting:** the model's `processor_config.json` sets `video_processor.num_frames = 32`
  and `max_soft_tokens = 70` per frame. vLLM's Gemma 4 path (`gemma4_mm.py`) passes the decoded frames with
  `do_sample_frames=False`, so there is **no second sampling**. It inserts `mm:ss` timestamps between frames, as the
  Transformers processor does.
- Input length per prompt is about 2,300–3,100 tokens (the `input_tokens` column), consistent with 32 × (70 soft tokens + markup) plus the question.
- Because the last frame is always sampled, the model sees the quiz frame. For Perception and Comprehension the
  official prompt tells it to ignore that frame. For Adaptation that frame shows the question image the prompt refers to.
- Performance-only detail: a video is decoded **once** per batch and the decoded frames are reused for its three
  questions (`_cached_fetch_video`). The cache key includes the URL and all decode kwargs. It returns the same object
  that vLLM would otherwise rebuild, so it **does not change model inputs**. It is cleared after every batch.

### 4.2 Question text: official lmms-eval prompts, verbatim

Built by `videommmu_official.doc_to_text`.

**Perception / Comprehension** (always multiple-choice):
```
{question}
A. {option 1}
B. {option 2}
...
Please ignore the Quiz question in last frame of the video.
```

**Adaptation, multiple-choice:**
```
You should watch and learn the video content. Then apply what you learned to answer the following multi-choice question. The image for this question is at the end of the video.
{question}
A. {option 1}
...
```

**Adaptation, open-ended** (and the `"None"` item above):
```
You should watch and learn the video content. Then apply what you learned to answer the following open-ended question. The image for this question is at the end of the video.
{question}
```

- Options already starting with `A.`, `B.` … are used as they are. Otherwise letters are added (`parse_options`, copied from upstream).
- The official `MCQ_POST_PROMPT` is the empty string. **No extra instruction is added**: no "answer with the letter only",
  no "think step by step", and no required output format.

### 4.3 Adaptation question image

- The dataset's `image` field (present for all 300 Adaptation questions) is decoded with PIL, converted to RGB,
  re-encoded **losslessly as PNG**, and attached as an `image_url` data URL **after** the question text.
- It is attached separately because inside the video the image is only *one* of 32 sampled frames. The dataset ships
  the image as its own `image` field for this purpose. The official prompt sentence "The image for this question is at the end of the
  video" stays true, because the last video frame is sampled (§4.1) and the image also follows the video in the message.
- Perception and Comprehension questions get **no** extra image (`used_question_image = 0` in the CSVs).
- A harmless PIL warning ("Palette images with Transparency…") appears in one log. It comes from the RGB conversion of a palette PNG.

### 4.4 Chat template and thinking switch

`llm.chat(..., chat_template_kwargs={"enable_thinking": True})`. With this flag the model's own template puts the
`<|think|>` control token in a system turn, which is Gemma 4's documented way to turn reasoning on. No other system text is added.

---

## 5. Generation settings

| Parameter | Value | Note |
|---|---|---|
| Decoding | **Sampling** | Not greedy |
| `temperature` | 1.0 | |
| `top_p` | 0.95 | |
| `top_k` | **20** | Google's `generation_config.json` default is 64. See below. |
| `min_p` | 0.0 | |
| `presence_penalty` | 1.5 | |
| `seed` | 0 | Per-request seed (`SamplingParams(seed=0)`) |
| Thinking | **On** | |
| Thinking budget | **16,378 tokens** | Enforced by vLLM (§6) |
| `max_tokens` (thinking + answer) | 16,378 + 4,096 = **20,474** | |
| `max_model_len` | 20,474 + 24,576 = **45,050** | Far above the longest prompt (~3.1k tokens), so prompts are never truncated |
| Samples per question | **1** | No best-of-n, no majority vote, no retries on a wrong answer |

**Where these values come from.** They are copied on purpose from the team's shared inference config,
`gurrt/automation/llama_inference.py` (a separate repository, not in this one). Every model in that comparison is run
with the same sampling settings and thinking budget. This is why `top_k` is 20 and not Google's 64, and why the
setting differs from lmms-eval's default (§12). The values are hard-coded constants in `vllm_eval.py`, cannot be
overridden from the command line, and were the same in every run (§8).

**Sampler implementation:** `VLLM_USE_FLASHINFER_SAMPLER=0`, so vLLM uses its PyTorch top-k/top-p sampler
(log line: "FlashInfer top-p/top-k sampling disabled"). This is a build workaround, not a change to the method:
FlashInfer would JIT-compile its sampler with the system CUDA 12.8 `nvcc`, which cannot target the RTX 5090 (SM 12.0).
Both samplers implement the same distribution.

**Other engine settings** (from the `non-default args` log line, the same in all runs): `tensor_parallel_size=2`,
`enable_prefix_caching=True` (the three questions of a video share the video prefix; this is an exact cache, so
outputs do not change), `limit_mm_per_prompt={"video": 1, "image": 1}`, `gpu_memory_utilization=0.92`, chunked
prefill on (vLLM default), and the TRITON_ATTN attention backend (forced by vLLM for Gemma 4's mixed head dimensions).

---

## 6. Thinking, the thinking budget, and what gets scored

Gemma 4 writes its reasoning between `<|channel>` and `<channel|>`, and then writes the answer.

**Budget enforcement (vLLM `ReasoningConfig` + `thinking_token_budget`):**
- `reasoning_start_str = "<|channel>"`
- `reasoning_end_str = "\n\nI have thought enough. Now I will give the final answer.<channel|>"`
- If the reasoning reaches 16,378 tokens, vLLM **forces** the end string. That closes the thought, and the model must
  then answer within the remaining 4,096 tokens. This is the same mechanism and message as `llama_inference.py`.

**Splitting** (`split_thinking`; output is decoded with `skip_special_tokens=False` so the delimiters are kept):
- **Thought closed:** the reasoning is the text before the last `<channel|>` (the leading `thought` label is removed),
  and the answer is the text after it. Chat-end markers (`<turn|>`, `<eos>`, `<end_of_turn>`) are removed from the answer.
- **Thought opened but never closed:** the answer is the empty string. That question is **scored wrong**.
- **No thought at all:** the whole output is the answer.

**Only the final answer is scored.** The reasoning is saved in the `reasoning` column for inspection, but the parser
**never sees it**. A correct letter that appears only inside the reasoning gets no credit.

**`hit_budget`** = 1 when the forced budget message appears in the reasoning, meaning the model used the full
16,378-token budget. These questions are kept and scored like any other (usually wrong, because a forced answer after
a cut-off thought is often poor). The share is reported in [§0](#0-results).

---

## 7. Scoring

### 7.1 Provenance

`videommmu_official.py` copies the prompt, parsing and judging functions of lmms-eval's `video_mmmu` task
(`lmms_eval/tasks/videommmu/{_default_template_yaml, utils.py}`, https://github.com/EvolvingLMMs-Lab/lmms-eval):

- Prompts, open-ended parsing, and judging (`eval_multi_choice`, `eval_open`) are the same at the original task commit
  `74b6d95ce7ca83c5ed0883581d91a19fcb5b66d0` (Feb 2025) and at `08affba133b97b19351897f4d42081ead2b3412e` (Oct 2026).
- **Multiple-choice parser:** we use the **original Video-MMMU parser** from `74b6d95ce7`. It was used for the
  published Video-MMMU leaderboard, and later moved unchanged to
  `mmmu_mcq_utils.parse_videommmu_multi_choice_response`. In April 2026 (`9ca4445d0c`), lmms-eval's default task
  switched to a stricter shared extractor. That extractor **abstains** on answers written like `**B. 3, 4**`, which is
  how Gemma usually states its choice, so it would score correct answers as wrong. **This is the one intentional
  scoring choice that differs from current lmms-eval HEAD.** It matches the parser behind the published numbers.
  Raw responses are saved, so the results can be re-scored with any other parser (§11).

Anyone can check the copy by comparing `videommmu_official.py` with those commits.

### 7.2 Multiple-choice (879 of 900 questions)

`parse_multi_choice_response(response, all_choices, index2ans)`, used unchanged:
1. Strip surrounding punctuation and pad with spaces.
2. Candidates are letters written as `X.` or `X:`. If none, `(X)`. If none, `X ` (letter then space).
3. If there are still none and the response is longer than 5 words: options whose **text** appears in the response.
4. With several candidates, pick the one whose match is **last** in the response.
5. No candidate gives `"No Answer Found."`, scored **wrong**. (The upstream comment mentions a random choice, but the
   code does not make one; we copy the code, and **no random guessing is used**.)
6. An empty response gives `"API Error"`, scored **wrong**.

Correct = the parsed letter equals the gold letter (exact match).

### 7.3 Open-ended (21 of 900: 20 `open` + 1 `"None"`)

`parse_open_response` → `eval_open`, used unchanged. Key sub-answers are taken after phrases like "is", "therefore",
"answer", and "=" (in the last sentence); numbers are extracted, rounded to 2 decimals, and compared as numbers;
strings are compared lowercase as substrings. The question is correct if any candidate matches the gold answer.

### 7.4 Our only additions to the scorer

| Change | Effect on the score |
|---|---|
| `strip_special_tokens` removes `<turn|>`, `<|turn>`, `<eos>`, `<end_of_turn>` before parsing | Neutral. These are template markers, not answer text. |
| Open-ended `"API Error"` (empty answer) is scored **wrong** and kept in the denominator. Upstream **drops** these from aggregation. | **Conservative**: it can only lower our score. |

There is no LLM judge, no manual grading, and no hand-corrected answers. Every `correct` value is produced by the code above.

### 7.5 Aggregation

- **Accuracy = correct ÷ 900**, a micro-average over all questions. Per-track accuracy = correct ÷ 300.
  Per-domain accuracy uses `SUB_CAT2DOMAIN` (copied from upstream) on the subject parsed from the video id.
- `summarize()` **re-scores every saved response from scratch** with the current `vm.score` before writing
  `_all_results.csv`. The `correct` and `parsed_pred` columns in the final file therefore always match the code above,
  even for CSVs written earlier.
- If any video were missing, the summary prints `WARNING: run incomplete - numbers below are NOT comparable`.
  The reported numbers come from a complete 300/300, 900/900 run.

---

## 8. How the run was executed (full history)

**No results were discarded, cherry-picked, or regenerated.** Each video id was generated exactly once, and its CSV
was written once. The runner skips any video whose CSV exists, and a CSV is only written after its whole batch
finishes, so a stopped run leaves no partial files.

| # | When (UTC, 2026-10-08) | Command | Log | Videos written |
|---|---|---|---|---|
| 1 | 11:49–11:56 | `vllm_eval.py --tp 2 --limit_videos 3` (smoke test) | `smoke.log` | 3 (`dev_Biology_3`, `dev_Clinical_Medicine_4`, `dev_Geography_5`) |
| 2 | 11:58–~12:07 | 2 shards, `--tp 2 --shard_id {0,1} --num_shards 2` | `logs/old/vllm_{0,1}_run1.log` | 32 (first batch of 16 per shard), then stopped by the operator |
| 3 | 12:13–end | same 2-shard command | `logs/vllm_{0,1}.log` | the remaining 265 (shard 0: 132, shard 1: 133) |

The counts add up: shard 0 = 2 smoke + 16 run-2 + 132 run-3 = 150, and shard 1 = 1 smoke + 16 run-2 + 133 run-3 = 150.
This matches the logs ("150 videos, 132 to run" and "150 videos, 133 to run").

**Code change between run 2 and run 3.** `vllm_eval.py` was last edited at 12:12, between the two runs. Compared with
the committed version (`534fef9`), the uncommitted changes are:
1. The video-decode cache (§4.1): performance only, same frames.
2. `traceback.print_exc()` on failures: logging only.
3. `VLLM_USE_FLASHINFER_SAMPLER=0` and the explicit `video_backend="opencv"` (the same as vLLM's default). Both were
   **already active in runs 1 and 2**: the "sampling disabled" line and `video_backend: 'opencv'` appear in `smoke.log`
   and in the run-2 logs.

**Proof that all three runs used the same configuration:** the `non-default args` line and the full
`Initializing a V1 LLM engine` config line are **identical** in `smoke.log`, `logs/old/vllm_0_run1.log` and
`logs/vllm_0.log` (model, dtype, `max_model_len=45050`, TP=2, prefix caching, 32 frames, OpenCV backend, reasoning
config, `seed=0`). Sampling parameters are hard-coded constants that no flag can change. Check with:

```bash
grep -h -o "non-default args.*" smoke.log logs/old/*.log logs/*.log | sort | uniq -c
```

**Sharding.** Two independent vLLM engines, each on a GPU pair: worker 0 with `CUDA_VISIBLE_DEVICES=0,1`, worker 1
with `CUDA_VISIBLE_DEVICES=2,3`, each with TP=2. Videos are sorted naturally and dealt round-robin
(`ids[shard_id::num_shards]`). Sharding decides only *which engine* runs a video, not *how*. The two halves run the
same model and settings. Each batch sends 16 videos × 3 questions = 48 requests to `llm.chat`.

**Why TP=2 when the 12B model fits on one 32 GB card:** a single 5090 would leave about 6 GB for KV cache, which is
too little for several 20k-token reasoning traces at once. A pair leaves about 14.8 GiB per GPU, or 198,685 tokens of
KV cache (log line). This affects speed only.

**Failures:** if a batch raises an error, its videos are retried one by one, and a video that still fails is not saved
(it is retried on the next launch). The logs contain **no** `NOT SAVED` or `failed` lines (see §0 for the final check).

---

## 9. Environment

| Item | Value |
|---|---|
| GPUs | 4 × NVIDIA GeForce RTX 5090 (32 GB, SM 12.0), driver 580.173.02 |
| Host | Vast.ai container, 128 CPU threads, 251 GB RAM |
| Python | 3.12.14 (uv-managed `.venv`, see `pyproject.toml` / `uv.lock`) |
| vLLM | **0.30.0+cu129** (official release wheel), V1 engine |
| PyTorch | 2.13.0+cu129 |
| Transformers | 5.19.0 |
| FlashInfer | 0.6.18.post1 (sampler disabled, see §5) |
| Triton | 3.7.1 |
| OpenCV (headless) | 5.0.0.93 (video decoding) |
| pandas / numpy / pyarrow | 3.0.6 / 2.3.5 / 25.0.1 |

Setup on a fresh machine: [`../setup_gemma_vast.sh`](../setup_gemma_vast.sh) (installs deps with `uv sync`, downloads
the model and dataset, and unzips the videos).

Harmless warnings that appear in the logs and do not affect results: "SM 12.x requires CUDA >= 12.9" (capability probe),
"Custom allreduce is disabled … P2P" (NCCL is used instead), `SiglipImageProcessorFast` deprecation, and Triton JIT
latency warnings.

---

## 10. Output files

`../results/videommmu_gemma4_12b_vllm/<video_id>.csv`: one file per video, with 3 rows (Perception, Comprehension,
Adaptation). `_all_results.csv` holds all 900 rows, re-scored by `summarize()`.

| Column | Meaning |
|---|---|
| `id`, `track` | Video id and track |
| `subject`, `domain` | Subject parsed from the id, and the official domain mapping |
| `question_type`, `qa_type` | Taken from the dataset |
| `answer` | Gold answer from the dataset |
| `parsed_pred` | What the official parser took out of `response` (a letter, `No Answer Found.`, `API Error`, or a JSON list for open-ended questions) |
| `correct` | 0/1 from the official judge |
| `input_tokens` | Prompt tokens, including video and image tokens |
| `output_tokens` | Generated tokens (reasoning + answer) |
| `hit_budget` | 1 if the 16,378-token thinking budget was used up and the end of thinking was forced |
| `used_question_image` | 1 if the Adaptation image was attached |
| `response` | **The scored text**: the model's final answer after thinking |
| `reasoning` | The model's thinking trace (not scored) |

Every number in this README can be recomputed from these columns.

---

## 11. How to reproduce or audit

```bash
# setup (once)
HF_TOKEN=hf_xxx bash setup_gemma_vast.sh
cd video_mmmu_gemma

# smoke test
uv run --no-sync python vllm_eval.py --tp 2 --limit_videos 3

# full run as executed (4 x RTX 5090, two TP=2 engines)
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$((2*i)),$((2*i+1)) nohup uv run --no-sync python vllm_eval.py \
      --tp 2 --shard_id $i --num_shards 2 > logs/vllm_$i.log 2>&1 &
done

# final summary (re-scores every saved response)
uv run --no-sync python vllm_eval.py --summary_only
# or
uv run --no-sync python score.py --out_dir ../results/videommmu_gemma4_12b_vllm --save
```

**Audit without a GPU.** Re-scoring needs only the CSVs, the parquet files and `videommmu_official.py`. Things to check:
- Re-score `response` with another parser (for example lmms-eval's current strict extractor) and compare.
- Read `reasoning` and `response` for any question and confirm `parsed_pred` and `correct` by hand.
- Check that each per-video CSV has exactly 3 rows, and that the 300 ids match the dataset (`--summary_only` prints both counts).
- Check the run history and configuration claims in §8 against the logs.

**Expect variation on reruns.** Decoding uses sampling at temperature 1.0. The fixed seed makes each request's random
draws repeatable, but vLLM does not promise bit-identical results across different batch compositions, GPU counts, or
kernel versions. A rerun can therefore change individual answers and move accuracy by a small amount. The reported
figure is **one sample per question**, not an average over several runs.

---

## 12. Limitations and comparability: read before comparing numbers

1. **Not lmms-eval's default generation setting.** lmms-eval's `video_mmmu` task uses `max_new_tokens=1024` with no
   thinking. Here, thinking is on with a 16k-token budget and sampled decoding (§5), to match the team's shared config.
   Prompts, data and scoring are official; **generation is not**. Compare these numbers with other models run under
   the same `llama_inference.py` settings, or label the difference clearly when comparing with leaderboard numbers.
2. **MCQ parser** = the original Video-MMMU / leaderboard parser, not lmms-eval's post-April-2026 strict extractor (§7.1).
3. **Adaptation image given separately as well as in the video** (§4.3), using the dataset's own `image` field.
   Setups that rely only on the video frames may score differently on Adaptation.
4. **32 frames per video** (Gemma 4's native video setting). Long lecture videos are heavily subsampled. Other frame
   counts were not tested.
5. **One sampled run**, so no confidence interval. With 300 questions per track, a single track's accuracy has a
   sampling standard error of roughly ±2.9 points (binomial, near 50%). The 900-question overall figure has roughly ±1.7.
6. **Dataset quirks** (§3) are kept as in the official data, so a few items are effectively unanswerable or scored loosely.
7. **Hard-coded settings.** The hard-coded settings were not tuned on this benchmark. No prompt, parser, budget or
   frame count was chosen by looking at test accuracy. All of them come from the official task, the model card, or the
   shared team config.

---

## 13. FAQ for reviewers

**Were any questions skipped or excluded?** No. All 900 are in the denominator. Empty or unparsable answers count as wrong.

**Was the model given the answer options?** Yes, for every multiple-choice question, in the official format. The one
exception is `validation_Accounting_13`, mislabelled in the dataset itself (§3).

**Is the reasoning trace searched for the answer?** No. Only the text after the thought closes is parsed.

**Was any prompt engineering done for Gemma?** No. The prompt is the official lmms-eval prompt, unchanged, with no
added format instructions. The only Gemma-specific item is the `enable_thinking` flag of its own chat template.

**Was the scorer changed to favour Gemma?** The scorer is upstream's original Video-MMMU parser (§7.1). Its use is
disclosed, and it is the parser behind the published leaderboard. Our one change to aggregation (counting empty
open-ended answers as wrong) can only *lower* the score.

**Were runs repeated and the best one kept?** No. Each video was generated once (§8). Nothing was deleted and regenerated.

**Was the code changed during the run?** Yes, once, between run 2 and run 3. Only performance and logging changed;
the logs show the engine configuration was identical before and after (§8).

**Were the frames the model saw the right ones?** 32 evenly spaced frames from `dataset/videommmu/<Domain>/<id>.mp4`,
always including the first and last frame. The question-only videos and macOS metadata files can never be picked (§3).

**Is the model quantized or modified?** No. It is the BF16 Hub snapshot `707f0a3b…`, loaded unchanged.
