# Hands-on labs: build the core of each model yourself

Reading a guide tells you what a model does. These labs make you write it.
Each lab is one Python file with function signatures, docstrings that spell out
the maths with tensor shapes, graded hints, and `raise NotImplementedError  # TODO`
where your code goes. A checker compares your code with independent references:
PyTorch's own implementations (`F.scaled_dot_product_attention`,
`nn.MultiheadAttention` with copied weights, `F.conv2d`, `F.grid_sample`), closed-form
maths, the worked examples from the guides, and brute-force Python loops.

You need only PyTorch (2.1 or newer) and a CPU. No downloads, no GPU.

## Run the checks

From the repository root:

```bash
python labs/check.py                 # check every lab
python labs/check.py 3               # check lab 3 only (also: 03, lab03, or several numbers)
python labs/lab03_gpt_generate.py    # the same, by running the lab file itself
python labs/check.py 3 -v            # also print the full traceback of a failing step
python labs/check.py --solutions     # run the checks against the reference solutions
```

Every lab is split into steps, usually one per function, and each step prints one line:

| Status | Meaning |
|--------|---------|
| `PASS` | your code matches the reference |
| `TODO` | your code still raises `NotImplementedError`; the line names the function to write next |
| `FAIL` | your code runs but gives a wrong answer, or crashes; the message shows expected vs got (worst entry, shapes) and often a hint about the likely mistake |

Everything is seeded, so the same code always gives the same result. Each check runs in
well under a second, except the two training steps (labs 06 and 10), which train a tiny
MLP for about 5-20 s on a CPU.

## Recommended order

The labs follow the main track of the repository. Do 01-03 in order, right after the
[Transformer guide](../transformer/transformer_guide.html) and before the tutorial
[Build an LLM](../build-llm/build_llm_guide.html). After that, do each later lab right
after its module guide; each one only assumes 01-03 (lab 05 also uses the ideas of 04,
lab 10 those of 06).

| Lab | File | What you build | Read with | Time |
|-----|------|----------------|-----------|------|
| 01 | [`lab01_attention.py`](lab01_attention.py) | Causal mask, stable softmax, scaled dot-product attention from matmul + softmax, head split/merge, multi-head attention (checked against `nn.MultiheadAttention`) | [step-by-step page](../transformer/transformer_step_by_step.html); [transformer guide](../transformer/transformer_guide.html): [2a worked example](../transformer/transformer_guide.html#worked-example), [2c-2e multi-head](../transformer/transformer_guide.html#mha), [2.3 causal mask](../transformer/transformer_guide.html#causal-mask) | 45-60 min |
| 02 | [`lab02_transformer_block.py`](lab02_transformer_block.py) | LayerNorm and RMSNorm, GELU MLP and SwiGLU, a pre-norm block, a closed-form parameter count (GPT-2 block, Llama-2-7B block) | [step-by-step page](../transformer/transformer_step_by_step.html); [architecture](../transformer/transformer_guide.html#architecture), [what changed since 2017](../transformer/transformer_guide.html#modern); preview, covered in the next tutorial: [Build an LLM: model](../build-llm/build_llm_guide.html#model) | 60 min |
| 03 | [`lab03_gpt_generate.py`](lab03_gpt_generate.py) | A tiny GPT forward pass, greedy and top-k sampling, generation with a KV cache that must equal full recompute (including chunked prefill) | [step-by-step page](../transformer/transformer_step_by_step.html); [2.4 generation with a KV cache](../transformer/transformer_guide.html#kv-cache); preview, covered in the next tutorial: [Build an LLM: sampling](../build-llm/build_llm_guide.html#sampling), [KV cache](../build-llm/build_llm_guide.html#kvcache) | 60-90 min |
| 04 | [`lab04_vit.py`](lab04_vit.py) | Patchify by reshape and by Conv2d (they must agree), unpatchify, CLS token, position embeddings, the ViT forward pass, a shape tracker | [ViT guide](../vit/vit_guide.html): 2.1 inference pipeline (shape flow), 2.2 patch embedding, 2.3 position embeddings, 2.5 where the parameters and compute go | 45-60 min |
| 05 | [`lab05_vlm_fusion.py`](lab05_vlm_fusion.py) | Image tokens per resolution, the 2x2 pixel-shuffle merge, an MLP projector from vision width to LLM width, the joint image + text sequence and its causal / image-bidirectional mask (a partial prefix-LM) | [VLM guide](../vlm/vlm_guide.html): the key idea, 1.1 shape flow, 2 the fusion step, 4 image token reduction | 45-60 min |
| 06 | [`lab06_diffusion.py`](lab06_diffusion.py) | Noise schedule, closed-form forward noising q(x_t \| x_0), the epsilon-prediction loss, a training loop, a DDIM sampler; trains on 8 Gaussians on a ring | [Diffusion guide](../diffusion/diffusion_guide.html): 1 forward process, 3.1 DDIM, 5 code walkthrough | 60-90 min |
| 07 | [`lab07_mamba_scan.py`](lab07_mamba_scan.py) | Discretization (simple and ZOH), the sequential selective scan, a log-depth parallel scan, the one-token recurrent step; all three must agree | [Mamba guide](../mamba/mamba_guide.html): 1 selective SSM, 2 two modes of computation | 60-90 min |
| 08 | [`lab08_moe_router.py`](lab08_moe_router.py) | Softmax and sigmoid top-k routers, the load-balancing loss, per-expert loop vs sort/scatter dispatch, capacity factor with token dropping | [MoE guide](../moe/moe_guide.html): [2 the router](../moe/moe_guide.html#router), [5 inference](../moe/moe_guide.html#inference) (5.4 capacity, 5.5 grouped GEMM) | 60-90 min |
| 09 | [`lab09_bev_projection.py`](lab09_bev_projection.py) | Camera intrinsics and extrinsics, projecting 3-D points to pixels, BEV pillar points, reference points with a visibility mask, bilinear sampling with `grid_sample`, averaging over views | [BEV guide](../bev/bev_guide.html): 3 camera projection (with the worked example), 4 spatial cross-attention | 60-90 min |
| 10 | [`lab10_flow_matching.py`](lab10_flow_matching.py) | The linear interpolation path, the velocity target, training a velocity MLP on two-mode action chunks, an Euler sampler, chunking and receding-horizon execution | [VLA guide](../vla/vla_guide.html): 2 flow matching from scratch; [Diffusion guide](../diffusion/diffusion_guide.html) 3.2 flow matching | 45-60 min |

Robot foundation models ([robot-fm](../robot-fm/)) have no separate lab: GR00T N1's
"System 1" is a flow-matching action head like the one in lab 10, conditioned on a VLM
like the one in lab 05.

## How to work through a lab

1. Read the guide section listed above, then the docstring at the top of the lab file.
   It writes the maths out step by step with the shape of every tensor.
2. Implement the functions in file order. Each function's docstring states its inputs,
   outputs and conventions (for example "True = may attend").
3. After each function, run the lab file. Its step should turn from `TODO` to `PASS`.
4. When everything passes, try the variations suggested in the docstrings and in the
   guide's exercises; rerun the checks to make sure you broke nothing.

To start a lab again from scratch: `git checkout labs/lab03_gpt_generate.py`.

## How to get unstuck

Go down this list in order and stop as soon as you can continue:

1. **Read the FAIL message.** It names the quantity that is wrong, shows the worst entry
   (got vs expected), and for common mistakes a `hint:` line (for example "you
   reshaped [B, N, d] straight to [B, h, N, d_k]" or "this is POST-norm").
2. **Check shapes.** Print `x.shape` after every line, and compare with the shapes in the
   docstring. Most bugs in these labs are a transpose or a missing broadcast dimension.
3. **Rerun with `-v`** to see the full traceback when your code crashes.
4. **Read Hint 1** in the comments above the `TODO`. Only if you are still stuck,
   Hint 2, then Hint 3 (Hint 3 is close to the answer).
5. **Compare with the reference solution** in [`solutions/`](solutions/). Look only at the
   function you are stuck on, close it, and write it again yourself.
6. The full models in the module folders (`../transformer/transformer.py`,
   `../mamba/mamba.py`, `../bev/bev.py`, ...) show the same pieces in context.

Do not edit `check.py` to make a step pass; the checks are the specification.

## Files

```
labs/
  README.md                  this page
  check.py                   the checker (all check functions live here)
  lab01_attention.py ...     the ten learner skeletons
  lab10_flow_matching.py
  solutions/                 complete reference solutions with the same API
```
