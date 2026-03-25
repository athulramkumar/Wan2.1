"""Phase 1: Run 14B steps with USP on 2xA40, skip VAE to save memory, save checkpoint."""
import math, os, sys, time, gc, torch, torch.distributed as dist, torch.cuda.amp as amp
from contextlib import contextmanager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STEPS_14B = int(os.environ.get("STEPS_14B", "15"))
TOTAL_STEPS = 50
PROMPT = "The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
NEG = "\u8272\u8c03\u8273\u4e3d\uff0c\u8fc7\u66dd\uff0c\u9759\u6001\uff0c\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05\uff0c\u5b57\u5e55\uff0c\u98ce\u683c\uff0c\u4f5c\u54c1\uff0c\u753b\u4f5c\uff0c\u753b\u9762\uff0c\u9759\u6b62\uff0c\u6574\u4f53\u53d1\u7070"
SEED = 42
CKPT_PATH = "/workspace/results/hybrid_checkpoint.pt"

rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

if world_size > 1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    init_distributed_environment(rank=rank, world_size=world_size)
    initialize_model_parallel(sequence_parallel_degree=world_size, ring_degree=1, ulysses_degree=world_size)

device = torch.device(f"cuda:{local_rank}")

if rank == 0:
    print(f"=== Phase 1: 14B x{STEPS_14B} steps (USP, {world_size} GPUs) ===")

# Load T5 on CPU for encoding (before loading DiT to save peak memory)
if rank == 0:
    print("  Encoding text on CPU...")
from wan.modules.t5 import T5EncoderModel
from wan.configs.wan_t2v_14B import t2v_14B

t5 = T5EncoderModel(text_len=t2v_14B.text_len, dtype=t2v_14B.t5_dtype,
                     device=torch.device('cpu'),
                     tokenizer_path=os.path.join("Wan2.1-T2V-14B", t2v_14B.t5_tokenizer),
                     checkpoint_path=os.path.join("Wan2.1-T2V-14B", t2v_14B.t5_checkpoint))
context = t5([PROMPT], torch.device('cpu'))
context_null = t5([NEG], torch.device('cpu'))
del t5; gc.collect()

# Move to GPU
context = [c.to(device) for c in context]
context_null = [c.to(device) for c in context_null]

if rank == 0:
    print(f"  Text encoded. GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f}GB")

# Load DiT model directly (skip VAE to save ~7GB)
if rank == 0:
    print("  Loading 14B DiT (no VAE)...")
from wan.modules.model import WanModel

model = WanModel.from_pretrained("Wan2.1-T2V-14B", **{
    k: v for k, v in t2v_14B.items()
    if k in ('dim', 'ffn_dim', 'freq_dim', 'num_heads', 'num_layers',
             'window_size', 'qk_norm', 'cross_attn_norm', 'eps')
})
model = model.to(dtype=t2v_14B.param_dtype)

# Apply USP monkey-patching
if world_size > 1:
    from wan.distributed.xdit_context_parallel import usp_dit_forward, usp_attn_forward
    for block in model.blocks:
        block.self_attn.forward = lambda *args, _block=block, **kwargs: usp_attn_forward(_block.self_attn, *args, **kwargs)
    model.forward = lambda *args, _m=model, **kwargs: usp_dit_forward(_m, *args, **kwargs)

model.to(device)
model.eval()

if rank == 0:
    print(f"  DiT loaded. GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f}GB")
    print(f"  Free: {(torch.cuda.get_device_properties(device).total_mem - torch.cuda.memory_allocated(device))/1e9:.1f}GB")

if world_size > 1:
    dist.barrier()

# Setup scheduler and noise
vae_stride = t2v_14B.vae_stride
patch_size = t2v_14B.patch_size
z_dim = 16  # Wan2.1 T2V z_dim
target_shape = (z_dim, (81-1)//vae_stride[0]+1, 480//vae_stride[1], 832//vae_stride[2])

sp_size = world_size if world_size > 1 else 1
seq_len = math.ceil((target_shape[2]*target_shape[3])/(patch_size[1]*patch_size[2])*target_shape[1])
if sp_size > 1:
    seq_len = math.ceil(seq_len / sp_size) * sp_size

from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=t2v_14B.num_train_timesteps, shift=1, use_dynamic_shifting=False)
scheduler.set_timesteps(TOTAL_STEPS, device=device, shift=5.0)
timesteps = scheduler.timesteps

seed_g = torch.Generator(device=device)
seed_g.manual_seed(SEED)
latents = torch.randn(*target_shape, dtype=torch.float32, device=device, generator=seed_g)

arg_c = {'context': context, 'seq_len': seq_len}
arg_null = {'context': context_null, 'seq_len': seq_len}

@contextmanager
def noop():
    yield

no_sync = getattr(model, 'no_sync', noop)

if rank == 0:
    print(f"  Running {STEPS_14B} steps...")
t_start = time.time()

with amp.autocast(dtype=t2v_14B.param_dtype), torch.no_grad(), no_sync():
    for i, t in enumerate(timesteps[:STEPS_14B]):
        noise_cond = model([latents], t=torch.stack([t]), **arg_c)[0]
        noise_uncond = model([latents], t=torch.stack([t]), **arg_null)[0]
        noise_pred = noise_uncond + 5.0 * (noise_cond - noise_uncond)
        temp = scheduler.step(noise_pred.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False)[0]
        latents = temp.squeeze(0)
        if rank == 0 and (i+1) % 5 == 0:
            print(f"    Step {i+1}/{STEPS_14B} ({(time.time()-t_start)/(i+1):.1f}s/step)")

gen_time = time.time() - t_start
if rank == 0:
    print(f"  14B done: {gen_time:.1f}s ({gen_time/STEPS_14B:.2f}s/step)")

# Save checkpoint (rank 0 only)
if rank == 0:
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    sched_state = {
        'model_outputs': [x.cpu() if isinstance(x, torch.Tensor) else x for x in scheduler.model_outputs],
        'lower_order_nums': scheduler.lower_order_nums,
        '_step_index': scheduler._step_index, '_begin_index': scheduler._begin_index,
        'sigmas': scheduler.sigmas, 'timesteps': scheduler.timesteps.cpu(),
        'num_inference_steps': scheduler.num_inference_steps,
    }
    if hasattr(scheduler, 'timestep_list'): sched_state['timestep_list'] = list(scheduler.timestep_list)
    if hasattr(scheduler, 'last_sample'): sched_state['last_sample'] = scheduler.last_sample.cpu() if isinstance(scheduler.last_sample, torch.Tensor) else scheduler.last_sample
    if hasattr(scheduler, 'this_order'): sched_state['this_order'] = scheduler.this_order

    torch.save({
        'latents': latents.cpu(), 'context': [c.cpu() for c in context],
        'context_null': [c.cpu() for c in context_null],
        'scheduler_state': sched_state, 'global_step': STEPS_14B,
        'seq_len': seq_len, 'target_shape': target_shape, 'gen_time_14b': gen_time,
    }, CKPT_PATH)
    print(f"  Checkpoint saved: {CKPT_PATH}")

if world_size > 1:
    dist.barrier()
    dist.destroy_process_group()
