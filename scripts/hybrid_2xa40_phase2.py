"""Phase 2: Load 1.3B, resume from checkpoint, finish generation, save video."""
import math, os, sys, time, torch, torch.cuda.amp as amp
from contextlib import contextmanager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOTAL_STEPS = 50
CKPT_PATH = "/workspace/results/hybrid_checkpoint.pt"
SAVE_FILE = os.environ.get("SAVE_FILE", "/workspace/results/hybrid_2xA40.mp4")

from wan.text2video import WanT2V
from wan.configs.wan_t2v_1_3B import t2v_1_3B
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video

print(f"=== Phase 2: 1.3B (remaining steps) ===")

# Load checkpoint
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
latents = ckpt['latents']
context = ckpt['context']
context_null = ckpt['context_null']
global_step = ckpt['global_step']
seq_len = ckpt['seq_len']
target_shape = ckpt['target_shape']
gen_time_14b = ckpt['gen_time_14b']
steps_1_3b = TOTAL_STEPS - global_step
print(f"  Loaded checkpoint: step {global_step}, {steps_1_3b} steps remaining")
print(f"  14B phase took: {gen_time_14b:.1f}s")

# Load 1.3B
print("  Loading 1.3B model...")
device = torch.device("cuda:0")
model = WanT2V(config=t2v_1_3B, checkpoint_dir="Wan2.1-T2V-1.3B", device_id=0, rank=0,
               t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=True)

# Move tensors to device
latents = latents.to(device)
context = [c.to(device) for c in context]
context_null = [c.to(device) for c in context_null]

# Restore scheduler
scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=model.num_train_timesteps, shift=1, use_dynamic_shifting=False)
scheduler.set_timesteps(TOTAL_STEPS, device=device, shift=5.0)

ss = ckpt['scheduler_state']
scheduler.model_outputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in ss['model_outputs']]
scheduler.lower_order_nums = ss['lower_order_nums']
scheduler._step_index = ss['_step_index']
scheduler._begin_index = ss['_begin_index']
scheduler.sigmas = ss['sigmas']
scheduler.timesteps = ss['timesteps'].to(device)
scheduler.num_inference_steps = ss['num_inference_steps']
if 'timestep_list' in ss:
    scheduler.timestep_list = ss['timestep_list']
if 'last_sample' in ss:
    scheduler.last_sample = ss['last_sample'].to(device) if isinstance(ss['last_sample'], torch.Tensor) else ss['last_sample']
if 'this_order' in ss:
    scheduler.this_order = ss['this_order']

timesteps = scheduler.timesteps
arg_c = {'context': context, 'seq_len': seq_len}
arg_null = {'context': context_null, 'seq_len': seq_len}

@contextmanager
def noop():
    yield

model.model.to(device)
no_sync = getattr(model.model, 'no_sync', noop)

print(f"  Running {steps_1_3b} steps of 1.3B...")
t_start = time.time()

with amp.autocast(dtype=model.param_dtype), torch.no_grad(), no_sync():
    for i, t in enumerate(timesteps[global_step:global_step + steps_1_3b]):
        latent_input = [latents]
        ts = torch.stack([t])
        noise_cond = model.model(latent_input, t=ts, **arg_c)[0]
        noise_uncond = model.model(latent_input, t=ts, **arg_null)[0]
        noise_pred = noise_uncond + 5.0 * (noise_cond - noise_uncond)
        temp = scheduler.step(noise_pred.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False)[0]
        latents = temp.squeeze(0)
        if (i+1) % 10 == 0:
            print(f"    Step {i+1}/{steps_1_3b}")

gen_time_1_3b = time.time() - t_start
print(f"  1.3B done: {gen_time_1_3b:.1f}s ({gen_time_1_3b/steps_1_3b:.2f}s/step)")

# VAE decode
print("  Decoding video...")
t_decode = time.time()
with torch.no_grad():
    videos = model.vae.decode([latents])

os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
cache_video(videos[0][None], save_file=SAVE_FILE, fps=16, nrow=1, normalize=True, value_range=(-1, 1))
decode_time = time.time() - t_decode

total = gen_time_14b + gen_time_1_3b + decode_time
print(f"\n{'='*60}")
print(f"  HYBRID COMPLETE on 2xA40")
print(f"  14B ({TOTAL_STEPS - steps_1_3b} steps): {gen_time_14b:.1f}s")
print(f"  1.3B ({steps_1_3b} steps): {gen_time_1_3b:.1f}s")
print(f"  VAE decode: {decode_time:.1f}s")
print(f"  Total: {total:.1f}s")
print(f"  Video: {SAVE_FILE}")
print(f"{'='*60}")
