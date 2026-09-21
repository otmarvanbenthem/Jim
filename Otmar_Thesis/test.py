import os, sys
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false"
import jax, jax.numpy as jnp

mode, x64 = sys.argv[1], sys.argv[2] == "x64"
jax.config.update("jax_enable_x64", x64)
dt = jnp.float64 if x64 else jnp.float32

for i in range(300):
    if mode == "fft":
        out = jnp.fft.rfft(jnp.ones(16384, dtype=dt) * i)
    elif mode == "matmul":
        x = jnp.ones((1000, 1000), dtype=dt)
        out = (x @ x).sum()
    else:
        out = jnp.exp(jnp.linspace(0, 1, 100000, dtype=dt) * i).sum()
    jax.block_until_ready(out)
print(mode, "x64" if x64 else "f32", "ok")