"""JAX GPU smoke test: device visible, matmul runs on GPU, DiffLogic forward+grad compiles."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax, jax.numpy as jnp, numpy as np
print("jax", jax.__version__, "devices", jax.devices())
assert jax.devices()[0].platform == "gpu", "no GPU backend"
x = jnp.ones((4096, 4096)); t = time.perf_counter(); y = (x @ x).block_until_ready(); print("matmul ok", float(y[0, 0]), f"{time.perf_counter()-t:.2f}s incl. compile")
from ca.difflogic import init_logits, make_wiring, net, hard_lut
w = [jnp.asarray(a) for a in make_wiring(np.random.default_rng(0), 10, [32, 2])]
P = init_logits(jax.random.PRNGKey(0), [32, 2])
g = jax.grad(lambda P: net(P, w, jnp.ones((5, 10)) * 0.5).sum())(P)
assert all(bool(jnp.isfinite(a).all()) for a in g)
print("difflogic grad ok; hard lut shape", hard_lut(P, w, 10).shape)
