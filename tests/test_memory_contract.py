"""Static regression tests for Promera's inference-memory lifecycle.

These tests deliberately avoid importing the GPU stack so they run quickly on a
plain GitHub runner while still protecting the code paths that previously retained
hundreds of CUDA trajectory frames.
"""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MemoryContractTests(unittest.TestCase):
    def test_sampler_does_not_use_mutable_extra_defaults(self):
        source = (ROOT / "promera" / "diffusion" / "sampler.py").read_text()
        self.assertNotIn("extra={}", source)
        self.assertIn("record_history=None", source)
        self.assertIn('PROMERA_RECORD_TRAJECTORY", "0"', source)

    def test_trajectory_frames_are_cpu_backed_and_opt_in(self):
        source = (ROOT / "promera" / "diffusion" / "sampler.py").read_text()
        self.assertIn('extra["traj"].append(x0.detach().cpu())', source)
        self.assertIn('extra["noisy"].append(x.detach().cpu())', source)
        self.assertIn("if extra is not None:", source)
        self.assertIn('final_cpu = noisy_batch["coords"].detach().cpu()', source)

    def test_predict_batches_have_explicit_cleanup_hooks(self):
        source = (ROOT / "promera" / "model" / "template.py").read_text()
        self.assertIn("def _release_inference_memory", source)
        self.assertIn("gc.collect()", source)
        self.assertIn("torch.cuda.empty_cache()", source)
        self.assertIn("def on_predict_batch_start", source)
        self.assertIn("def on_predict_batch_end", source)
        self.assertIn("def on_predict_epoch_end", source)


if __name__ == "__main__":
    unittest.main()
