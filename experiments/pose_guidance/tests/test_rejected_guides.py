import unittest
import numpy as np
from experiments.pose_guidance.pose_geometry import AnchorClash
from promera_angles.batches import accepted_batch
from promera_angles.geometry import placement


class GuideTests(unittest.TestCase):
    def test_placement(self):
        frame = dict(origin=[0, 0, 0], normal=[0, 0, 1], tangent=[1, 0, 0])
        shape = dict(local_xyz=[[5, 0, 0], [6, 1, 1], [7, 2, 3], [-3, 2, 0]])
        args = (frame, shape, [0, 20], 5)
        original, first = placement(*args)
        clear, record = placement(*args, target_anchors=original + 100)
        np.testing.assert_array_equal(original, clear)
        shifted, retried = placement(*args, target_anchors=original)
        self.assertEqual(first["ray"], retried["ray"])
        self.assertGreater(retried["roll_attempts"], 1)
        self.assertGreater(np.linalg.norm(shifted[:, None] - original, axis=-1).min(), 1)
        tip = np.array(first["ray"]) * 6
        with self.assertRaises(AnchorClash):
            placement(frame, dict(local_xyz=[[0, 0, 0]]), [0, 20], 5,
                      target_anchors=[tip])

    def test_rejection_is_local(self):
        rejected = {"record_status": "rejected_guide"}
        batch = dict(angle_record=[{}, rejected, {}], name=["a", "b", "c"],
                     backbone_idx=np.array([41, 42, 43]))
        kept = accepted_batch(batch)
        self.assertEqual(kept["name"], ["a", "c"])
        self.assertEqual(kept["backbone_idx"].tolist(), [41, 43])
        self.assertIs(accepted_batch(kept), kept)
        self.assertIsNone(accepted_batch({"angle_record": [rejected]}))
