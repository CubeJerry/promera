import unittest
from promera_angles.report import tables


class EndpointCounts(unittest.TestCase):
    def test_parent_denominators_and_missing_normal(self):
        def row(i, angle=10, **kw):
            return dict(dict(arm="guided", replicate=1, backbone=i, spatial_deg=angle,
                             file_exists=True, pose_class="cdr_driven", cdr_patch_contact=True), **kw)
        parents = [row(0), row(1, None), row(2, 60), row(3, None, file_exists=False, pose_class=None)]
        folds = [row(0), row(0), row(2)]
        _, _, totals, effects = tables(parents, folds, [0, 20])
        pose = totals.set_index("endpoint").loc["pose"]
        self.assertEqual((pose.requested, pose.generated, pose.measurable), (4, 3, 2))
        self.assertEqual((pose.qualified, pose.qualified_with_refolds, pose.recovered, pose.final_hit),
                         (1, 1, 1, 2))
        self.assertTrue(effects.empty)
