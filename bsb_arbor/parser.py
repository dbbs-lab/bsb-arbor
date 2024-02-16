import typing

from bsb import config
from bsb.morphologies import Morphology
from bsb.storage._files import FileDependency
from bsb.morphologies.parsers.parser import MorphologyParser
import arbor
import numpy as np

@config.node
class ArborMorphologyParser(MorphologyParser, classmap_entry="arbor"):
    centering = config.attr(type=bool, default=True)

    def parse(self, file: typing.Union["FileDependency", str]) -> Morphology:

        arb_m = arbor.load_whatever(file)

        decor = arbor.decor()
        morpho_roots = set(
            i for i in range(arb_m.num_branches) if arb_m.branch_parent(i) == 4294967295
        )
        root_prox = [r[0].prox for r in map(arb_m.branch_segments, morpho_roots)]
        center = np.mean([[p.x, p.y, p.z] for p in root_prox], axis=0)
        parent = None
        roots = []
        stack = []
        cable_id = morpho_roots.pop()
        while True:
            segments = arb_m.branch_segments(cable_id)
            if not segments:
                branch = self.branch_cls([], [], [], [])
            else:
                # Prepend the proximal end of the first segment to get [p0, p1, ..., pN]
                x = np.array([segments[0].prox.x] + [s.dist.x for s in segments])
                y = np.array([segments[0].prox.y] + [s.dist.y for s in segments])
                z = np.array([segments[0].prox.z] + [s.dist.z for s in segments])
                r = np.array(
                    [segments[0].prox.radius] + [s.dist.radius for s in segments]
                )
                if self.centering:
                    x -= center[0]
                    y -= center[1]
                    z -= center[2]
                branch = self.branch_cls(x, y, z, r)
            branch._cable_id = cable_id
            if parent:
                parent.attach_child(branch)
            else:
                roots.append(branch)
            children = arb_m.branch_children(cable_id)
            if children:
                stack.extend((branch, child) for child in reversed(children))
            if stack:
                parent, cable_id = stack.pop()
            elif not morpho_roots:
                break
            else:
                parent = None
                cable_id = morpho_roots.pop()

        morpho = self.cls(roots)
        branches = morpho.branches
        branch_map = {branch._cable_id: branch for branch in branches}
        labels = self.labels or arbor.label_dict()
        cc = arbor.cable_cell(arb_m, labels, decor)
        for label in labels:
            if "excl:" in label or label == "all":
                continue
            label_cables = cc.cables(f'"{label}"')
            for cable in label_cables:
                cable_id = cable.branch
                branch = branch_map[cable_id]
                if cable.dist == 1 and cable.prox == 0:
                    branch.label([label])
                else:
                    prox_index = branch.get_arc_point(cable.prox, eps=1e-7)
                    if prox_index is None:
                        prox_index = branch.introduce_arc_point(cable.prox)
                    dist_index = branch.get_arc_point(cable.dist, eps=1e-7)
                    if dist_index is None:
                        dist_index = branch.introduce_arc_point(cable.dist)
                    mask = np.array(
                        [False] * prox_index
                        + [True] * (dist_index - prox_index + 1)
                        + [False] * (len(branch) - dist_index - 1)
                    )
                    branch.label([label], mask)

        morpho.optimize()
        return morpho
