"""Group contours from an edge map by proximity."""

from typing import List

import numpy as np
import cv2

from ...config import Config
from ..processor import Processor


class ContourGrouper(Processor):
    """Group nearby contours into logical object groups."""

    def __init__(self, config: Config) -> None:
        if not isinstance(config, Config):
            raise TypeError(f"config must be a Config instance, got {type(config)}")
        self.config = config

    def apply(self, edge_map: np.ndarray) -> List[List[np.ndarray]]:
        """Return contour groups found in `edge_map` (empty if none)."""

        contours, _ = cv2.findContours(
            edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        bboxes = [cv2.boundingRect(c) for c in contours]
        proximity = self.config.grouping_proximity
        used = [False] * len(contours)
        groups = []

        for i in range(len(contours)):
            if used[i]:
                continue

            group = [contours[i]]
            used[i] = True
            gx, gy, gw, gh = bboxes[i]

            for j in range(i + 1, len(contours)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = bboxes[j]
                # Merge if bounding boxes are within proximity pixels of each other
                if (
                    x2 <= gx + gw + proximity
                    and x2 + w2 >= gx - proximity
                    and y2 <= gy + gh + proximity
                    and y2 + h2 >= gy - proximity
                ):
                    group.append(contours[j])
                    used[j] = True
                    # Expand the group bounding box to encompass the new contour
                    nx = min(gx, x2)
                    ny = min(gy, y2)
                    gw = max(gx + gw, x2 + w2) - nx
                    gh = max(gy + gh, y2 + h2) - ny
                    gx, gy = nx, ny

            groups.append(group)

        return groups

    # Backwards-compatible alias
    def group(self, edge_map: np.ndarray) -> List[List[np.ndarray]]:
        return self.apply(edge_map)
