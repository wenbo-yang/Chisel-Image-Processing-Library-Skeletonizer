"""Contour finding and proximity-based grouping of edge fragments."""

from typing import List

import numpy as np
import cv2

from .config import Config


class ContourGrouper:
    """Groups raw contours from an edge map into logical objects by proximity.

    Contours whose bounding boxes are within config.grouping_proximity pixels
    of each other are merged into a single group, each group representing one
    distinct object in the image.

    TODO: Replace proximity-union with Jaccard Index grouping once the
    algorithm is decided:
        jaccard = intersection_area / union_area
        Merge contours where jaccard >= config.grouping_threshold
    """

    def __init__(self, config: Config) -> None:
        if not isinstance(config, Config):
            raise TypeError(f"config must be a Config instance, got {type(config)}")
        self.config = config

    def group(self, edge_map: np.ndarray) -> List[List[np.ndarray]]:
        """Find and group contours from an edge map into logical objects.

        Args:
            edge_map: 2D binary numpy array (output of edge detection).

        Returns:
            List of groups. Each group is a list of contour arrays (each
            contour is a numpy array of shape (N, 1, 2)).
            Returns an empty list if no contours are found.
        """
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
