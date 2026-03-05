# Extraction Pipeline Plan

## Overview

The image extraction pipeline isolates distinct objects from an image through a sequence of
preprocessing, edge detection, grouping, and closure stages. The pipeline operates on a
greyscale `np.ndarray` internally regardless of the original input format.

Each processing stage (blur, edge detection, contour grouping, edge closure) is implemented
in its own dedicated class. `Extractor` acts as the orchestrator — it converts the input,
instantiates each stage class with `config`, calls them in order, and assembles the final
`List[EdgeBoundedObject]`. This keeps every stage independently testable.

```
Extractor (orchestrator)
├── ImageBlur          — Stage 2: size-adaptive Gaussian blur
├── EdgeDetector       — Stage 3: Canny / Sobel / DEXI edge detection
├── ContourGrouper     — Stage 4: contour finding and proximity grouping
└── EdgeCloser         — Stage 5: dilation-based gap bridging
```

---

## Pipeline Stages

### Stage 1 — Format Detection & Conversion

**Location:** `Extractor._extract_byte_array_2d`, `Extractor._extract_file`, `Extractor._extract_bitmap`

Detect the input format and normalise it into a 2D greyscale `np.ndarray` before any
processing begins.

| Input Type    | Conversion Strategy                                         |
|---------------|-------------------------------------------------------------|
| `np.ndarray`  | Validate 2D, non-empty; use directly                        |
| `List[bytes]` | Stack rows via `np.frombuffer` → 2D `uint8` array           |
| `str` / `Path`| Detect extension; read with `cv2.imread(IMREAD_GRAYSCALE)`  |

Supported file formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.gif`

---

### Stage 2 — Size-Adaptive Gaussian Blur (in-memory cache)

**Class:** `ImageBlur` — `src/image_blur.py`

**Interface:**
```python
class ImageBlur:
    def __init__(self, config: Config) -> None: ...
    def apply(self, image: np.ndarray) -> np.ndarray: ...
```

Apply Gaussian blur to suppress noise before edge detection.

Current implementation: `ImageBlur` uses the configured `border_blur_size` directly as
the kernel size (forced odd with `| 1`). `blur_kernel_min` and `blur_kernel_max` were
added to `Config` as parameters but are not used by the `ImageBlur` implementation at
the moment.

`Extractor` stores the blurred result as `self._last_blurred: np.ndarray` after calling
`ImageBlur.apply()`, keeping it in memory for downstream inspection or reuse.

---

### Stage 3 — Edge Detection

**Class:** `EdgeDetector` — `src/edge_bounded_object.py` (already exists, add `detect()`)

**Interface:**
```python
class EdgeDetector:
    def __init__(self, config: Config) -> None: ...
    def detect(self, image: np.ndarray) -> np.ndarray: ...
```

Apply the configured edge detection algorithm to the blurred image to produce an edge map.

#### Supported Methods (`EdgeDetectionMethod` enum)

| Method  | Implementation                                          | Config Fields                          |
|---------|---------------------------------------------------------|----------------------------------------|
| `CANNY` | `cv2.Canny(image, low, high)`                           | `canny_threshold: [low, high]`         |
| `SOBEL` | `cv2.Sobel` in X and Y, compose magnitude image         | `sobel_kernel_size: int` (default `3`) |
| `DEXI`  | *(future — placeholder stub)*                           | TBD                                    |

The active method is selected via `config.edge_detection_method` (default: `"canny"`).

---

### Stage 4 — Object Grouping

**Class:** `ContourGrouper` — `src/contour_grouper.py`

**Interface:**
```python
class ContourGrouper:
    def __init__(self, config: Config) -> None: ...
    def group(self, edge_map: np.ndarray) -> List[List[np.ndarray]]: ...
```

Find and group the raw edges from the edge map into logical objects.

1. Run `cv2.findContours` on the edge map (`RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`).
2. Merge contours whose bounding boxes are within `config.grouping_proximity` pixels using
   bounding-box union.
3. *(Future)* Replace proximity-union with **Jaccard Index** overlap scoring to handle
   complex, overlapping, or concave shapes.

```python
# TODO: Replace with Jaccard Index grouping
# jaccard = intersection_area / union_area
# Merge contours where jaccard >= config.grouping_threshold
```

---

### Stage 5 — Edge Closure (Dilation)

**Class:** `EdgeCloser` — `src/edge_closer.py`

**Interface:**
```python
class EdgeCloser:
    def __init__(self, config: Config) -> None: ...
    def close(self, edge_region: np.ndarray) -> np.ndarray: ...
```

Disconnected edge fragments around an object leave gaps in the boundary. Apply
morphological dilation to bridge those gaps and produce a closed contour.

```
kernel = cv2.getStructuringElement(MORPH_ELLIPSE, (dilation_size, dilation_size))
closed = cv2.dilate(edge_region, kernel, iterations=1)
```

The `dilation_size` comes from `config.dilation_size` (default: `3`).

After closure, `Extractor` extracts the bounding-box crop from the **original** (pre-blur)
image so that the returned object retains full pixel detail.

---

### Stage 6 — Output: `EdgeBoundedObject`

Each detected object is wrapped in an `EdgeBoundedObject` (defined in
`src/edge_bounded_object.py`) containing:

| Field                   | Content                                            |
|-------------------------|----------------------------------------------------|
| `edge_detection_method` | `EdgeDetectionMethod` used                         |
| `blur_strength`         | `[kernel_min, kernel_max]` applied                 |
| `edge_data_description` | `CannyData` or `SobelData` with algorithm params   |
| `coordinates`           | `[(x, y), ...]` contour points in original image   |
| `bounded_image`         | Cropped `np.ndarray` from original image           |

`extract_from_image` returns `List[EdgeBoundedObject]`.

---

## Config Changes Required

Add the following fields to `Config` in `src/config.py`:

| Field                   | Type        | Default     | Description                          |
|-------------------------|-------------|-------------|--------------------------------------|
| `edge_detection_method` | `str`       | `"canny"`   | `"canny"`, `"sobel"`, or `"dexi"`    |
| `canny_threshold`       | `List[int]` | `[50, 150]` | `[low, high]` thresholds for Canny   |
| `sobel_kernel_size`     | `int`       | `3`         | Kernel size for Sobel operator       |
| `dilation_size`         | `int`       | `3`         | Kernel size for edge closure step    |
| `grouping_proximity`    | `int`       | `10`        | Max pixel gap to merge contours      |

---

## File Change Summary

| File                         | Change                                                                                         |
|------------------------------|------------------------------------------------------------------------------------------------|
| `src/config.py`              | Add 5 new fields + JSON deserialization support                                                |
| `src/extractor.py`           | Implement `_extract_bitmap`, `_extract_byte_array_2d`, `_extract_file`; orchestrate stages     |
| `src/image_blur.py`          | New class `ImageBlur` — size-adaptive Gaussian blur                                            |
| `src/edge_bounded_object.py` | Add `detect()` to `EdgeDetector`; add `DEXI` stub to `EdgeDetectionMethod`                     |
| `src/contour_grouper.py`     | New class `ContourGrouper` — `cv2.findContours` + proximity-union grouping                     |
| `src/edge_closer.py`         | New class `EdgeCloser` — morphological dilation to close edge gaps                             |
| `src/__init__.py`            | Export `Extractor`, `EdgeBoundedObject`, `ImageBlur`, `ContourGrouper`, `EdgeCloser`           |
| `tests/test_extractor.py`    | Tests for all input types and error cases using `object_extraction.png`                        |
| `tests/test_image_blur.py`   | Unit tests for `ImageBlur.apply()` — adaptive kernel, manual override, output shape            |
| `tests/test_edge_detector.py`| Unit tests for `EdgeDetector.detect()` — Canny, Sobel, unsupported method error               |
| `tests/test_contour_grouper.py` | Unit tests for `ContourGrouper.group()` — merging, proximity threshold, empty edge map     |
| `tests/test_edge_closer.py`  | Unit tests for `EdgeCloser.close()` — gap bridging, dilation size from config                 |

---

## Open Questions

1. **Return type** — Update `extract_from_image` signature to `List[EdgeBoundedObject]` now,
   or keep `List[np.ndarray]` with a separate `extract_objects()` method?
2. **Jaccard grouping** — Implement proximity-union now with a `TODO`, or defer the entire
   grouping step until the algorithm is decided?
3. **Blur override** — Should `border_blur_size` in `Config` override or stack on top of
   the adaptive blur?
