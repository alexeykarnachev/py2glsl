from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pytest
from PIL import Image

from py2glsl import render_image

# Directory for gold files relative to tests directory
GOLD_DIR = Path(__file__).parent / "gold"
GOLD_DIR.mkdir(exist_ok=True)


def compare_images(img1: np.ndarray, img2: np.ndarray, tolerance: float = 1e-7) -> bool:
    """Compare two images with tolerance."""
    if img1.shape != img2.shape:
        return False
    return np.allclose(img1, img2, rtol=tolerance, atol=tolerance)


def verify_shader_output(
    shader_func: Callable,
    test_name: str,
    tmp_path: Path,
    uniforms: Optional[Dict[str, Any]] = None,
    size: tuple[int, int] = (512, 512),
) -> None:
    """Verify shader output against gold image."""
    uniforms = uniforms or {}

    # Paths for test and gold images
    test_path = tmp_path / f"{test_name}.png"
    gold_path = GOLD_DIR / f"{test_name}.png"

    # Render current output
    result = render_image(shader_func, size=size, **uniforms)
    result.save(test_path)

    if not gold_path.exists():
        # Show output and ask for verification
        result.show()
        response = input(f"\nAccept this as gold for {test_name}? [y/N]: ").lower()
        if response != "y":
            pytest.fail("User rejected gold image creation")
        result.save(gold_path)
        print(f"Created gold file: {gold_path}")
    else:
        # Compare with existing gold
        gold = np.array(Image.open(gold_path))
        current = np.array(result)

        if not compare_images(gold, current):
            # Save diff image for debugging
            diff = np.abs(gold.astype(float) - current.astype(float))
            diff = (diff * 255).astype(np.uint8)
            diff_path = tmp_path / f"{test_name}_diff.png"
            Image.fromarray(diff).save(diff_path)

            pytest.fail(
                f"Shader output differs from gold file.\n"
                f"Gold: {gold_path}\n"
                f"Test: {test_path}\n"
                f"Diff: {diff_path}"
            )
