"""Unit tests for integer DBU and layer contracts."""

import numpy as np
import pytest

from layout import DbuBox, LayerSpec


def test_dbu_box_geometry_and_numpy_integer_normalization() -> None:
    """DbuBox keeps exact integers and treats touching as zero-area overlap."""
    box = DbuBox(np.int64(-10), 0, 10, 20)
    assert (box.width, box.height, box.area) == (20, 20, 400)
    assert box.expanded(5) == DbuBox(-15, -5, 15, 25)
    assert box.intersection(DbuBox(0, 10, 30, 40)) == DbuBox(0, 10, 10, 20)
    assert not box.overlaps(DbuBox(10, 0, 20, 20))


@pytest.mark.parametrize("coords", [(0, 0, 0, 1), (0, 1, 1, 1), (2, 0, 1, 1)])
def test_dbu_box_rejects_non_positive_area(coords: tuple[int, int, int, int]) -> None:
    """Empty and inverted windows must fail before native queries run."""
    with pytest.raises(ValueError):
        DbuBox(*coords)


def test_layer_spec_is_orderable_and_validated() -> None:
    """Layer specifications sort deterministically and reject negative identifiers."""
    assert sorted([LayerSpec(2, 0), LayerSpec(1, 5)]) == [LayerSpec(1, 5), LayerSpec(2, 0)]
    with pytest.raises(ValueError):
        LayerSpec(-1, 0)
