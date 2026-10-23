import numpy as np

from core.tiles import batch_time, comp_to_tiles, grid, tile_effects, tile_index, tile_scores, window


def test_grid_and_index_agree_with_clipped_last_row():
    size, tile = (8, 6), 4                      # 2 × 2 патча, нижний ряд высотой 2
    boxes = grid(size, tile)
    assert boxes == [(0, 0, 4, 4), (4, 0, 8, 4), (0, 4, 4, 6), (4, 4, 8, 6)]
    tidx = tile_index(size, tile)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        assert (tidx[y0:y1, x0:x1] == k).all()


def test_window_is_shifted_inside_frame():
    assert window((0, 0, 4, 4), 4, 1, (8, 6)) == (0, 0)          # у края — сдвиг внутрь
    assert window((4, 4, 8, 6), 4, 1, (8, 6)) == (2, 0)          # окно 6×6 в кадре 8×6
    assert window((4, 0, 8, 4), 4, 1, (16, 16)) == (3, 0)


def test_tile_scores_are_area_weighted_component_scores():
    size, tile = (4, 2), 2
    tidx = tile_index(size, tile)                # два патча 2×2
    comp = np.array([[0, 0, 0, 1], [0, 0, 1, 1]])
    overlap = comp_to_tiles(comp, 2, tidx, 2)
    assert overlap.tolist() == [[4, 1], [0, 3]]
    assert np.allclose(tile_scores(np.array([1.0, 5.0]), overlap), [1.0, 4.0])


def test_tile_effects_delta_and_confusion():
    tidx = tile_index((4, 1), 2)
    gt = np.array([[0, 1, 1, 255]])
    b0 = np.array([[0, 0, 1, 1]])
    b2 = np.array([[1, 1, 1, 0]])
    d, c0, delta = tile_effects(gt, b0, b2, tidx, 2)
    assert delta.tolist() == [0, 0]              # патч 0: +1 (пиксель 1) −1 (пиксель 0); патч 1 без изменений
    assert (c0 + d.sum(0))[1, 1] == 2 and c0[1, 1] == 1


def test_batch_time():
    per = [0.0, 10.0, 15.0]
    assert batch_time(0, per, 2) == 0 and batch_time(3, per, 2) == 25.0 and batch_time(4, per, 2) == 30.0
