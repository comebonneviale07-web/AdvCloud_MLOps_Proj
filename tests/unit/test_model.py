import pytest

torch = pytest.importorskip("torch")

from trajclf.model import RideCNN, RideMLP, count_parameters  # noqa: E402


def test_mlp_matches_research_architecture():
    m = RideMLP(30, 5)
    # research/train_neural_network.py: 30->64->32->5
    assert count_parameters(m) == 30 * 64 + 64 + 64 * 32 + 32 + 32 * 5 + 5
    assert m(torch.randn(4, 30)).shape == (4, 5)


@pytest.mark.parametrize("seq_len", [64, 1024])
def test_cnn_forward_shape(seq_len):
    assert RideCNN(6, 5).eval()(torch.randn(3, seq_len, 6)).shape == (3, 5)


def test_models_are_small():
    assert count_parameters(RideCNN(6, 5)) < 1_000_000
