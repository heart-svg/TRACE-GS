import torch

from dem_fix_model import DEMFix


def main() -> None:
    torch.manual_seed(19)
    model = DEMFix([12], n_heads=1, n_encoders=1, hidden_dim=16, output_dim=1, dropout=0.0)
    model.eval()
    x = torch.randn(4, 12)
    with torch.no_grad():
        reference = model([x]).clone()
        changed = x.clone()
        changed[3] += 100.0
        perturbed = model([changed])
    delta = float((reference[:3] - perturbed[:3]).abs().max())
    print("DEM_FIX_CROSS_SAMPLE_DELTA", delta)
    assert delta < 1e-7
    print("DEM_FIX_TEST_PASSED")


if __name__ == "__main__":
    main()

