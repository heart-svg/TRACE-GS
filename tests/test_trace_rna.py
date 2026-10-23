import torch

from trace_rna import TRACERNAOnly


def main() -> None:
    torch.manual_seed(23)
    model = TRACERNAOnly(input_dim=64, n_traits=4)
    model.eval()
    x = torch.randn(5, 64)
    traits = torch.tensor([0, 1, 2, 3, 0])
    with torch.no_grad():
        result = model(x, traits)
        altered = x.clone()
        altered[4] += 50
        changed = model(altered, traits)
    assert result["prediction"].shape == (5,)
    assert result["programs"].shape == (5, 32, 64)
    assert result["program_attention"].shape == (5, 4, 32)
    delta = float((result["prediction"][:4] - changed["prediction"][:4]).abs().max())
    print("RNA_BATCH_INVARIANCE_DELTA", delta)
    print("RNA_PARAMETERS", sum(p.numel() for p in model.parameters()))
    assert delta < 1e-7
    print("TRACE_RNA_TEST_PASSED")


if __name__ == "__main__":
    main()

