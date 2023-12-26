"""Test mnist-sagemaker-ci-cd."""

import mnist_sagemaker_ci_cd


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(mnist_sagemaker_ci_cd.__name__, str)
