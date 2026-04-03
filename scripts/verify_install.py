"""Verification script to check that everything is installed and working."""

import sys

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()


def verify_imports():
    """Verify all required packages can be imported."""
    print("Verifying package imports...")
    
    packages = [
        "jax",
        "equinox",
        "diffrax",
        "optax",
        "jaxtyping",
        "yaml",
        "h5py",
        "numpy",
        "matplotlib",
        "streamlit",
        "plotly",
        "casadi",
        "pytest",
        "tqdm",
        "dte",
    ]
    
    failed = []
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError as e:
            print(f"  ✗ {pkg} - {e}")
            failed.append(pkg)
    
    return len(failed) == 0, failed


def verify_jax_backend():
    """Verify JAX backend."""
    import jax
    
    print("\nJAX Backend:")
    print(f"  Devices: {jax.devices()}")
    print(f"  Default backend: {jax.default_backend()}")
    
    # Test simple computation
    import jax.numpy as jnp
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x ** 2)
    print(f"  Test computation: sum([1,2,3]^2) = {y}")
    
    return True


def verify_dte_modules():
    """Verify DTE modules can be imported."""
    print("\nVerifying DTE modules...")
    
    modules = [
        "dte.simulators.cstr",
        "dte.simulators.heat_exchanger",
        "dte.simulators.two_tank",
        "dte.data.generation",
        "dte.data.dataset",
        "dte.models.encoder",
        "dte.models.decoder",
        "dte.models.latent_sde",
        "dte.models.digital_twin",
        "dte.physics.conservation",
        "dte.physics.heat_exchanger",
        "dte.physics.two_tank",
        "dte.control.pid",
        "dte.control.mpc",
        "dte.training.trainer",
        "dte.training.losses",
        "dte.autoresearch.workflow",
        "dte.utils.plotting",
        "dte.utils.logging",
    ]
    
    failed = []
    for mod in modules:
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except ImportError as e:
            print(f"  ✗ {mod} - {e}")
            failed.append(mod)
    
    return len(failed) == 0, failed


def verify_configs():
    """Verify configuration files exist."""
    print("\nVerifying configuration files...")
    
    from pathlib import Path
    
    configs = [
        "configs/cstr_default.yaml",
        "configs/heat_exchanger_default.yaml",
        "configs/two_tank_default.yaml",
        "configs/training_default.yaml",
        "configs/heat_exchanger_training.yaml",
        "configs/two_tank_training.yaml",
        "configs/mpc_default.yaml",
        "configs/autoresearch_default.yaml",
    ]
    
    missing = []
    for config in configs:
        if Path(config).exists():
            print(f"  ✓ {config}")
        else:
            print(f"  ✗ {config} - NOT FOUND")
            missing.append(config)
    
    return len(missing) == 0, missing


def main():
    print("="*60)
    print("DIGITAL TWIN ENGINE - INSTALLATION VERIFICATION")
    print("="*60)
    
    all_passed = True
    
    # Verify imports
    passed, failed = verify_imports()
    all_passed = all_passed and passed
    
    # Verify JAX backend
    passed = verify_jax_backend()
    all_passed = all_passed and passed
    
    # Verify DTE modules
    passed, failed = verify_dte_modules()
    all_passed = all_passed and passed
    
    # Verify configs
    passed, missing = verify_configs()
    all_passed = all_passed and passed
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("="*60)
        print("\nYou're ready to use Digital Twin Engine!")
        print("\nNext steps:")
        print("  1. Generate data: python scripts/generate_data.py")
        print("  2. Train model: python scripts/train.py")
        print("  3. Run tests: pytest tests/")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
