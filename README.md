# CNCV.jl


A Julia package that implements a conditional neural control variates.

## Installation

### Prerequisites

Make sure you have `matplotlib` installed in your Python environment for plotting:
```bash
pip install matplotlib
```

Configure PyCall to use your Python installation:
```julia
ENV["PYTHON"] = "/usr/bin/python3"  # Adjust path to your Python
using Pkg
Pkg.build("PyCall")
# Restart Julia
```

### Install the package

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running Examples

### Train a model
```julia
julia --project=. scripts/rosenbrock_training.jl
# Or for conditional version:
julia --project=. scripts/rosenbrock_amortized_training.jl
```

### Test a trained model

```julia
julia --project=. scripts/rosenbrock_sampling.jl
# Or for conditional version:
julia --project=. scripts/rosenbrock_amortized_sampling.jl
```

## Author

Ali Siahkoohi (alisk@ucf.edu)
University of Central Florida, 2025

## License

MIT
