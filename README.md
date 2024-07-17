
### Create conda environment
```bash
conda create -n inspeqtor python=3.11
```

### Test with coverage
```bash
pytest --cov=inspeqtor tests/
```

Project submodules
- data
- model
- training
- quantum_info
- visualization
- typing
- utils
- qiskit
- pennylane
- pulse
- constant