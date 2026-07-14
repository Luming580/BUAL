# Rethinking Uncertainty in Active Learning for Image Classification: When Confidence Is Misleading -- MM2026


## Run

```bash
python BUAL.py --data-dir C:/data --output-dir Log/CIFAR10
```



## Structure

- `BUAL.py` - experiment runner, data loading, active-learning acquisition.
- `config.py` - typed experiment configuration plus legacy constants.
- `utils.py` - training, evaluation, normalization, and distribution logging.
- `sampler.py` - deterministic subset sampler for acquisition.
- `model/` - ResNet backbone and loss-prediction module.
- `pytorch_wavelets/` - vendored wavelet dependency used by the DWT branch.


