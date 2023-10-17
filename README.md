# acl-2022-camera-ready

- **Note: This repo is still under construction**

## Paper
- [Link]([https://openreview.net/forum?id=oQ43ZecmCwe](https://aclanthology.org/2022.acl-long.28/)) to the paper.

## Data
- [Link](https://www.dropbox.com/s/jronowy3gau9pku/USPTO-patents-processed.gz?dl=0) to the data

## Codebase
- `src/trainer.py`: main handler for training.
- `src/model.py`: model definitions.
- `src/preprocessing.py`: data preparation steps, including DataFrame transformation, dataset spliting, and dataloader creation.

## Get Started
### Dependencies
- See `requirements.txt`

### Command
- Train: `python run.py -r -m CONFIG_FILE`
- Test / Inference: `python run.py -s test`

## Contact
- Codebase: [Zhaoyi H.](mailto:joeyhou@seas.upenn.edu), [Yifei N.](mailto:y3ning@ucsd.edu)
- Research / Paper: [Xiaocheng G.](mailto:Xiaochen.Gao@rady.ucsd.edu), Jingbo S., Vish K.
