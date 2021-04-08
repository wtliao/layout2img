# layout2img
This repository includes the implementation for [Context-Aware Layout to Image Generation with Enhanced Object Appearance](https://arxiv.org/abs/2004.14231) (to appear in CVPR 2021).

This repo is not completely.

## Requirements
* python3
* pytorch >1.0
* numpy
* matplotlib
* opencv

Or install full requirements by running:
```bash
pip install -r requirements.txt
```

## TODO
- [x] instruction to prepare dataset
- [ ] remove all unnecessary files
- [ ] add link to download our pre-trained model
- [ ] clean code including comments
- [ ] instruction for training
- [ ] instruction for evaluation

## Training ImageTransformer

### Data Preparation
Download COCO dataset to datasets/coco
```bash
bash scripts/download_coco.sh
```
Download VG dataset to datasets/vg
```bash
bash scripts/download_vg.sh
python scripts/preprocess_vg.py
```


### Start training


See `opts.py` for the options. (You can download the pretrained models from [here]()


### Evaluation


### Trained model
you can download our trained model from our [onedrive repo]()

### Performance
You will get the scores close to below after training under xe loss for xxxxx epochs:

## Reference

If you find this repo helpful, please consider citing:

```
@inproceedings{he2021context,
  title={Context-Aware Layout to Image Generation with Enhanced Object Appearance},
  author={He, Sen and Liao, Wentong and Yang, Michael and Yang, Yongxin and Song, Yi-Zhe and Rosenhahn, Bodo and Xiang, Tao},
  booktitle={CVPR},
  year={2021}
}
```

## Acknowledgements

This repository is based on [LostGAN](https://github.com/WillSuen/LostGANs), and the propsoed modules can be applied in the [layout2img](https://github.com/zhaobozb/layout2im).
