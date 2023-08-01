# Unrolled Sentinel-2 Multispectral Estimation

## Repository to accompany "Superresolving Sentinel-2 Using Learned Multispectral Regularization"
### Presented at IGARSS 2023

This repository contains the code used in the paper ["Superresolving Sentinel-2 Using Learned Multispectral Regularization."](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=Armannsson "Correct link here once available...")

The method is set up to load data from `.mat` files, custom data loading can be added to `mat_loaders.py`.

The method will log results and training metrics to [wandb.ai](https://wandb.ai).


### Usage example

```bash
python training.py --filter_size=3 --learn_alpha --learn_beta --n_filters=12 --n_its=3 --train --lr=1e-2 --epochs=40 --batch_size=1 --lamb=0.5 --beta_1=0.98 --beta_2=0.9999 --sched_gamma=0.6 --projectname="igarss-2023" --test --datadir="./dataset" --dataset="escondido_s2" --no_metrics --save_test_preds --accelerator="gpu" --devices=0
```
The above command should give the same results as presented in the paper.
The flags used are as follows:
- `filter_size` determines the spatial dimensions of the `G` filter, here $3 \times 3$.
- `learn_alpha` determines that the $\alpha$ parameters should be learned.
- `learn_beta` determines that the $\beta$ parameters should be learned.
- `n_filters` determines the number of filters applied per channel, here $12$.
- `n_its` determines the number of layers in the network (the unrolled equivalent to the number of iterations), here $3$.
- `train` specifies that that training should be performed.
- `lr` sets the learning rate.
- `epochs` sets the number of training epochs.
- `batch_size` sets the training batch size.
- `lamb` tunes the balance between the MAE loss and the gradient loss during training.
- `beta_1` tunes the `beta_1` parameter of the `NAdam` optimizer.
- `beta_2` tunes the `beta_2` parameter of the `NAdam` optimizer.
- `sched_gamma` tunes the learning rate decay using a multi step learning rate scheduler.
- `projectname` sets a project name for [wandb.ai](https://wandb.ai)
- `train` specifies that that testing should be performed.
- `datadir` specifies the directory in which to find Sentinel-2 image data.
- `dataset` specifies the name of the data to process.
- `no_metrics` specifies that no test metrics should be evaluated since ground truth is not available for this particular dataset. 
  - In the case of test data that includes ground truth information the `eval_metrics` flag should be used instead.
- `save_test_preds` specifies that the output should be saved. An `npz` file with the output image is saved with a filename generated from the regularizer name (default `USeME`, can be customized using the `--regularizer` flag) and the [wandb.ai](https://wandb.ai) run name.
- `accelerator` determines whether GPU acceleration is used or not.
- `devices` selects which GPUs to use if multiple are available.
