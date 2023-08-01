from argparse import ArgumentParser
from data_modules import S2ImageModule
from models import SteepestDescentZS

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def main(hparams):
    if hparams.testdata == "None" and hparams.test:
        hparams.testdata = hparams.dataset

    seed_everything(42, workers=True)
    s2_data = S2ImageModule(hparams.datadir, hparams.dataset, valfile='apex', testfile=hparams.testdata, batch_size=int(hparams.batch_size), num_workers=int(hparams.num_workers), patch_repeat=int(hparams.patch_repeat))
    target_size = (int(hparams.output_height), int(hparams.output_width))
    
    sr_method = SteepestDescentZS(lr=float(hparams.lr),
                                  lamb=float(hparams.lamb),
                                  regularizer=hparams.regularizer,
                                  n_its=int(hparams.n_its),
                                  n_filters=int(hparams.n_filters),
                                  filter_size=int(hparams.filter_size),
                                  beta_1=float(hparams.beta_1),
                                  beta_2=float(hparams.beta_2),
                                  sched_gamma=float(hparams.sched_gamma),
                                  learn_alpha=hparams.learn_alpha,
                                  learn_beta=hparams.learn_beta,
                                  output_size=target_size,
                                  eval_metrics=hparams.metrics,
                                  save_test_pred=hparams.save_test_preds)
    
    tags=hparams.tags.split(',')
    if hparams.train and hparams.dataset not in tags:
        tags.append(hparams.dataset)
    if hparams.test and hparams.testdata not in tags:
        tags.append(hparams.testdata)
    if hparams.test and not hparams.train:
        tags.append("baseline")

    wandb_logger = WandbLogger(project=hparams.projectname,
                                tags=tags,
                                log_model=False)

    if hparams.save_checkpoints:
        callbacks = [
            ModelCheckpoint(
                dirpath="checkpoints",
                monitor="train_loss",
                mode="min",
                save_top_k=5,
                save_on_train_epoch_end=True,
                save_last=True
            ),
        ]
    else:
        callbacks = []

    trainer = Trainer(
        precision=64,
        max_epochs=int(hparams.epochs),
        log_every_n_steps=int(hparams.log_every_n_steps),
        limit_val_batches=1,
        logger=wandb_logger,
        accelerator=hparams.accelerator,
        devices=[int(hparams.devices)],
        strategy=hparams.strategy,
        callbacks=callbacks,
        gradient_clip_val=float(hparams.g_clip),
    )

    wandb.config.update(hparams)

    if hparams.train:
        trainer.fit(sr_method, s2_data)
    if hparams.test:
        if hparams.regularizer in ["learned","Proposed"]:
            trainer.test(sr_method, s2_data, ckpt_path="best")
        else:
            trainer.test(sr_method, s2_data)

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=0)
    parser.add_argument("--dataset", default="escondido")
    parser.add_argument("--testdata", default="None")
    parser.add_argument("--datadir", default="./dataset")
    parser.add_argument("--projectname", default="igarss-test")
    parser.add_argument("--strategy", default="ddp_find_unused_parameters_false")
    parser.add_argument("--epochs", default=40)
    parser.add_argument("--n_runs", default=1)
    parser.add_argument("--n_its", default=3)
    parser.add_argument("--n_filters", default=12)
    parser.add_argument("--filter_size", default=3)
    parser.add_argument("--lr", default=1e-2)
    parser.add_argument("--beta_1", default=0.98)
    parser.add_argument("--beta_2", default=0.9999)
    parser.add_argument("--sched_gamma", default=0.6)
    parser.add_argument("--lamb", default=1.0)
    parser.add_argument("--rho", default=5000)
    parser.add_argument("--sigma", default=1.0)
    parser.add_argument("--checkpoint_interval", default=500)
    parser.add_argument("--output_width", default=108)
    parser.add_argument("--output_height", default=456)
    parser.add_argument("--patch_repeat", default=1)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--num_workers", default=4)
    parser.add_argument("--limsub", default=6)
    parser.add_argument("--log_every_n_steps", default=5)
    parser.add_argument("--tags", default="sweep")
    parser.add_argument("--g_clip", default="0.5")

    parser.add_argument("--regularizer", default='USeME')

    parser.add_argument("--initialize_state", dest='initialize_state', action='store_true')
    parser.set_defaults(initialize_state=False)

    parser.add_argument("--save_test_preds", dest='save_test_preds', action='store_true')
    parser.set_defaults(save_test_preds=False)

    parser.add_argument("--learn_alpha", dest='learn_alpha', action='store_true')
    parser.add_argument("--fix_alpha", dest='learn_alpha', action='store_false')
    parser.set_defaults(learn_alpha=True)

    parser.add_argument("--learn_beta", dest='learn_beta', action='store_true')
    parser.add_argument("--fix_beta", dest='learn_beta', action='store_false')
    parser.set_defaults(learn_beta=False)

    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--no_train', dest='train', action='store_false')
    parser.set_defaults(train=True)

    parser.add_argument('--eval_metrics', dest='metrics', action='store_true')
    parser.add_argument('--no_metrics', dest='metrics', action='store_false')
    parser.set_defaults(metrics=False)

    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no_test', dest='test', action='store_false')
    parser.set_defaults(test=True)

    parser.add_argument("--save_checkpoints", dest='save_checkpoints', action='store_true')
    parser.set_defaults(save_checkpoints=False)

    args = parser.parse_args()

    main(args)