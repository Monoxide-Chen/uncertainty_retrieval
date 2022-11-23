from data import DEFAULT_VOCAB_PATHS, create_dataloaders
from evaluators import get_evaluator_cls
from language import vocabulary_factory
from loggers.file_loggers import BestModelTracker
from loggers.wandb_loggers import WandbSimplePrinter, WandbSummaryPrinter
from losses import loss_factory
from models import create_models
from optimizers import create_optimizers, create_lr_schedulers
from options import get_experiment_config
from set_up import setup_experiment
from trainers import get_trainer_cls
from transforms import image_transform_factory, text_transform_factory


def main():
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)

    vocabulary = vocabulary_factory(config={
        'vocab_path': configs['vocab_path'] if configs['vocab_path'] else DEFAULT_VOCAB_PATHS[configs['dataset']],
        'vocab_threshold': configs['vocab_threshold']
    })
    image_transform = image_transform_factory(config=configs)
    text_transform = text_transform_factory(config={'vocabulary': vocabulary})
    train_dataloader, test_dataloaders, train_val_dataloaders = create_dataloaders(image_transform, text_transform,
                                                                                   configs)
    criterions = loss_factory(configs)
    models = create_models(configs, vocabulary)
    optimizers = create_optimizers(models=models, config=configs)
    lr_schedulers = create_lr_schedulers(optimizers, config=configs)
    train_loggers = [WandbSimplePrinter('train/')]
    summary_keys = get_summary_keys(configs)
    best_metric_key = [key for key in summary_keys if '10' in key][0]
    val_loggers = [WandbSimplePrinter('val/'), WandbSummaryPrinter('best_', summary_keys),
                   BestModelTracker(export_root, metric_key=best_metric_key)]
    evaluators = {key: get_evaluator_cls(configs)(models, value, top_k=configs['topk']) 
                  for key, value in test_dataloaders.items()}
    train_evaluators = {key: get_evaluator_cls(configs)(models, value, top_k=configs['topk'])
                       for key, value in train_val_dataloaders.items()}
    trainer = get_trainer_cls(configs)(models, train_dataloader, criterions, optimizers, lr_schedulers,
                                       configs['epoch'], train_loggers, val_loggers, evaluators, train_evaluators,
                                       start_epoch=0)
    trainer.run()


def get_summary_keys(configs):
    summary_keys = ['recall_@{}'.format(k) for k in configs['topk'].split(",")]
    return summary_keys


if __name__ == '__main__':
    main()
