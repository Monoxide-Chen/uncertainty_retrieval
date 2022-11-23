from trainers.tirg_trainer import TIRGTrainer

TRAINER_CODE_DICT = {
    TIRGTrainer.code(): TIRGTrainer,
}


def get_trainer_cls(configs):
    trainer_code = configs['trainer']
    return TRAINER_CODE_DICT[trainer_code]
