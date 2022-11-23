from tqdm import tqdm

from trainers.abc import AbstractBaseTrainer
from utils.metrics import AverageMeterSet


class TIRGTrainer(AbstractBaseTrainer):
    def __init__(self, models, train_dataloader, criterions, optimizers, lr_schedulers, num_epochs,
                 train_loggers, val_loggers, evaluators, *args, **kwargs):
        super().__init__(models, train_dataloader, criterions, optimizers, lr_schedulers, num_epochs,
                         train_loggers, val_loggers, evaluators, *args, **kwargs)
        self.lower_image_encoder = self.models['lower_image_encoder']
        self.upper_image_encoder = self.models['upper_image_encoder']
        self.text_encoder = self.models['text_encoder']
        self.text_fc = self.models['text_fc'] if 'text_fc' in self.models else None
        self.compositor = self.models['layer4']
        self.augmenter = self.models['augmenter'] if 'augmenter' in self.models else None
        self.metric_loss = self.criterions['metric_loss']

    def train_one_epoch(self, epoch):
        average_meter_set = AverageMeterSet()
        train_dataloader = tqdm(self.train_dataloader, desc="Epoch {}".format(epoch))

        for batch_idx, (ref_images, tar_images, modifiers, len_modifiers, attn_mask) in enumerate(train_dataloader):
            ref_images, tar_images = ref_images.to(self.device), tar_images.to(self.device)
            modifiers, len_modifiers = modifiers.to(self.device), len_modifiers.to(self.device)

            self._reset_grad()
            # Encode Target Images
            tar_mid_features, _ = self.lower_image_encoder(tar_images)
            tar_features = self.upper_image_encoder(tar_mid_features)

            # Encode and Fuse Reference Images with Texts
            ref_mid_features, _ = self.lower_image_encoder(ref_images)
            if self.text_fc != None:
                attn_mask = attn_mask.to(self.device)
                text_features = self.text_encoder(modifiers, attn_mask)
                text_features = self.text_fc(text_features)
            else:
                text_features = self.text_encoder(modifiers, len_modifiers)

            composed_ref_features, _ = self.compositor(ref_mid_features, text_features)
            composed_ref_features = self.upper_image_encoder(composed_ref_features)

            # Add Gaussian noisy to feature and compute Loss
            if self.augmenter != None:
                augmented_tar_features = self.augmenter(tar_features)
                loss = self.metric_loss(composed_ref_features, tar_features, augmented_tar_features, epoch)
            else:
                loss = self.metric_loss(composed_ref_features, tar_features)

            loss.backward()
            average_meter_set.update('loss', loss.item())
            self._update_grad()

        train_results = average_meter_set.averages()
        optimizers_dict = self._get_state_dicts(self.optimizers)
        for key in optimizers_dict.keys():
            train_results[key+'_lr'] = optimizers_dict[key]["param_groups"][0]["lr"]
        self._step_schedulers()
        return train_results

    @classmethod
    def code(cls) -> str:
        return 'tirg'
