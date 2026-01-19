# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.datasets import get_collactor
from .pet_hook import PETHook

import copy
import faiss
import numpy as np
import torch
import pprint
import time

from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate


def _extract_embeddings(model, loader, idx_list, device):
    was_training = model.training
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for batch in loader:
            if "idx_lb" in batch:
                batch_indices = batch["idx_lb"]
            elif "idx_ulb" in batch:
                batch_indices = batch["idx_ulb"]
            elif "idx" in batch:
                batch_indices = batch["idx"]
            else:
                raise KeyError(f"Batch missing idx key. Keys: {list(batch.keys())}")
            actual_indices = [int(idx_list[i]) for i in batch_indices.tolist()]
            images = batch.get("x_lb", batch.get("x_ulb", batch.get("x_ulb_w", batch.get("x"))))
            images = images.to(device)
            feats = model.forward_features(images)
            feats = feats.detach().cpu().numpy().astype(np.float32)
            for actual_idx, feat in zip(actual_indices, feats):
                embeddings[int(actual_idx)] = feat
    if was_training:
        model.train()
    return embeddings


def _compute_grouping(model, lb_loader, ulb_loader, lb_indices, ulb_indices, device):
    lb_embeddings = _extract_embeddings(model, lb_loader, lb_indices, device)
    ulb_embeddings = _extract_embeddings(model, ulb_loader, ulb_indices, device)

    lb_matrix = np.stack([lb_embeddings[int(i)] for i in lb_indices]).astype(np.float32)
    ulb_matrix = np.stack([ulb_embeddings[int(i)] for i in ulb_indices]).astype(np.float32)

    faiss.normalize_L2(lb_matrix)
    faiss.normalize_L2(ulb_matrix)

    index = faiss.IndexFlatIP(lb_matrix.shape[1])
    index.add(lb_matrix)
    _, nearest = index.search(ulb_matrix, 1)

    return {int(ulb_idx): int(lb_indices[int(nearest_idx)]) for ulb_idx, nearest_idx in zip(ulb_indices, nearest[:, 0])}


def _compute_centroid_logits(model, lb_loader, lb_indices, device):
    was_training = model.training
    model.eval()
    centroid_logits = {}
    with torch.no_grad():
        for batch in lb_loader:
            batch_indices = batch["idx_lb"].tolist()
            actual_indices = [int(lb_indices[i]) for i in batch_indices]
            images = batch["x_lb"].to(device)
            logits = model(images)["logits"].detach().cpu()
            for idx_actual, logit in zip(actual_indices, logits):
                centroid_logits[int(idx_actual)] = logit
    if was_training:
        model.train()
    return centroid_logits

@ALGORITHMS.register('pet')
class PET(AlgorithmBase):
    """
        Pseudo Label algorithm (https://arxiv.org/abs/1908.02983).

        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
        - unsup_warm_up (`float`, *optional*, defaults to 0.4):
            Ramp up for weights for unsupervised loss
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.pl = {}
        self.n_rounds = args.n_rounds
        self.conf_threshold = args.conf_threshold
        self.conf_ratio = args.conf_ratio
        self.w_alpha = args.w_alpha
        self.s_alpha = args.s_alpha
        self.kd_w_alpha = args.kd_w_alpha
        self.kd_s_alpha = args.kd_s_alpha
        self.temperature = args.temperature
        self.pretrained_path = args.pretrained_path

        self._model = copy.deepcopy(self.model)
        self._ema_model = copy.deepcopy(self.ema_model)

        # Robust loss settings (off by default)
        self.lambda_1 = getattr(args, "lambda_1", 0.0)
        self.lambda_2 = getattr(args, "lambda_2", 0.0)
        self.grouping_update_interval = getattr(args, "grouping_update_interval", 0)
        self.robust_enable = (self.lambda_1 > 0) or (self.lambda_2 > 0)
        self.robust_ulb2centroid = {}
        self.robust_centroid_logits = {}
        self.robust_lb_targets = {}
        self.robust_lb_idx_list = None
        self.robust_ulb_idx_list = None
        self.robust_centroid_to_ulb_pos = {}
        self._ulb_collate_fn = get_collactor(args, args.net)
        self._robust_base_seed = getattr(args, "seed", None)
        if self._robust_base_seed is None:
            self._robust_rng = np.random.RandomState()
        else:
            self._robust_rng = np.random.RandomState(int(self._robust_base_seed) + int(self.rank))

    def reset_model(self):
        assert self._model is not None, "Model is not initialized"
        assert self._ema_model is not None, "EMA Model is not initialized"
        self.print_fn("Resetting model and EMA model")
        # Load the initial model state
        _model_state_dict = self._model.state_dict()
        if 'module' in list(self.model.state_dict().keys())[0]:
            _model_state_dict = {f'module.{k}': v for k, v in _model_state_dict.items()}
        self.model.load_state_dict(_model_state_dict)
        # Load the initial EMA model state
        _ema_model_state_dict = self._ema_model.state_dict()
        if 'module' in list(self.ema_model.state_dict().keys())[0]:
            _ema_model_state_dict = {f'module.{k}': v for k, v in _ema_model_state_dict.items()}
        self.ema_model.load_state_dict(_ema_model_state_dict)

    def reset_optim(self):
        self.print_fn("Resetting optimizer and scheduler")
        self.optimizer, self.scheduler = self.set_optimizer()

    def set_hooks(self):
        self.register_hook(PETHook(), "PETHook")
        super().set_hooks()

    def _refresh_robust_grouping(self):
        if not self.robust_enable:
            return
        lb_loader = self.loader_dict.get("train_lb_oracle")
        ulb_loader = self.loader_dict.get("train_ulb_oracle")
        if lb_loader is None or ulb_loader is None:
            self.print_fn("Robust grouping skipped: oracle loaders not available.")
            return
        device = next(self.model.parameters()).device
        lb_indices = np.array(lb_loader.dataset.idx_list)
        ulb_indices = np.array(ulb_loader.dataset.idx_list)
        # use EMA model for smoother clustering if available
        clustering_model = self.ema_model if self.ema_model is not None else self.model
        self.robust_ulb2centroid = _compute_grouping(clustering_model, lb_loader, ulb_loader, lb_indices, ulb_indices, device)
        self.robust_centroid_logits = _compute_centroid_logits(clustering_model, lb_loader, lb_indices, device)
        self.robust_lb_idx_list = lb_indices
        self.robust_ulb_idx_list = ulb_indices
        self.robust_centroid_to_ulb_pos = {int(lb_idx): [] for lb_idx in lb_indices}
        ulb_actual_to_pos = {int(actual_idx): pos for pos, actual_idx in enumerate(ulb_indices)}
        for ulb_actual, lb_actual in self.robust_ulb2centroid.items():
            ulb_pos = ulb_actual_to_pos.get(int(ulb_actual))
            if ulb_pos is not None:
                self.robust_centroid_to_ulb_pos[int(lb_actual)].append(int(ulb_pos))
        # cache targets for centroids
        lb_targets = lb_loader.dataset.targets
        self.robust_lb_targets = {int(idx): int(target) for idx, target in zip(lb_indices, lb_targets)}
        self.print_fn(f"Refreshed robust grouping: {len(self.robust_ulb2centroid)} unlabeled assignments, {len(self.robust_centroid_logits)} centroids.")

    def _sample_ulb_positions_for_lb(self, idx_lb):
        if not self.robust_centroid_to_ulb_pos or self.robust_lb_idx_list is None:
            return None
        ulb_positions = []
        lb_positions = idx_lb.tolist()
        for lb_pos in lb_positions:
            if lb_pos < 0 or lb_pos >= len(self.robust_lb_idx_list):
                continue
            lb_actual = int(self.robust_lb_idx_list[lb_pos])
            candidates = self.robust_centroid_to_ulb_pos.get(lb_actual, [])
            if not candidates:
                continue
            n_pick = int(self.args.uratio)
            if len(candidates) >= n_pick:
                picks = self._robust_rng.choice(candidates, size=n_pick, replace=False).tolist()
            else:
                picks = self._robust_rng.choice(candidates, size=n_pick, replace=True).tolist()
            ulb_positions.extend(picks)
        if not ulb_positions:
            return None
        return ulb_positions

    def _build_ulb_batch_from_positions(self, ulb_positions):
        ulb_dset = self.loader_dict['train_ulb'].dataset
        samples = [ulb_dset[pos] for pos in ulb_positions]
        collate_fn = self._ulb_collate_fn
        if collate_fn is None:
            collate_fn = getattr(self.loader_dict['train_ulb'], "collate_fn", None)
        if collate_fn is None:
            collate_fn = default_collate
        return collate_fn(samples)

    def train_step(self, idx_ulb, x_ulb, x_ulb_w, x_ulb_s, pl_ulb, distill_scores_ulb, y_true_ulb):
        # import pdb; pdb.set_trace()
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.args.w_alpha > 0 or self.args.kd_w_alpha > 0:
                logits_x_ulb_w = self.model(x_ulb_w)['logits']
            if self.args.s_alpha > 0 or self.args.kd_s_alpha > 0:
                logits_x_ulb_s = self.model(x_ulb_s)['logits']

            if self.args.w_alpha > 0:
                sup_loss_w = self.ce_loss(logits_x_ulb_w, pl_ulb, reduction='mean')
            else:
                sup_loss_w = torch.tensor(0.0).cuda(self.gpu)
            if self.args.s_alpha > 0:
                sup_loss_s = self.ce_loss(logits_x_ulb_s, pl_ulb, reduction='mean')
            else:
                sup_loss_s = torch.tensor(0.0).cuda(self.gpu)

            if self.args.kd_w_alpha > 0:
                log_probs_x_ulb_w = F.log_softmax(logits_x_ulb_w / self.temperature, dim=1)
                sup_loss_w_kd = F.kl_div(log_probs_x_ulb_w, distill_scores_ulb, reduction='batchmean')
            else:
                sup_loss_w_kd = torch.tensor(0.0).cuda(self.gpu)

            if self.args.kd_s_alpha > 0:
                log_probs_x_ulb_s = F.log_softmax(logits_x_ulb_s / self.temperature, dim=1)
                sup_loss_s_kd = F.kl_div(log_probs_x_ulb_s, distill_scores_ulb, reduction='batchmean')
            else:
                sup_loss_s_kd = torch.tensor(0.0).cuda(self.gpu)

            with torch.no_grad():                
                logits_x_ulb = self.model(x_ulb)['logits']
            acc_pl = torch.mean((torch.argmax(logits_x_ulb, axis=1) == pl_ulb).float()).item()
            acc_gt = torch.mean((torch.argmax(logits_x_ulb, axis=1) == y_true_ulb).float()).item()

            loss = (self.w_alpha * sup_loss_w) + (self.s_alpha * sup_loss_s) + (self.kd_w_alpha * sup_loss_w_kd) + (self.kd_s_alpha * sup_loss_s_kd)

            # Optional robustness loss added on top of PET loss
            robust_alignment = torch.tensor(0.0, device=logits_x_ulb_w.device if self.args.w_alpha > 0 else logits_x_ulb_s.device)
            robust_robustness = torch.tensor(0.0, device=robust_alignment.device)
            if self.robust_enable and self.robust_ulb2centroid:
                # choose logits to regularize (prefer weak views)
                if self.args.w_alpha > 0:
                    unlabeled_logits = logits_x_ulb_w
                elif self.args.s_alpha > 0:
                    unlabeled_logits = logits_x_ulb_s
                else:
                    unlabeled_logits = self.model(x_ulb)['logits']

                member_centroids = []
                for idx in idx_ulb.tolist():
                    if int(idx) not in self.robust_ulb2centroid:
                        member_centroids.append(None)
                    else:
                        member_centroids.append(self.robust_ulb2centroid[int(idx)])

                if all(c is None for c in member_centroids):
                    robust_alignment = torch.tensor(0.0, device=unlabeled_logits.device)
                    robust_robustness = torch.tensor(0.0, device=unlabeled_logits.device)
                else:
                    unique_centroids = [c for c in sorted(set(c for c in member_centroids if c is not None))]
                    centroid_to_local = {c: i for i, c in enumerate(unique_centroids)}
                    centroid_logits = torch.stack(
                        [self.robust_centroid_logits[c].to(unlabeled_logits.device) for c in unique_centroids]
                    )
                    centroid_targets = torch.tensor(
                        [self.robust_lb_targets[c] for c in unique_centroids], device=unlabeled_logits.device, dtype=torch.long
                    )

                    batch_grouping = {centroid_to_local[c]: [] for c in unique_centroids}
                    for pos, c in enumerate(member_centroids):
                        if c is None:
                            continue
                        batch_grouping[centroid_to_local[c]].append(pos)

                    unlabeled_size = max(unlabeled_logits.shape[0], 1)
                    for local_c, member_ids in batch_grouping.items():
                        if len(member_ids) == 0:
                            continue
                        centroid_probs = centroid_logits[local_c].unsqueeze(0).expand(len(member_ids), -1)
                        member_probs = unlabeled_logits[member_ids]
                        in_cluster_distance = torch.norm(centroid_probs - member_probs, dim=1, p=2).sum()
                        robust_alignment = robust_alignment + in_cluster_distance / unlabeled_size

                        if len(member_ids) > 1:
                            cluster_probs = member_probs
                            cluster_size = len(member_ids)
                            pairwise_distance = 0.0
                            for i in range(cluster_size):
                                for j in range(i + 1, cluster_size):
                                    pairwise_distance += torch.norm(cluster_probs[i] - cluster_probs[j], p=2)
                            robust_robustness = robust_robustness + pairwise_distance / (cluster_size * unlabeled_size)

                    robust_loss = self.lambda_1 * robust_alignment + self.lambda_2 * robust_robustness
                    loss = loss + robust_loss

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(
            sup_loss=sup_loss_w.item() + sup_loss_s.item(),
            sup_loss_w=sup_loss_w.item(),
            sup_loss_s=sup_loss_s.item(),
            sup_loss_w_kd=sup_loss_w_kd.item(),
            sup_loss_s_kd=sup_loss_s_kd.item(),
            acc_pl=acc_pl,
            acc_gt=acc_gt,
            robust_alignment=robust_alignment.item() if isinstance(robust_alignment, torch.Tensor) else 0.0,
            robust_robustness=robust_robustness.item() if isinstance(robust_robustness, torch.Tensor) else 0.0,
        )
        return out_dict, log_dict


    def _check_stop(self):
        if self.it >= self.num_train_iter*self.n_rounds:
            self.print_fn("***************** Iteration limit reached")
            return True
        return False


    def train(self):
        if self.args.pretrained_path != '':
            self.print_fn(f"Using pretrained model from {self.args.pretrained_path}, performing initial evaluation")
            eval_dict = self.evaluate('eval')
            self.print_fn(f"Initial eval: {pprint.pformat(eval_dict)}")

        self.model.train()
        self.call_hook("before_run")
        _t_start = time.time()
        for i_round in range(self.n_rounds):
            self.print_fn(f"Round {i_round+1}/{self.n_rounds}")
            for epoch in range(self.start_epoch, self.epochs):
                t1 = time.time()
                self.epoch = epoch
                if self._check_stop():
                    print("***************** Stopped")
                    break


                self.print_fn(f"Epoch {self.epoch}/{self.epochs}")

                self.call_hook("before_train_epoch")
                if self._robust_base_seed is not None:
                    self._robust_rng = np.random.RandomState(int(self._robust_base_seed) + int(self.rank) + int(self.epoch))
                if self.robust_enable and (self.grouping_update_interval == 0 or epoch % self.grouping_update_interval == 0):
                    self._refresh_robust_grouping()
                ulb_iter = iter(self.loader_dict['train_ulb'])
                lb_iter = iter(self.loader_dict['train_lb'])

                pl = self.pl[i_round]['pl']
                y_true = self.pl[i_round]['y_true']
                distill_scores = self.pl[i_round]['distill_scores']
                pl = torch.tensor(pl).long().cuda(self.gpu)
                y_true = torch.tensor(y_true).long().cuda(self.gpu)
                distill_scores = torch.tensor(distill_scores).cuda(self.gpu)


                try:
                    for data_lb in lb_iter:
                        # prevent the training iterations exceed args.num_train_iter
                        if self._check_stop():
                            print("***************** Stopped")
                            break

                        data_ulb = None
                        if self.robust_enable:
                            ulb_positions = self._sample_ulb_positions_for_lb(data_lb['idx_lb'])
                            if ulb_positions is not None:
                                data_ulb = self._build_ulb_batch_from_positions(ulb_positions)
                        if data_ulb is None:
                            try:
                                data_ulb = next(ulb_iter)
                            except StopIteration:
                                ulb_iter = iter(self.loader_dict['train_ulb'])
                                data_ulb = next(ulb_iter)

                        idx_ulb = data_ulb['idx_ulb']

                        pl_ulb = pl[idx_ulb]
                        distill_scores_ulb = distill_scores[idx_ulb]
                        y_true_ulb = y_true[idx_ulb]

                        _data_ulb = {**data_ulb, 
                                    'pl_ulb': pl_ulb,
                                    'distill_scores_ulb': distill_scores_ulb, 'y_true_ulb': y_true_ulb}
                    
                        self.call_hook("before_train_step")

                        # import pdb; pdb.set_trace()
                        self.out_dict, self.log_dict = self.train_step(**self.process_batch(**_data_ulb))

                        self.n_data_lb = 0
                        self.n_data_ulb = len(idx_ulb)
                        self.call_hook("after_train_step")
                        self.it += 1
                finally:
                    pass
                
                print(f"############ Epoch {self.epoch}:")

                self.call_hook("after_train_epoch")
                t9 = time.time()
                self.print_fn(f"Time taken for epoch {self.epoch}: {t9-t1}")
            if i_round < self.n_rounds - 1:
                self.hooks_dict['PETHook'].update_pl(self, i_round)
                self.reset_model()
                self.reset_optim()
                self.start_epoch = 0
        self.call_hook("after_run")
        self.print_fn(f"Total time taken: {time.time()-_t_start}")
