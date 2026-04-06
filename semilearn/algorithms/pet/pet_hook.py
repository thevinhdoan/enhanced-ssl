# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

import copy
from dotwiz import DotWiz
from ruamel.yaml import YAML
import os

import numpy as np
from semilearn.core.hooks import Hook
import pickle
from scipy.special import softmax

_ROOT_DIR = os.path.abspath(os.curdir)
_yaml = YAML()


def _normalize_sample_identifier(sample_path):
    sample_path = os.path.normpath(str(sample_path)).replace("\\", "/")
    marker = "/images/"
    if marker in sample_path:
        return sample_path.split(marker, 1)[1]
    marker = "images/"
    if marker in sample_path:
        return sample_path.split(marker, 1)[1]
    return sample_path


def _config_uses_filtered_classes(config):
    return config.get("selected_classes") is not None or bool(config.get("remap_selected_classes", False))


def _candidate_source_log_paths(source, config):
    candidates = []

    save_dir = config.get('save_dir')
    save_name = config.get('save_name', 'log')
    if save_dir:
        if os.path.isabs(save_dir):
            candidates.append(os.path.join(save_dir, save_name))
        else:
            candidates.append(os.path.join(_ROOT_DIR, save_dir, save_name))

    extra_roots = []
    env_root = os.environ.get("PET_SOURCE_SAVE_ROOT")
    if env_root:
        extra_roots.extend([root for root in env_root.split(os.pathsep) if root])
    if not _config_uses_filtered_classes(config):
        extra_roots.extend([
            "/mnt/extra_storage/users/vinhdt/thesis/dtd01_3/saved_models",
            "/mnt/extra_storage/users/vinhdt/thesis/dtd_3/saved_models",
        ])

    source_rel = source
    if source_rel.startswith("config/"):
        source_rel = source_rel[len("config/"):]
    if source_rel.endswith("/config.yaml"):
        source_rel = source_rel[:-len("/config.yaml")]
    for root in extra_roots:
        candidates.append(os.path.join(root, source_rel, "log"))

    deduped = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _resolve_source_artifact(source, config, artifact_name):
    attempted = []
    for log_path in _candidate_source_log_paths(source, config):
        artifact_path = os.path.join(log_path, artifact_name)
        attempted.append(artifact_path)
        if os.path.exists(artifact_path):
            return artifact_path
    attempted_str = "\n".join(attempted)
    if _config_uses_filtered_classes(config):
        raise AssertionError(
            f'Artifact {artifact_name} for {source} does not exist in the expected two-class teacher outputs. '
            f'Rerun run_two_class_supervised_flow.py / eval.py for that filtered teacher config.\nTried:\n{attempted_str}'
        )
    raise AssertionError(
        f'Artifact {artifact_name} for {source} does not exist. Tried:\n{attempted_str}'
    )


def _predict_sample_identifiers(predict):
    sample_ids = predict.get('sample_relpaths')
    if sample_ids is None:
        sample_ids = predict.get('sample_paths')
    if sample_ids is None:
        return None
    return [_normalize_sample_identifier(sample_id) for sample_id in sample_ids]


def _dataset_sample_identifiers(dataset):
    return [_normalize_sample_identifier(sample_path) for sample_path in dataset.samples]


def _reindex_predict_values(value, ordered_indices):
    if isinstance(value, np.ndarray):
        return value[ordered_indices]
    if torch.is_tensor(value):
        return value[ordered_indices]
    if isinstance(value, list):
        return [value[int(idx)] for idx in ordered_indices]
    return value


def _realign_predict_to_dataset(predict, dataset, source, artifact_name):
    dataset_ids = _dataset_sample_identifiers(dataset)
    predict_ids = _predict_sample_identifiers(predict)

    if predict_ids is None:
        if len(predict['y_logits']) != len(dataset_ids):
            raise AssertionError(
                f'Artifact {artifact_name} for {source} has {len(predict["y_logits"])} samples, '
                f'but current PET train_ulb has {len(dataset_ids)} samples and the artifact does not contain sample_relpaths. '
                'Rerun eval.py for the two-class teacher configs so pl.pkl includes sample_relpaths.'
            )
        return predict

    id_to_idx = {}
    for idx, sample_id in enumerate(predict_ids):
        if sample_id in id_to_idx:
            raise AssertionError(f'Duplicate sample identifier {sample_id} found in {artifact_name} for {source}.')
        id_to_idx[sample_id] = idx

    missing = [sample_id for sample_id in dataset_ids if sample_id not in id_to_idx]
    if missing:
        preview = ", ".join(missing[:5])
        raise AssertionError(
            f'Artifact {artifact_name} for {source} is missing {len(missing)} samples needed by current PET train_ulb. '
            f'Examples: {preview}'
        )

    ordered_indices = np.array([id_to_idx[sample_id] for sample_id in dataset_ids], dtype=np.int64)
    aligned = copy.deepcopy(predict)
    for key in (
        'y_true',
        'y_pred',
        'y_logits',
        'y_feats',
        'sample_relpaths',
        'sample_paths',
        'dataset_indices',
    ):
        if key in aligned:
            aligned[key] = _reindex_predict_values(aligned[key], ordered_indices)
    return aligned


def _validate_predict_num_classes(predict, expected_num_classes, source, artifact_name):
    if 'y_logits' not in predict:
        raise AssertionError(f'Artifact {artifact_name} for {source} is missing y_logits.')
    if predict['y_logits'].shape[1] != expected_num_classes:
        raise AssertionError(
            f'Artifact {artifact_name} for {source} has logits dim {predict["y_logits"].shape[1]}, '
            f'but the current PET run expects {expected_num_classes} classes. '
            'This usually means stale pseudo-label files from a different class setup.'
        )


def _score_to_pl(score):
    # Randomized tie-breaking argmax
    return np.random.choice(np.where(score == score.max())[0])

def scores_to_pl(scores):
    # Randomized tie-breaking argmax
    max_scores = np.max(scores, axis=1)
    candidates = (scores == max_scores[:, None])
    pl = np.argmax(np.random.rand(*candidates.shape) * candidates, axis=1)
    return pl

def sort_conf(conf):
    # Sort conf from low to high, randomize tie-breaking for the same conf
    conf_sort = np.argsort(conf)
    conf_unique = np.unique(conf)
    for _conf in conf_unique:
        indices = np.where(conf[conf_sort] == _conf)[0]
        if len(indices) > 1:
            conf_sort[indices] = np.random.permutation(conf_sort[indices])
    return conf_sort

def get_balanced_pseudo_labels(scores, n_per_cls, mode, _targets, generation):
    assert mode in ['max', 'min', 'rand']
    assert generation == 'hard'
    # Get all classes
    clzes = np.arange(scores.shape[1])
    # Extract pl and conf
    pl = scores_to_pl(scores)
    conf = scores[np.arange(scores.shape[0]), pl]
    # Add noise to confidence to break ties
    conf += 1e-9 * np.random.rand(len(conf))
    # Sort by confidence
    if mode == 'max':
        conf_sort = sort_conf(-conf)
    elif mode == 'min':
        conf_sort = sort_conf(conf)
    elif mode == 'rand':
        conf_sort = np.random.permutation(len(conf))
    sorted_pl = pl[conf_sort]
    # Get indices
    conf_indices = []
    for clz in clzes:
        clz_indices = np.where(sorted_pl == clz)[0]
        assert len(clz_indices) >= n_per_cls, f'Not enough samples for class {clz}. '
        # TODO: 1. balanced pseudo-label when no enough max confidence samples
        conf_indices += list(conf_sort[clz_indices][:n_per_cls])
    conf_indices = np.array(conf_indices)
    return conf_indices, conf, pl

class PETHook(Hook):
    """
    Pseudo Labeling Hook
    """
    def __init__(self):
        super().__init__()
    

    def update_pl(self, algorithm, i_round):
        algorithm.print_fn(f'PETHook: update_pl, i_round: {i_round}')

        y_logits = algorithm.evaluate('train_ulb_oracle', return_logits=True)['train_ulb_oracle/logits']
        y_true = algorithm.pl['y_true']
        y_probs = softmax(y_logits, axis=-1)
        pl = scores_to_pl(y_probs)

        if algorithm.conf_ratio > 0:
            algorithm.print_fn(f'Confidence threshold: {algorithm.conf_ratio}')
            conf = y_probs.max(axis=1)
            # top conf according to conf_threshold
            conf_threshold = np.sort(conf)[::-1][int(len(conf) * algorithm.conf_ratio)]
            conf_mask = conf > conf_threshold
            y_true = y_true[conf_mask]
            y_probs = y_probs[conf_mask]
            pl = pl[conf_mask]
            algorithm.loader_dict['train_ulb'].dataset.update_idx_list(np.where(conf_mask)[0], y_true=y_true, y_pred=pl)
        elif algorithm.conf_threshold > 0:
            assert False
            conf_mask = y_probs.max(axis=1) > algorithm.conf_threshold
            y_true = y_true[conf_mask]
            y_probs = y_probs[conf_mask]
            pl = pl[conf_mask]
            algorithm.loader_dict['train_ulb'].dataset.update_idx_list(np.where(conf_mask)[0], y_true=y_true, y_pred=pl)

        pl_i = {
            'distill_scores': y_probs,
            'pl': pl,
            'y_true': y_true,
        }
        algorithm.print_fn(f'PL accuracy for round {i_round}: {np.mean(pl_i["y_true"] == pl_i["pl"])}')
        
        algorithm.pl[i_round+1] = pl_i
        algorithm.print_fn(f'Number of pseudo labels: {len(pl)}')


    def init_pl(self, algorithm):
        args = algorithm.args
        pet_sources = args.pet_sources
        logits_ensemble = args.logits_ensemble
        pl_selection = args.pl_selection
        predicts = {}
        train_ulb_dataset = algorithm.loader_dict['train_ulb'].dataset
        expected_num_classes = int(args.num_classes)
        
        algorithm.print_fn(f'PET sources: {pet_sources}')
        for source in pet_sources:
            # Read yaml file
            _source = os.path.join(_ROOT_DIR, source)
            with open(_source, 'r') as f:
                config = _yaml.load(f)

            save_dir = config['save_dir']
            save_name = config['save_name']

            log_path = os.path.join(_ROOT_DIR, save_dir, save_name)
            pl_path = _resolve_source_artifact(source, config, 'pl.pkl')

            # Load pseudo labels
            algorithm.print_fn(f'Loading pseudo labels for {source}')
            algorithm.print_fn(f'Using pseudo labels from {pl_path}')

            with open(pl_path, 'rb') as f:
                predict = pickle.load(f)
            predict = _realign_predict_to_dataset(predict, train_ulb_dataset, source, 'pl.pkl')
            _validate_predict_num_classes(predict, expected_num_classes, source, 'pl.pkl')

            predicts[source] = predict
            if args.bootstrapping:
                bootstrapping_pl_path = _resolve_source_artifact(source, config, 'bootstrapping_pl')
                y_logits = np.zeros(predict['y_logits'].shape)
                n = 0
                for file in os.listdir(bootstrapping_pl_path):
                    if not file.startswith('weak_'):
                        continue
                    _source = f"{source}:{file}"
                    pl_path = os.path.join(bootstrapping_pl_path, file)
                    with open(pl_path, 'rb') as f:
                        predict = pickle.load(f)
                    predict = _realign_predict_to_dataset(predict, train_ulb_dataset, source, file)
                    _validate_predict_num_classes(predict, expected_num_classes, source, file)
                    # predicts[_source] = predict
                    y_logits += predict['y_logits']
                    n += 1
                assert n == 10, 'Bootstrapping error!'
                y_logits /= n
                predicts[source] = copy.deepcopy(predicts[source])
                predicts[source]['y_logits'] = y_logits
            
        assert not (algorithm.conf_threshold > 0 and len(predicts) > 1), 'Confidence threshold only works for single source. '

        algorithm.pl['y_true'] = predicts[list(predicts.keys())[0]]['y_true']

        if algorithm.conf_threshold > 0:
            pl_k = list(predicts.keys())[0]
            predict = predicts[pl_k]
            y_true = predict['y_true']
            y_pred = predict['y_pred']
            y_logits = predict['y_logits']
            y_probs = softmax(y_logits, axis=-1)
            conf_mask = y_probs.max(axis=1) > algorithm.conf_threshold

            predict['y_true'] = y_true[conf_mask]
            predict['y_pred'] = y_pred[conf_mask]
            predict['y_logits'] = y_logits[conf_mask]
            # algorithm.loader_dict['train_ulb'].dataset.init_samples(np.where(conf_mask)[0])
            # train_ulb_dset = algorithm.loader_dict['train_ulb'].dataset
            algorithm.loader_dict['train_ulb'].dataset.update_idx_list(np.where(conf_mask)[0], y_true=y_true, y_pred=y_pred)


        for k, v in predicts.items():
            y_pred = v['y_pred']
            y_true = v['y_true']
            algorithm.print_fn(f'PL accuracy for {k}: {np.mean(y_true == y_pred)}')
        
        logits = np.array([v['y_logits'] for v in predicts.values()])

        if logits_ensemble == 'voting':
            vote = np.argmax(logits, axis=-1)
            scores = np.zeros((logits.shape[1], logits.shape[2]))
            for i, _vote in enumerate(vote):
                # Sanity check version
                scores[np.arange(scores.shape[0]), _vote] += 1
                assert np.all(np.sum(scores, axis=1) == i+1), 'Voting error!'    
            distill_scores = softmax(scores, axis=-1)
        elif logits_ensemble == 'mean':
            scores = np.mean(logits, axis=0)
            distill_scores = softmax(scores, axis=-1)
        elif logits_ensemble == 'mean_softmax':
            scores = np.mean(softmax(logits, axis=-1), axis=0)
            distill_scores = copy.deepcopy(scores)

        # get idx and y_true
        y_true = predicts[list(predicts.keys())[0]]['y_true']

        # Sanity check
        for _y_true in np.array([v['y_true'] for v in predicts.values()]):
            assert np.all(y_true == _y_true), 'Pseudo label error!'

        # Format pl
        pl = np.empty(scores.shape[0], dtype=np.int32)
        for _idx, (_y, score) in enumerate(zip(y_true, scores)):
            pl[_idx] = _score_to_pl(score)
        
        algorithm.pl[0] = {
            'distill_scores': distill_scores,
            'pl': pl, 
            'y_true': y_true, 
        }
        algorithm.print_fn(f'Number of pseudo labels: {len(pl)}')
        algorithm.print_fn(f'PL accuracy for ensemble: {np.mean(algorithm.pl[0]["y_true"] == algorithm.pl[0]["pl"])}')
        algorithm.print_fn(f'distill_scores shape: {algorithm.pl[0]["distill_scores"].shape}')
        # Sanity check
        for k, v in predicts.items():
            assert np.all(v['y_true'] == algorithm.pl[0]['y_true']), 'Pseudo label error!'
        
        if args.with_labeled:
            lb_loader = algorithm.loader_dict['train_lb']
            ulb_loader = algorithm.loader_dict['train_ulb']
            lb_dset = algorithm.loader_dict['train_lb'].dataset
            ulb_dset = algorithm.loader_dict['train_ulb'].dataset
            pl_with_labeled = copy.deepcopy(pl)
            lb_tot = 0
            lb_correct = 0
            for lb_sample, lb_y in zip(lb_dset.samples, lb_dset.targets):
                pl_idx = np.where(ulb_dset.samples == lb_sample)[0][0]
                lb_tot += 1
                if pl_with_labeled[pl_idx] == lb_y:
                    lb_correct += 1
                pl_with_labeled[pl_idx] = lb_y
            print(f"Accuracy with labeled: {np.mean(pl_with_labeled == y_true)}")
            print(f"PL labeled accuracy: {lb_correct / lb_tot}")
            algorithm.pl[0]['pl'] = pl_with_labeled

    def before_run(self, algorithm):
        algorithm.print_fn('PETHook: before_run')
        self.init_pl(algorithm)
