# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MyPath(object):
    """
    User-specific path configuration.
    """

    _DEFAULT_DB_ROOT = '/home/jy/multi_task_datasets'
    _DEFAULT_DATASET_ROOTS = {
        'PASCAL_MT': '/home/jy/multi_task_datasets/PASCAL_MT',
        'NYUD_MT': '/home/jy/multi_task_datasets/NYUD_MT',
        'cityscapes': '/home/jy/multi_task_datasets/cityscapes',
    }
    _DEFAULT_SEISM_ROOT = '/path/to/seism/'
    _path_config = {}

    @classmethod
    def set_path_config(cls, path_config):
        """Set path configuration loaded from YAML."""
        cfg = path_config or {}
        # Support both flat format and nested "paths" format.
        if isinstance(cfg.get('paths'), dict):
            merged_cfg = dict(cfg['paths'])
            for key, value in cfg.items():
                if key != 'paths':
                    merged_cfg[key] = value
            cfg = merged_cfg
        cls._path_config = cfg

    @staticmethod
    def _normalize_path(path):
        return os.path.expandvars(os.path.expanduser(path))

    @classmethod
    def db_root_dir(cls, database=''):
        db_root = cls._normalize_path(
            cls._path_config.get('db_root_dir', cls._DEFAULT_DB_ROOT)
        )
        dataset_roots = dict(cls._DEFAULT_DATASET_ROOTS)
        if isinstance(cls._path_config.get('dataset_roots'), dict):
            dataset_roots.update(cls._path_config['dataset_roots'])

        if not database:
            return db_root

        if database in dataset_roots:
            return cls._normalize_path(dataset_roots[database])

        raise NotImplementedError('Unknown dataset root for database: {}'.format(database))

    @classmethod
    def seism_root(cls):
        return cls._normalize_path(
            cls._path_config.get('seism_root', cls._DEFAULT_SEISM_ROOT)
        )
