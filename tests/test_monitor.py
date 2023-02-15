import os
import shutil
import unittest

import yaml
from ymir_exc import env, monitor, settings
from ymir_exc.util import get_merged_config


class TestMonitor(unittest.TestCase):
    # life cycle
    def __init__(self, methodName: str = ...) -> None:  # type: ignore
        super().__init__(methodName)
        self._test_root = os.path.join("/tmp", "test_tmi", *self.id().split(".")[-3:])
        self._custom_env_file = os.path.join(self._test_root, "in", "env.yml")
        self._custom_config_file = os.path.join(self._test_root, "in", 'config.yaml')
        self._training_result_file = os.path.join(self._test_root, "out", "training-result.yaml")
        self._mining_result_file = os.path.join(self._test_root, "out", "mining-result.tsv")
        self._infer_result_file = os.path.join(self._test_root, "out", "infer-result.json")
        self._monitor_file = os.path.join(self._test_root, "out", "monitor.txt")

    def setUp(self) -> None:
        settings.DEFAULT_ENV_FILE_PATH = self._custom_env_file
        settings.DEFAULT_CONFIG_FILE_PATH = self._custom_config_file
        self._prepare_dirs()
        self._prepare_env_config()
        self._prepare_config()
        return super().setUp()

    def tearDown(self) -> None:
        self._deprepare_dirs()
        return super().tearDown()

    # protected: setup and teardown
    def _prepare_dirs(self) -> None:
        if os.path.isdir(self._test_root):
            shutil.rmtree(self._test_root)
        os.makedirs(self._test_root)
        os.makedirs(os.path.join(self._test_root, "in"), exist_ok=True)
        os.makedirs(os.path.join(self._test_root, "out"), exist_ok=True)

    def _prepare_env_config(self) -> None:
        env_obj = {
            "task_id": "task0",
            "protocol_version": '2.0.0',
            "run_infer": False,
            "run_mining": False,
            "run_training": True,
            "input": {
                "root_dir": os.path.join(self._test_root, 'in'),
                "annotations_dir": os.path.join(self._test_root, "in", "annotations"),
                "assets_dir": os.path.join(self._test_root, 'in', 'assets'),
                "config_file": self._custom_config_file,
                "models_dir": os.path.join(self._test_root, 'in', 'models'),
                "training_index_file": os.path.join(self._test_root, 'in', 'training-index.tsv'),
                "candidate_index_file": os.path.join(self._test_root, 'in', 'candidate-index.tsv'),
                "val_index_file": os.path.join(self._test_root, 'in', 'val-index.tsv'),
            },
            "output": {
                "root_dir": os.path.join(self._test_root, "out"),
                "models_dir": os.path.join(self._test_root, "out", "models"),
                "training_result_file": self._training_result_file,
                "mining_result_file": self._mining_result_file,
                "infer_result_file": self._infer_result_file,
                "monitor_file": self._monitor_file,
            },
        }
        with open(self._custom_env_file, "w") as f:
            yaml.safe_dump(env_obj, f)

    def _prepare_config(self) -> None:
        config_obj = {
            "gpu_id": '0',
            "task_id": '1',
            "class_names": ['dog', 'cat'],
            "epochs": 100,
        }

        with open(self._custom_config_file, 'w') as f:
            yaml.safe_dump(config_obj, f)

    def _deprepare_dirs(self) -> None:
        if os.path.isdir(self._test_root):
            shutil.rmtree(self._test_root)

    # protected: check results
    def _check_monitor(self, percent: float) -> None:
        with open(self._monitor_file, "r") as f:
            lines = f.read().splitlines()
        task_id, timestamp_str, percent_str, state_str, *_ = lines[0].split()

        self.assertEqual(task_id, env.get_current_env().task_id)
        self.assertTrue(float(timestamp_str) > 0)
        self.assertEqual(percent, float(percent_str))
        self.assertEqual(2, int(state_str))

    # public: test cases
    def test_write_monitor(self) -> None:
        monitor.write_monitor_logger(percent=0.2)
        self._check_monitor(percent=0.2)

    def test_write_monitor_for_multiple_tasks(self) -> None:
        cfg = get_merged_config()

        for task in ['training', 'mining', 'infer']:
            monitor.write_monitor_logger_for_multiple_tasks(cfg, task, 0.5)
            self._check_monitor(percent=0.5)

        cfg.ymir.run_infer = True
        cfg.ymir.run_mining = True

        percent = 1.0
        for task in ['mining', 'infer']:
            monitor.write_monitor_logger_for_multiple_tasks(cfg, task, percent=percent, order='tmi')
            if task == 'mining':
                self._check_monitor(percent=0.5 * percent)
            else:
                self._check_monitor(percent=0.5 + 0.5 * percent)

        for task in ['mining', 'infer']:
            monitor.write_monitor_logger_for_multiple_tasks(cfg, task, percent=percent, order='tim')
            if task == 'mining':
                self._check_monitor(percent=0.5 + 0.5 * percent)
            else:
                self._check_monitor(percent=0.5 * percent)
