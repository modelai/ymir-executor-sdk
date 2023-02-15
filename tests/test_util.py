import math
import os
import random
import shutil
import unittest

import yaml
from easydict import EasyDict as edict
from ymir_exc import result_writer, settings
from ymir_exc.util import (YmirStage, YmirStageWeight, filter_saved_files,
                           get_bool, get_merged_config, get_weight_files,
                           get_ymir_process, write_ymir_training_result)


def test_get_ymir_process():
    weights = YmirStageWeight()
    w0, w1, _ = weights.weights
    assert math.isclose(sum(weights.weights), 1)
    for stage, stage_init, stage_weight in zip(
        [YmirStage.PREPROCESS, YmirStage.TASK, YmirStage.POSTPROCESS],
        [0, w0, w0 + w1],
        weights.weights,
    ):
        for _ in range(5):
            p = random.random()
            x = get_ymir_process(stage, p=p)
            assert math.isclose(x, stage_init + p * stage_weight)
            assert 0 <= x <= 1

    for stage, stage_init, stage_weight in zip(
        [YmirStage.PREPROCESS, YmirStage.TASK, YmirStage.POSTPROCESS],
        [0, w0, w0 + w1],
        weights.weights,
    ):
        for _ in range(5):
            p = random.random()
            x = get_ymir_process(stage, p=p, task_idx=0, task_num=2)
            assert math.isclose(x, 0.5 * (stage_init + p * stage_weight))
            assert 0 <= x <= 0.5

            x = get_ymir_process(stage, p=p, task_idx=1, task_num=2)
            assert math.isclose(x, 0.5 + 0.5 * (stage_init + p * stage_weight))
            assert 0.5 <= x <= 1


def test_get_bool():
    cfg = edict()
    cfg.param = edict()
    cfg.param.a = 0
    cfg.param.b = 1
    cfg.param.false = "false"
    cfg.param.true = "true"
    cfg.param.c = False
    cfg.param.d = True
    cfg.param.f = "F"
    cfg.param.t = "T"
    cfg.param.h = "False"
    cfg.param.i = "True"
    for key in ["a", "false", "c", "f", "h"]:
        assert not get_bool(cfg, key)

    for key in ["b", "true", "d", "t", "i"]:
        assert get_bool(cfg, key)

    assert get_bool(cfg, "undefine", True)
    assert not get_bool(cfg, "undefine", False)


class TestWriteResult(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:  # type: ignore
        super().__init__(methodName)
        self._test_root = os.path.join("/tmp", "test_tmi", *self.id().split(".")[-3:])
        self._custom_env_file = os.path.join(self._test_root, "in", "env.yml")
        self._custom_config_file = os.path.join(self._test_root, "in", "config.yaml")
        self._training_result_file = os.path.join(self._test_root, "out", "models", 'result.yaml')
        self._mining_result_file = os.path.join(self._test_root, "out", "mining-result.tsv")
        self._infer_result_file = os.path.join(self._test_root, "out", "infer-result.json")
        self._monitor_file = os.path.join(self._test_root, "out", "monitor.txt")

        settings.DEFAULT_ENV_FILE_PATH = self._custom_env_file
        settings.DEFAULT_CONFIG_FILE_PATH = self._custom_config_file

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

    def test_write_ymir_training_result(self) -> None:

        def check_training_result(cfg):
            with open(cfg.ymir.output.training_result_file, "r") as f:
                training_result = yaml.safe_load(stream=f)

            if result_writer.multiple_model_stages_supportable():
                assert "model_stages" in training_result.keys()
            else:
                assert "model" in training_result.keys()
                assert "map" in training_result.keys()

        cfg = get_merged_config()

        os.makedirs(cfg.ymir.output.models_dir, exist_ok=True)

        for i in range(3):
            files = [f"checkpoint_{i}.pth", "config_{i}.json"]
            map50 = random.random()
            id = f"epoch_{i}"

            for f in files:
                with open(os.path.join(cfg.ymir.output.models_dir, f), 'w') as fw:
                    fw.write(f'{f}\n')
            write_ymir_training_result(cfg, map50=map50, files=files, id=id)
            check_training_result(cfg)

        write_ymir_training_result(cfg, map50=0, id="last", files=[])
        check_training_result(cfg)

    def test_get_weight_files(self) -> None:
        cfg = get_merged_config()

        # test for training with relative path
        cfg.ymir.run_training = True
        result_files = ['epoch10.pth', 'epoch20.pth']
        cfg.param.pretrained_model_params = result_files + ['other.py', 'other.txt']
        os.makedirs(cfg.ymir.input.models_dir, exist_ok=True)

        for f in cfg.param.pretrained_model_params:
            weight_file = os.path.join(cfg.ymir.input.models_dir, f)
            with open(weight_file, 'w') as fw:
                fw.write(f'fake {f}\n')

        model_params_path = get_weight_files(cfg, suffix=(".pth", ".pt"))
        model_params_basename = [os.path.basename(f) for f in model_params_path]
        assert set(model_params_basename) == set(result_files)

        # test for mining and infer with abusolute path
        cfg.ymir.run_training = False
        result_files = [os.path.join(cfg.ymir.input.models_dir, f) for f in result_files]
        cfg.param.model_params_path = result_files + [
            os.path.join(cfg.ymir.input.models_dir, f) for f in ['other.py', 'other.txt']
        ]

        model_params_path = get_weight_files(cfg, suffix=(".pth", ".pt"))
        assert set(model_params_path) == set(result_files)

    def test_filter_saved_files(self) -> None:
        cfg = get_merged_config()

        out_dir = cfg.ymir.output.models_dir
        os.makedirs(out_dir, exist_ok=True)

        rel_files = ['epoch10.pth', 'epoch20.pth', 'a.py', 'b.txt', 'c.jpg']
        abs_files = [os.path.join(out_dir, f) for f in rel_files]

        for f in abs_files:
            with open(f, 'w') as fw:
                fw.write(f'fake {f}\n')

        cfg.param.ymir_saved_file_patterns = ""

        # check all files without filter
        all_rel_files = filter_saved_files(cfg, [])
        assert set(rel_files) == set(all_rel_files)

        # check all files with single filter *.txt
        cfg.param.ymir_saved_file_patterns = ".*.txt"
        txt_rel_files = filter_saved_files(cfg, [])
        assert set(txt_rel_files) == set(['b.txt'])

        # check all files with single filter *.txt
        cfg.param.ymir_saved_file_patterns = "a.*"
        txt_rel_files = filter_saved_files(cfg, [])
        self.assertEqual(set(txt_rel_files), set(['a.py']))

        # check all files with multiple filter
        cfg.param.ymir_saved_file_patterns = ".*.txt, .*.jpg"
        filtered_rel_files = filter_saved_files(cfg, [])
        self.assertEqual(set(filtered_rel_files), set(['b.txt', 'c.jpg']))

        # check files with single filter '*.jpg'
        cfg.param.ymir_saved_file_patterns = ".*.jpg"
        jpg_rel_files = filter_saved_files(cfg, ['a.py', 'b.txt', 'c.jpg'])
        self.assertEqual(set(jpg_rel_files), set(['c.jpg']))

        # check relative files with multiple filter
        cfg.param.ymir_saved_file_patterns = ".*.txt, .*.jpg"
        filtered_rel_files = filter_saved_files(cfg, rel_files)
        self.assertEqual(set(filtered_rel_files), set(['b.txt', 'c.jpg']))

        # check absolute files with multiple filter
        cfg.param.ymir_saved_file_patterns = ".*.txt, .*.jpg"
        filtered_rel_files = filter_saved_files(cfg, abs_files)
        self.assertEqual(set(filtered_rel_files), set(['b.txt', 'c.jpg']))

        # check with invalid regular expression pattern
        cfg.param.ymir_saved_file_patterns = ".*.txt, .*.jpg, .*.png"
        filtered_rel_files = filter_saved_files(cfg, abs_files)
        self.assertEqual(set(filtered_rel_files), set(['b.txt', 'c.jpg']))
