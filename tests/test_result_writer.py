import json
import math
import os
import shutil
import unittest
from typing import Dict, List, Tuple

import yaml
from ymir_exc import result_writer as rw
from ymir_exc import settings


class TestResultWriter(unittest.TestCase):
    # life cycle
    def __init__(self, methodName: str = ...) -> None:  # type: ignore
        super().__init__(methodName)
        self._test_root = os.path.join("/tmp", "test_tmi", *self.id().split(".")[-3:])
        self._custom_env_file = os.path.join(self._test_root, "env.yml")
        self._training_result_file = os.path.join(self._test_root, "out", "training-result.yaml")
        self._mining_result_file = os.path.join(self._test_root, "out", "mining-result.tsv")
        self._infer_result_file = os.path.join(self._test_root, "out", "infer-result.json")
        self._monitor_file = os.path.join(self._test_root, "out", "monitor.txt")

    def setUp(self) -> None:
        settings.DEFAULT_ENV_FILE_PATH = self._custom_env_file
        self._prepare_dirs()
        self._prepare_env_config()
        return super().setUp()

    def tearDown(self) -> None:
        # self._deprepare_dirs()
        return super().tearDown()

    # protected: setup and teardown
    def _prepare_dirs(self) -> None:
        if os.path.isdir(self._test_root):
            shutil.rmtree(self._test_root)
        os.makedirs(self._test_root)
        os.makedirs(os.path.join(self._test_root, "out"), exist_ok=True)

    def _prepare_env_config(self) -> None:
        env_obj = {
            "task_id": "task0",
            "protocol_version": "2.0.0",
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

    def _deprepare_dirs(self) -> None:
        if os.path.isdir(self._test_root):
            shutil.rmtree(self._test_root)

    # protected: check results
    def _check_training_result(self, files: List[str], mAP: float) -> None:
        with open(self._training_result_file, "r") as f:
            result_obj = yaml.safe_load(f)
            self.assertEqual(result_obj["mAP"], mAP)

    def _check_mining_result(self, mining_result: List[Tuple[str, float]]) -> None:
        with open(self._mining_result_file, "r") as f:
            lines = f.read().splitlines()
            self.assertEqual(len(lines), len(mining_result))
            self.assertEqual(lines[0], "b\t0.3")
            self.assertEqual(lines[1], "c\t0.2")
            self.assertEqual(lines[2], "a\t0.1")

    def _check_infer_result(self, infer_result: Dict[str, List[rw.Annotation]]) -> None:
        with open(self._infer_result_file, "r") as f:
            infer_result_obj = json.loads(f.read())
            self.assertEqual(set(infer_result_obj["detection"].keys()), set(infer_result.keys()))

    def test_write_training_result(self) -> None:
        files = ["model-symbols.json", "model-0000.params"]
        mAP = 0.86
        rw.write_training_result(stage_name='best', files=files, evaluation_result=dict(mAP=mAP))  # type: ignore
        self._check_training_result(
            files=files,
            mAP=mAP,
        )  # type ignore

    def test_write_model_stage(self) -> None:
        stage_names = ['epoch001', 'epoch002', 'epoch003']
        files = ['best.pt', 'config.py']

        N = len(stage_names)
        for metric in ['mAP', 'mIoU', 'maskAP']:
            # remove training result file
            if os.path.exists(self._training_result_file):
                os.system(f'rm {self._training_result_file}')

            for idx, stage_name in enumerate(stage_names):
                evaluation_result = {metric: idx / len(stage_names)}
                rw.write_model_stage(stage_name=stage_name, files=files, evaluation_result=evaluation_result)

            with open(self._training_result_file, 'r') as fr:
                result_obj = yaml.safe_load(fr)

            self.assertTrue(math.isclose(result_obj[metric], (N - 1) / N))
            self.assertEqual(result_obj['best_stage_name'], stage_names[-1])
            for stage_name in result_obj['model_stages']:
                self.assertIn(stage_name, stage_names)
                m = result_obj['model_stages'][stage_name][metric]
                n = stage_names.index(stage_name) / N
                self.assertTrue(math.isclose(m, n))

    def test_write_mining_result(self) -> None:
        mining_result = [("a", 0.1), ("b", 0.3), ("c", 0.2)]
        rw.write_mining_result(mining_result=mining_result)
        self._check_mining_result(mining_result=mining_result)

    def test_write_infer_result(self) -> None:
        infer_result = {
            "a": [
                rw.Annotation(box=rw.Box(x=0, y=0, w=50, h=50), class_name="cat", score=0.2),
                rw.Annotation(box=rw.Box(x=150, y=0, w=50, h=50), class_name="person", score=0.3),
            ],
            "b": [rw.Annotation(box=rw.Box(x=0, y=0, w=50, h=150), class_name="person", score=0.2)],
            "c": [],
        }
        rw.write_infer_result(infer_result=infer_result)  # type: ignore
        self._check_infer_result(infer_result=infer_result)  # type: ignore

    def test_write_segmentation_infer_result(self) -> None:
        infer_result = {'result': 'fake infer result'}
        rw.write_infer_result(infer_result=infer_result, algorithm='segmentation')

        with open(self._infer_result_file, "r") as f:
            infer_result_obj = json.loads(f.read())
            self.assertEqual(set(infer_result_obj.keys()), set(infer_result.keys()))
