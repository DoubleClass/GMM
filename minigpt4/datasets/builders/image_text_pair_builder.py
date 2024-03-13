import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.laion_dataset import LaionDataset
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset


@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self, task_id):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        cap_name = '10_10/filter_cap_10_10_task' + str(task_id) + '.json'
        # cap_name = '10_10/filter_cap_10_10_task0.json'

        cap_name = '20_20_wop/task' + str(task_id) +'.json'

        cap_name = '10_10_wop/task' + str(task_id) +'.json'

        cap_name = '10_10_wop/task' + str(task_id) +'.json'
        cap_name = '5_5_wo_exp/task' + str(task_id) +'.json'
        cap_name = '10_10_wop_few30/task' + str(task_id) +'.json'
        cap_name = 'old_10_10_wop_few200/task' + str(task_id) +'.json'
        # cap_name = 'old_10_10_wop_few50/task' + str(task_id) +'.json'
        # cap_name = 'old_10_10_wop_few25/task' + str(task_id) +'.json'
        # cap_name = '10_10_wop/task' + str(task_id) +'.json'
        # cap_name = '10_10_w2000_exp_order1_all/task' + str(task_id) +'.json'
        # cap_name = '10_10_wop/task' +  str(task_id) +'.json'
        # cap_name = '20_20_order3_2000exp/task' + str(task_id) +'.json'

        # cap_name = '100_20_order1_wo_exp_all/task' + str(task_id) + '.json'
        # cap_name = '100_20_order3_wo_exp_all/task' + str(task_id) + '.json'
        # cap_name = '100_5_order2_w2000exp_all/task' + str(task_id) + '.json'
        # # cap_name = '100_20_order1_w2000exp_all/task' + str(task_id) + '.json'

        cap_name = '20_20_order3_2000exp/task' + str(task_id) + '.json'

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            # ann_paths=[os.path.join(storage_path, 'filter_cap_first50_task5.json')],
            ann_paths=[os.path.join(storage_path, cap_name)],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets
