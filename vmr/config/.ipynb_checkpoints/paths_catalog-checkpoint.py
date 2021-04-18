"""Centralized catalog of paths."""
import os

class DatasetCatalog(object):
    DATA_DIR = "datasets"

    DATASETS = {
        # TACoS
        "tacos_train":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/train.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_val":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/val.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_test":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/test.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_train_rp_noun":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/train_rp_noun.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_val_rp_noun":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/val_rp_noun.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_test_rp_noun":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/test_rp_noun.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_train_rp_verb":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/train_rp_verb.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_val_rp_verb":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/val_rp_verb.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_test_rp_verb":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/test_rp_verb.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_train_rp_verb_noun":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/train_rp_verb_noun.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_val_rp_verb_noun":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/val_rp_verb_noun.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_test_rp_verb_noun":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/test_rp_verb_noun.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_train_mask_vn":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/train_mask_vn.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_val_mask_vn":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/val_mask_vn.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_test_mask_vn":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/test_mask_vn.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },

        "tacos_train_shuffle":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/train_shuffle.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_val_shuffle":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/val_shuffle.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_test_shuffle":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/test_shuffle.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },

        "tacos_train_del_vn":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/train_del_vn.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_val_del_vn":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/val_del_vn.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_test_del_vn":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/test_del_vn.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },


        # ActivityNet
        "activitynet_train":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_test.hdf5",
        },
        "activitynet_train_del_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_del_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_del_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_del_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_del_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_del_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_test.hdf5",
        },
        "activitynet_train_del_verb":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_del_verb.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_del_verb":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_del_verb.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_del_verb":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_del_verb.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_test.hdf5",
        },
        "activitynet_train_del_verb_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_del_verb_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_del_verb_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_del_verb_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_del_verb_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_del_verb_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_test.hdf5",
        },
        #
        "activitynet_train_rp_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_rp_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_rp_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_rp_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_rp_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_rp_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_test.hdf5",
        },
        "activitynet_train_rp_verb":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_rp_verb.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_rp_verb":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_rp_verb.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_rp_verb":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_rp_verb.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_test.hdf5",
        },
        "activitynet_train_rp_verb_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_rp_verb_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_rp_verb_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_rp_verb_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_rp_verb_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_rp_verb_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_test.hdf5",
        },
        #
        "activitynet_train_only_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_only_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_only_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_only_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_only_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_only_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_test.hdf5",
        },
        "activitynet_train_only_verb":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_only_verb.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_only_verb":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_only_verb.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_only_verb":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_only_verb.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_test.hdf5",
        },
        "activitynet_train_only_verb_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_only_verb_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_only_verb_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_only_verb_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_only_verb_noun":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_only_verb_noun.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_test.hdf5",
        },

        "activitynet_train_mask_vn":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_mask_vn.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_mask_vn":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_mask_vn.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_mask_vn":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_mask_vn.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
        },

        "activitynet_train_del_vn":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_del_vn.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_del_vn":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_del_vn.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_del_vn":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_del_vn.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
        },

        "activitynet_train_shuffle":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train_shuffle.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_train.hdf5",
        },
        "activitynet_val_shuffle":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val_shuffle.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d_val.hdf5",
        },
        "activitynet_test_shuffle":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test_shuffle.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
        },

        # Charades
        "charades_train":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
            #"feat_file": "charades_sta/features/vgg_rgb_features.hdf5",
        },
        "charades_test":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
            #"feat_file": "charades_sta/features/vgg_rgb_features.hdf5",
        },
        "charades_train_del_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_del_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_test_del_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_del_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_train_del_verb":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_del_verb.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_test_del_verb":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_del_verb.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_train_del_verb_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_del_verb_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_test_del_verb_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_del_verb_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_train_only_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_only_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_test_only_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_only_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_train_only_verb":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_only_verb.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_test_only_verb":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_only_verb.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_train_only_verb_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_only_verb_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_test_only_verb_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_only_verb_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_train_rp_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_rp_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_test_rp_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_rp_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_train_rp_verb":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_rp_verb.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_test_rp_verb":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_rp_verb.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_train_rp_verb_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_rp_verb_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_test_rp_verb_noun":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_rp_verb_noun.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },

        "charades_test_mask_vn":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_mask_vn.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_train_mask_vn":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_mask_vn.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },

        "charades_test_del_vn":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_del_vn.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_train_del_vn":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_del_vn.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },

        "charades_test_shuffle":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/test_shuffle.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
        "charades_train_shuffle":{
            "video_dir": "charades_sta/videos",
            "ann_file": "charades_sta/annotations/train_shuffle.json",
            "feat_file": "charades_sta/features/i3d_features.hdf5",
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            root=os.path.join(data_dir, attrs["video_dir"]),
            anno_file=os.path.join(data_dir, attrs["ann_file"]),
            feat_file=os.path.join(data_dir, attrs["feat_file"]),
        )
        if "tacos" in name:
            return dict(
                factory="TACoSDataset",
                args=args,
            )
        elif "activitynet" in name:
            return dict(
                factory = "ActivityNetDataset",
                args = args
            )
        elif "charades" in name:
            return dict(
                factory = "CharadesDataset",
                args = args
            )
        raise RuntimeError("Dataset not available: {}".format(name))
