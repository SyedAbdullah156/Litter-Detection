model_configs = {
    "yolov8n_baseline": {
        "model_name": "yolov8n.pt",
        "epochs": 100,
        "batch_size": 16,
        "img_size": 640,
        "optimizer": "SGD",
        "lr": 0.01,
        "device": "cuda",
        "fraction": 1.0,
        "patience": 50,
        "tags": {
            "model_size": "nano",
            "strategy": "baseline",
            "description": "Fast inference, good for real-time"
        }
    },

    "yolov8s_baseline": {
        "model_name": "yolov8s.pt",
        "epochs": 100,
        "batch_size": 16,
        "img_size": 640,
        "optimizer": "SGD",
        "lr": 0.01,
        "device": "cuda",
        "fraction": 1.0,
        "patience": 50,
        "tags": {
            "model_size": "small",
            "strategy": "baseline",
            "description": "Good balance of speed and accuracy"
        }
    },

    "yolov8s_adam_optimizer": {
        "model_name": "yolov8s.pt",
        "epochs": 100,
        "batch_size": 16,
        "img_size": 640,
        "optimizer": "Adam",
        "lr": 0.001,
        "device": "cuda",
        "fraction": 1.0,
        "patience": 50,
        "tags": {
            "model_size": "small",
            "strategy": "adam_optimizer",
            "description": "Testing Adam optimizer"
        }
    },

    "yolov8s_heavy_augmentation": {
        "model_name": "yolov8s.pt",
        "epochs": 100,
        "batch_size": 16,
        "img_size": 640,
        "optimizer": "SGD",
        "lr": 0.01,
        "device": "cuda",
        "fraction": 1.0,
        "patience": 50,
        "augmentation_config": {
            "hsv_h": 0.03,
            "hsv_s": 0.9,
            "hsv_v": 0.6,
            "degrees": 15.0,
            "translate": 0.2,
            "scale": 0.7,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.2
        },
        "tags": {
            "model_size": "small",
            "strategy": "heavy_augmentation",
            "description": "Strong augmentation for generalization"
        }
    },

    "yolov8s_adamw_tuned": {
        "model_name": "yolov8s.pt",
        "epochs": 100,
        "batch_size": 16,
        "img_size": 640,
        "optimizer": "AdamW",
        "lr": 0.0005,
        "device": "cuda",
        "fraction": 1.0,
        "patience": 50,
        "augmentation_config": {
            "hsv_h": 0.02,
            "hsv_s": 0.8,
            "hsv_v": 0.5,
            "degrees": 10.0,
            "translate": 0.15,
            "scale": 0.6,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.1
        },
        "tags": {
            "model_size": "small",
            "strategy": "adamw_tuned",
            "description": "AdamW with balanced augmentation"
        }
    }
}
