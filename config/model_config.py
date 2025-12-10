model_configs = {
    "yolov8n_baseline_mod": {
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
            "description": "Fast inference, good for real-time",
            "ran-on": "modified dataset"
        }
    },

    "yolov8s_baseline_mod": {
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
            "description": "Good balance of speed and accuracy",
            "ran-on": "modified dataset"
        }
    },

    "yolov8s_adam_optimizer_mod": {
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
            "description": "Testing Adam optimizer",
            "ran-on": "modified dataset"
        }
    },
}
