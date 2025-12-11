import mlflow
import os
from dotenv import load_dotenv
load_dotenv()

client = mlflow.tracking.MlflowClient()

# Model 1
try:
    client.transition_model_version_stage(
        name="litter-detection_mod",
        version="4",
        stage="Production"
    )
    print("✅ litter-detection_mod v4 → Production")
except Exception as e:
    print(f"❌ litter-detection_mod failed: {e}")

# Model 2
try:
    client.transition_model_version_stage(
        name="litter-detection_org",
        version="6",
        stage="Production"
    )
    print("✅ litter-detection_org v6 → Production")
except Exception as e:
    print(f"❌ litter-detection_org failed: {e}")


# Verify
for model_name in ["litter-detection_mod", "litter-detection_org"]:
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if prod_versions:
        print(f"✅ {model_name} is now in Production (v{prod_versions[0].version})")
    else:
        print(f"❌ {model_name} NOT in Production")
