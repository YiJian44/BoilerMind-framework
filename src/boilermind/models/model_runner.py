import joblib
import numpy as np
from pathlib import Path


class BoilerModelRunner:


    def __init__(self):

        self.model_root = Path(
            r"D:\BoilerMindTeamTest\model_library_20260805\model_library"
        )


    def load_model(
        self,
        model_name="bayesianridge"
    ):

        model_path = (
            self.model_root
            /
            "weights"
            /
            "benchmark_sklearn"
            /
            f"{model_name}.joblib"
        )


        return joblib.load(model_path)



    def predict(
        self,
        model_name,
        X
    ):

        model = self.load_model(
            model_name
        )

        X=np.asarray(X)


        y=model.predict(X)


        return {

            "model":model_name,

            "prediction":
            y.tolist(),

            "shape":
            list(X.shape),

            "status":
            "success"

        }