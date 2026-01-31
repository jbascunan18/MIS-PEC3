import argparse


class InputParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="TODO")
        self.args = None

    def parse_args(self):
        self.parser.add_argument(
            "-m",
            "--model",
            nargs="+",
            choices=["svm", "svm_ext", "zs_transformers", "ft_transformers", "all"],
            help="Select the pipeline to execute:\n\t1. SVM model\n\t2. SVM model with extended features (lexicon scores)\n\t3. Zero-shot transformers: run inference on pretrained model\n\t4. Fine-tune transformers: fine-tune a pretrained model with custom data\n\t5. all. Run all the models in cascade",
            required=True,
        )

        self.args = self.parser.parse_args()
        return self.args
