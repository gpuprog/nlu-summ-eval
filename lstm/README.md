The evaluation framework for the ACL 2017 paper *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*.

## Datasets and prerequisites
1. Please read high-level README.md about datasets and prerequisites.
2. Download pretrained_model_tf1.2.1.zip and extract to current (lstm) folder (it will create subfodler "pretrained_model_tf1.2.1")
3. Make folder "pretrained_model_tf1.2.1/test" and link (or copy) "pretrained_model_tf1.2.1/eval" and "pretrained_model_tf1.2.1/train" subfolders to appropriate subfolders of folder "pretrained_model_tf1.2.1/test"
(in other words you should have copies or links under "pretrained_model_tf1.2.1/test/eval" and "pretrained_model_tf1.2.1/test/train")
4. In Visual Studio Code set Python path to "tf1" environment (Using Ctrl+Shift+P then "Python: Select Interpreter")

## How to work
Evaluation process of each dataset needs three stages:
1. Prepare a dataset (exclude CNN dataset - it already in prepared form from ../data-cnn) with data-preparation/make_datafiles.py -  - use launch.json: "Prepare Amazon" and "Prepare WikiHow"
2. Process dataet with pointer-generator/run_summarization.py - use launch.json: "Process CNN", "Process Amazon" and "Process WikiHow" and run the next point after each process:
3. Run "evaluateInFiles.ipynb" to process results (uncomment and run "Optional: Evaluate results directly" cell after each "Process" launch)
P.S. You can backup and/or process results - see our backups under "backups" folder (to backup just keep "pretrained_model_tf1.2.1/test/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-238410" folder)

## References and Copyrights
data-preparation - https://github.com/abisee/cnn-dailymail with minimal changes
pointer-generator - https://github.com/abisee/pointer-generator with minimal changes