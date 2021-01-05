# Lifelong Event Detection with Knowledge Transfer

We are unable to include entire datasets of MAVEN and ACE due to large size of data. We show examples of processed data in the data folded. For the subsets that forms incremental tasks, please refer to `data/*/streams.json` for partitions (integers to label names is in `data/*/label2id.json`), and `run_train.py: PERM` for order permutations. Since splits collection includes manual check to make sure all types are coveraged in development/test data, we don't include scripts for this process. 

For Silver Negative settings, instances are collected with `prepare_stream_instances.py`.

This is an initial version of code. More details will be given in the final public release.