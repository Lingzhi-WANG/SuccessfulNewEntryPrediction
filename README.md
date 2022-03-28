# SuccessfulNewEntryPrediction

Code release for TheWebConf(WWW) 2022 paper "Successful New-entry Prediction for Multi-Party Online Conversations via Latent Topics and Discourse Modeling".

Here we list some commands to help users to run the model.

<code>python train.py twitter LSTMTDM --use_pretrained_TDM PRETRAINED_MODEL_PATH --train_weight 0.27 --lr 0.0001 --cuda_dev 0</code>

<code>python train.py reddit LSTMTDM --use_pretrained_TDM PRETRAINED_MODEL_PATH --train_weight 4.69 --lr 0.001 --cuda_dev 1</code>

You can also use this code to run some baselines. For examples, you can run ConvKiller baseline by using command

<code>python train.py reddit ConvKiller --train_weight 4.69 --lr 0.0001 --cuda_dev 2</code>
