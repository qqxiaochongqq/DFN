# DFN
Deep Feedback Network for Recommendation (IJCAI-2020)

https://www.ijcai.org/Proceedings/2020/349

# Operating environment

python 2.7.15 
tensorflow 1.13

# Train DFN model
run train_model.py

# About The Train Data
the first column is the label, the second column is the weight of the sample, others are features. Features can be divided into five parts by "|". The first part is the main feature, include age, gender, device type, etc. The second part is the target item featue. The third part is the click sequence feature, and each item is separated by ";".The fourth part is the unclick sequence feature and the fifth is the dislike sequence feature.


# CITE

If the codes help you, please cite the following paper:

Ruobing Xie*, Cheng Ling*, Yalong Wang, Rui Wang, Feng Xia, Leyu Lin. Deep Feedback Network for Recommendation. IJCAI-2020 (* indicates equal contribution).
