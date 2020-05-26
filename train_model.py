import tensorflow as tf
import numpy as np
import model


batch_size = 256
hist_size = 30
data_dict = {}
feed_dict = {}
batch_idx = 0
feature_size = 1048573
epoch = 25

def data_set(data_dict, feature, string):
        if string not in data_dict:
             data_dict[string] =[[feature]]
        else:
             if(len(data_dict[string]) < batch_idx + 1):
                 data_dict[string].append([feature])
             else:
                 data_dict[string][batch_idx].append(feature)

def input_data_set(data_dict, features, prefix=""):
    for feature in features:
        feature = feature.split(":")
        feature = int(feature[0])
        group_id = feature >> 48
        feature = feature % feature_size 
        data_set(data_dict, feature, prefix+str(group_id))

def input_hist_data_set(data_dict, hist_features, hist_group_ids, pos_group_ids, hist_size, prefix=""):
    hist_len = len(hist_features)
    if hist_features[0] == '\n' or hist_features[0] == '' or hist_features[0] == ' ':
          hist_len = 0
    for i in range(0, hist_size):
        if i < hist_len:
            features = hist_features[i].split()
            for feature in features:
                 feature = feature.split(":")
                 feature = int(feature[0])
                 group_id = feature >> 48
                 feature = feature % feature_size
                 if group_id in pos_group_ids:
                       data_set(data_dict, feature, prefix+"position_"+str(i)+"_"+str(group_id))
                 else:
                       data_set(data_dict, feature, prefix+str(i)+"_"+str(group_id))
        else:
            for group_id in hist_group_ids:
                 data_set(data_dict, 0, prefix+str(i)+"_"+str(group_id))
            for group_id in pos_group_ids:
                 data_set(data_dict, 0, prefix+"position_"+str(i)+"_"+str(group_id))
             
    if prefix+"histLen" not in data_dict:
            data_dict[prefix+"histLen"] = [hist_len]
    else:
            data_dict[prefix+"histLen"].append(hist_len)

def feed_dict_sparse_feature(model, feed_dict, data_dict, string):
    index, value = [], []
    for i in range(batch_size):
           for k in range(len(data_dict[string][i])):
                index.append(np.array([i, k], dtype=np.int64))
                value.append(data_dict[string][i][k])
    iv = tf.SparseTensorValue(index, value, [len(data_dict[string]), feature_size])
    feed_dict[model._group_feature[string]] = iv

def feed_dict_process(model, data_dict, feed_dict, main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids):
    for group_id in main_group_ids:
            data_name = "main_" + str(group_id)
            feed_dict_sparse_feature(model, feed_dict, data_dict, data_name)
    for group_id in candidate_group_ids:
            data_name = "candidate_" + str(group_id)
            feed_dict_sparse_feature(model, feed_dict, data_dict, data_name)
    for i in range(hist_size):
            for group_id in clicked_group_ids:
               data_name = "clicked_" + str(i) + "_" + str(group_id)
               feed_dict_sparse_feature(model, feed_dict, data_dict, data_name) 
            for group_id in unclick_group_ids:
               data_name = "unclick_" + str(i) + "_" + str(group_id)
               feed_dict_sparse_feature(model, feed_dict, data_dict, data_name) 
            for group_id in feedback_group_ids:
               data_name = "feedback_" + str(i) + "_" + str(group_id)
               feed_dict_sparse_feature(model, feed_dict, data_dict, data_name)
            for group_id in pos_group_ids:   
               data_name = "clicked_position_" + str(i) + "_" + str(group_id)
               feed_dict_sparse_feature(model, feed_dict, data_dict, data_name)
               data_name = "unclick_position_" + str(i) + "_" + str(group_id)
               feed_dict_sparse_feature(model, feed_dict, data_dict, data_name)
               data_name = "feedback_position_" + str(i) + "_" + str(group_id)
               feed_dict_sparse_feature(model, feed_dict, data_dict, data_name)
    feed_dict[model._group_feature["clicked_histLen"]] = data_dict["clicked_histLen"]
    feed_dict[model._group_feature["unclick_histLen"]] = data_dict["unclick_histLen"]
    feed_dict[model._group_feature["feedback_histLen"]] = data_dict["feedback_histLen"]

def train_data_process(sess, model, data, is_train, main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids):
    global data_dict, batch_idx, feed_dict 
    data = data.split('\t')
    label = float(data[0])
    weight = float(data[1])
    features = data[2].split('|')
    main_features = features[0].split()
    candidate_features = features[1].split()
    clicked_features = features[2].split(';')
    unclick_features = features[3].split(';')
    feedback_features = features[4].split(';')
    if "label" not in data_dict:
        data_dict["label"] = [label]
    else:
        data_dict["label"].append(label)
    
    if "weight" not in data_dict:
        data_dict["weight"] = [weight]
    else:
        data_dict["weight"].append(weight)
    
    input_data_set(data_dict, main_features, "main_")
    input_data_set(data_dict, candidate_features, "candidate_")
    input_hist_data_set(data_dict, clicked_features, clicked_group_ids, pos_group_ids, hist_size, "clicked_")
    input_hist_data_set(data_dict, unclick_features, unclick_group_ids, pos_group_ids, hist_size, "unclick_")
    input_hist_data_set(data_dict, feedback_features, feedback_group_ids, pos_group_ids, hist_size, "feedback_")
    if batch_idx < batch_size -1: 
        batch_idx += 1
        return
    feed_dict_process(model, data_dict, feed_dict, main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids)
    feed_dict[model.labels] = data_dict["label"]
    feed_dict[model.weights] = data_dict["weight"]
    _, loss=sess.run([model.train_op,model.loss], feed_dict=feed_dict)
    print loss 
    batch_idx = 0
    data_dict = {}   
    feed_dict = {}    



def main():    
    main_group_ids=[16,10001,10002,10003,21,10006,10019,10034,20147,20148,10035,20156,61,10047,10048,10049,10050,10055,10056,60]
    candidate_group_ids=[3060,3061,3062,3063,3064]
    clicked_group_ids=[3060,3061,3062,3063,3064]
    unclick_group_ids=[3060,3061,3062,3063,3064]
    feedback_group_ids=[3060,3061,3063,3064]
    pos_group_ids=[3065]
    sess = tf.Session()
    model_net = model.DFN(main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in xrange(epoch):
     f = open("example")
     line = f.readline()
     while line:
        train_data_process(sess, model_net, line, 1, main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids)
        line = f.readline()
     f.close()

      

if __name__ == '__main__':
      main()
