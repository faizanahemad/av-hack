import os
jobs=os.cpu_count()
embedding_dims = 25

max_features = 12500
full_txt_maxlen = 500 # try 750,1250
context_txt_maxlen = 120 # try 150,200
surround = 15 # try 15,25
mask_spread = 2 # try 1,3,4
mask_std = 1 # try 0.5,1.5
min_count = 4 # try 3,4
vocab_size = max_features
word_length_filter = 2

fasttext_dims=300

cutout_proba = 0.75
# cutout_proba = 1.0
max_cutout = 8
min_cutout = 3

conv_embedded_params = dict(full_text_conv_layer_width = 8, context_conv_layer_width = 6, 
                            fc_layer_width = 80, fc_layer_depth = 2,
                           training_policy=dict(epochs=20,batch_size = 64, policy="adam",lr=0.002,max_lr_olr=0.005, lr_shed_fn=lambda e,lr:0.9*lr))

conv_embedded_params_m1v1 = dict(context_conv_layer_width = 8, 
                            fc_layer_width = 80, fc_layer_depth = 2,
                           training_policy=dict(epochs=20,batch_size = 64, policy="adam",lr=0.002,max_lr_olr=0.005, lr_shed_fn=lambda e,lr:0.9*lr))


conv_non_embedded_params = dict(full_text_conv_layer_width = 6, context_conv_layer_width = 4, 
                            fc_layer_width = 80, fc_layer_depth = 2,
                            training_policy=dict(epochs=10,batch_size = 32,policy="adam",lr=0.005,max_lr_olr=0.005, lr_shed_fn=lambda e,lr:0.95*lr),)

lstm_embedded_params = dict(lstm_full_text_units = 64, lstm_context_units = 32,
                            fc_layer_width = 32, fc_layer_depth = 2,
                           training_policy=dict(epochs=20,batch_size = 64,policy="adam",lr=0.005,max_lr_olr=0.005, lr_shed_fn=lambda e,lr:0.95*lr))

lstm_non_embedded_params = dict(lstm_full_text_units = 64, lstm_context_units = 32,
                            fc_layer_width = 32, fc_layer_depth = 2,
                            training_policy=dict(epochs=5,batch_size = 32,policy="adam",lr=0.005,max_lr_olr=0.005, lr_shed_fn=lambda e,lr:0.95*lr),)









