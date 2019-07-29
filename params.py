import os
jobs=os.cpu_count()
embedding_dims = 25

max_features = 10000
full_txt_maxlen = 500
context_txt_maxlen = 120
surround = 15 # try 15,25
mask_spread = 2 # try 1,3,4
mask_std = 1 # try 0.5,1.5
min_count = 5 # try 3,4
vocab_size = max_features
word_length_filter = 3

fasttext_dims=200

cutout_proba = 0.8
# cutout_proba = 1.0
max_cutout = 45
min_cutout = 15
num_cuts = 4

def lr_decay(epoch,prev_lr):
    new_lr = 0
    if epoch==0:
        new_lr = prev_lr/20
    elif epoch==1:
        new_lr = prev_lr * 4
    elif epoch==2:
        new_lr = prev_lr * 5
    else:
        new_lr = 0.9 * prev_lr
    print("Epoch = %s, Prev LR = %.4f, New LR = %.4f"%((epoch+1),prev_lr,new_lr))
    return new_lr

conv_embedded_params = dict(full_text_conv_layer_width = 8, context_conv_layer_width = 6, 
                            lstm_full_text_units = 16, lstm_context_units = 8,
                            fc_layer_width = 96, fc_layer_depth = 2,
                           training_policy=dict(epochs=20,batch_size = 64, policy="adam",lr=0.003,max_lr_olr=0.1, lr_shed_fn=lr_decay))


conv_non_embedded_params = dict(full_text_conv_layer_width = 6, context_conv_layer_width = 4, 
                            fc_layer_width = 80, fc_layer_depth = 2,
                            training_policy=dict(epochs=10,batch_size = 16,policy="adam",lr=0.005,max_lr_olr=0.002, lr_shed_fn=lambda e,lr:0.9*lr),)










