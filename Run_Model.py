%%time
n_samples = 60000
batch = get_weighted_sampled_index(train_df, 
                                   class_cols,
                                   n_samples,
                                   classWeights_train,
                                   valueWeights_train)
train_df_60000 = train_df.loc[batch,:]
get_plot_histograms(train_df_60000, class_cols, f"{n_samples} Weighted Samples")

###
data_batch_df = train_df_60000.copy()
### Get class and value weights
print(f"New class weights for {n_samples} samples")
classWeights_train60000 = get_class_weights(train_df_60000, class_cols)
for weight,class_col in zip(classWeights_train60000, class_cols):
    print(f"{class_col}: {weight}")
    
print()

print(f"New value weights for {n_samples} samples")
valueWeights_train60000 = get_value_weights(train_df_60000, class_cols)
for weight,val in zip(valueWeights_train60000, [1,2]):
    print(f"{val}: {weight}")
#
train_df_60000[class_cols].describe()