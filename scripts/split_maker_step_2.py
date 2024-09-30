import polars as pl
# Load the combined dataset
combined_data = pl.read_parquet("dataset/train_folds.parquet")

# Count the values in the 'full_test' column
train_data = combined_data.filter(pl.col("bb_test_noshare") == False)
val_data = combined_data.filter(pl.col("bb_test_noshare") == True)

# Save the filtered datasets to Parquet files
train_data.write_parquet("dataset/train_folds_subsampled_train_bb.parquet")
val_data.write_parquet("dataset/train_folds_subsampled_val_bb.parquet")