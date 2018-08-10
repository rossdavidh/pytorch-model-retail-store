# pytorch-model-retail-store
neural network model of a small retail clothing store

To use, at the command line, issue the following in order:

python a_load_dataframe.py scaled_sales.csv sales_dataframe.p

   ...this will ingest the data in "scaled_sales.csv", and convert it to the pickled dataframes "pre_cutoff_sales_dataframe.p"
   and "post_cutoff_sales_dataframe.p"

python b_train_network.py pre_cutoff_sales_dataframe.p sales_prediction_network.p 10000

...this will create a neural network and train it using the data in pre_cutoff_sales_dataframe.p, up to a maximum of 10,000
epochs, and save the resulting network in the pickle file "sales_prediction_network.p"

python c_predict_future_results.py post_cutoff_sales_dataframe.p sales_prediction_network.p output_filename.csv

...this will take the trained neural network and use it to make predictions about the dates in post_cutoff_sales_dataframe.p,
   and calculate and print out the R-squared of the result.
