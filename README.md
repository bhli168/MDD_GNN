# MDD_GNN
The code is for paper 'Heterogeneous Graph Neural Network for Power Allocation in Multicarrier-Division Duplex Cell-Free Massive MIMO Systems' .
This is an original version, which will be improved in a more concise way in near future.

The implementation of code requires Pytorch Geometric and MATLAB-based CVX tool. The overall codes are divided into three parts for Section IV-B, -C and -D, respectively.

In general, to run the codes:

1. Generate the networks using 'MDD_ML_Produce' or 'MDD_ML_Produce_Cluster'

2. Obtain the results based on QT-SCA using 'MDD_ML_SE_main' or 'MDD_ML_SE_main_Cluster'

3. Obtain the results based on CF-HGNN using 'MDD_Newmain_XXXX'

Here are some tips:

1. Generally, the layer can be increased so as to achieve better performance but at the expense of high complexity.

2. According to our recent simulations, whether adding power budget and interference capabilities in node features or not have nearly the same influence on model performance. Hence, for the ease of model implementation, we would like to only use equivalent channel gains as node features.

3. The configuration of PA MLP shown in Table I is a default one. In fact, it acted as output layer can be adjusted for different cases.

4. In Section IV-C, the traning of AP/MS and subcarriers is seperated for the consideration of efficient training and texting. However, they can be integrated together with an example of dataset concatenation shown in the codes.
