# Introduction
In most of the nlp usecases first thing we do is tokenization based on whitespaces to create a vocabulary. 
One day I had a thought what if we don't have spaces in the text itself ?
That question led me to train a model which will take input sentences without whitespaces and predicts a sentence with whitespaces, e.g.

| Input | Expected Output |
| ------------- | ----------- |
| thismodelputspacesbetweencharacters      | this model put spaces between characters|

# Architecture
We can categorize this challenge as a sequence modelling problem because of the following reasons,
1. We have varying input and output sequence length.
2. Order is important


Hence I have used a <b> Seq2Seq model with attention </b> to tackle this challenge.


The genearal training steps to for seq2seq model are as follows,
1. Create Input Output pairs
2. Create a vocabulary
3. Encode Inputs and Outputs using vocabulary words
4. Define encoders and decoder layers based on the complexity of the problem
5. Define loss function
6. Train the model
7. Evaluate the results
8. Fine tune based on the results

# Challenge

The major challenge while creating training model was how to build vocabulary. The standard way to build Vocab is,

    1. Split the sentences into words using whitespaces.
    2. Preprocess the word (Convert word into its root form, etc.)
    3. Calculate word frequencies.
    4. Based on word frequencies remove rare words.
    5. Now create a list of all unique words, length of this list will be length of vocab which will be fed to the model.

In our case as our input sentence does not contains any white space we can't use the steps discussed above. So for this model we will use character based vocabulary, where list of all unique characters will be our vocab for the model. We have 33 as our vocab length, 26 lowercase characters, 6 punctuations and a whitespace.

Now as our vocab is built and ready to go, I have trained the model as mentioned in above training steps.

# Use of the Repo
you can open <b>char space rnn .ipynb</b> and see the code for all the steps mentioned above.
If you have any query feel free to contact me 
<a href="https://www.linkedin.com/in/swapnilskumbhar">here</a>




