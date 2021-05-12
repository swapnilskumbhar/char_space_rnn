# char_space_rnn
In most of the nlp usecases first thing we do is tokenization based on whitespaces to create a vocabulary. 
One day I had a thought what if we don't have spaces in the text itself ?
That question led me to train a model which will take input sentences without whitespaces and predicts a sentence with whitespaces, e.g.

| Input | Expected Output |
| ------------- | ----------- |
| thismodelputspacesbetweencharacters      | this model put spaces between characters|

# Architecture
Seq2Seq
