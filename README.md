# StockSentimentAnalysis
Using the news headlines about a company, we are predicting if its stock price will rise or fall using NLP.

##Steps Followed:

The initial lines that are commentes are using pandas, to check the distribution of data year wise, So as to split train and test data accordingly.

0.Train and test datasets are divided based on date.

1. Data Cleaning
    Train data is obtained by iloc[:,cols-of-texts]
    Data that is non alphabetical [^a-zA-Z] are replaced, regex=True
    
    --Column names are renamed to make life easy
    Column wise, the texts are converted to lowercase
    Row wise, the individual news headlines are concatenated to create a paragraph -> then appended to list
    
2. Vectorization
    The corpus list is passed to the CountVectoizer to get BOW - ngram_range=(2,2)  ... BOW_model.fit_transform(..)
   
3. Model
    RandomForest ensemble model is trained here with our vectors (FIT)
    
    The similar preprocessing is done on test dataset and is vectorized(BOW.transform(..) only)
    
    predection is obtained by rf_model.predict(X_test)
    
4. Metrics
    The predictions are then passed through confusion_matrix, accuracy_score and classification_report
    confusion_matrix(y_test,predictions)
    
    The scores are obtained for Review !
    
