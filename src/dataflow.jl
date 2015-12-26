
@doc """ data flow of whole project
""" ->
function dataflow()

    # make data
    if debug
        #get dict sample_label label_sample
        @info("processing label")
        @time readlabel()
        
        @info("processing training data")
        @time X,lbs = train_data()
        
        @info("shuffle and split data")
        data = hcat(X, lbs)
        data = shuffle(data)
        @time train,val,test = split_data(data)
        JLD.save(train_fl, "train", train)
        JLD.save(val_fl,   "val",   val)
        JLD.save(test_fl,  "test",  test)
    end
    train = JLD.load(train_fl, "train")
    val = JLD.load(val_fl, "val")
    test = JLD.load(test_fl, "test")
    # tain model
    @info("train data with XGBoost")
    model_xgboost(train,val,test,100_000)
    
end


@doc """ submission with the result model
""" ->
function submission()

end
