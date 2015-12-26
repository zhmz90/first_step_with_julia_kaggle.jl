@doc """
""" ->
function shuffle(data)
    m,n = size(data)
    inds = randperm(m)
    data[inds,:]
end

function split_class_data(data)
    m,n = size(data)
    point1 = round(Int64, 0.6*m)
    point2 = round(Int64, 0.8*m)
    train = data[1:point1,:]
    val   = data[point1+1:point2,:]
    test  = data[point2+1:end,:]

    train,val,test
end

@doc """ split each class dataset
""" ->
function split_data(data)
    num_label = maximum(data[:,end])

    function reducer(data1, data2)
        (vcat(data1[1],data2[1]), vcat(data1[2], data2[2]), vcat(data1[3], data2[3]))
    end
    
    data = @parallel (reducer) for i = 1:num_label
        split_class_data(data[data[:,end] .== i,:])
    end

    data
end

@doc """ eval accuracy of model
""" ->
function acc(preds,gts)
    @assert length(preds) == length(gts)
    sum(preds .== gts) / length(gts)
end

@doc """ xgboost for train,val,test dataset without gridtune
         eval with acc.
""" ->
function model_xgboost(train, val, test, num_iter::Int64)
    train_X,train_Y = train[:,1:end-1],vec(full(train[:,end]))
    val_X,  val_Y = val[:,1:end-1],vec(full(val[:,end]))
    test_X, test_Y = test[:,1:end-1],vec(full(test[:,end]))    

    num_class = maximum(train_Y)+1
    m_tr,n_tr   = size(train_X)
    m_val,n_val = size(val_X)    
    @info("in the model, there are $num_class classes")
    @info("train_X size is :$m_tr,$n_tr")
    @info("val_X size is : $m_val, $n_tr")
    #stat_class_sample(train,class_names())
    #sleep(3)
    
    if debug
        num_round = 1
    else
        num_round = num_iter
    end

    dtrain = DMatrix(train_X, label = train_Y)
    dval   = DMatrix(val_X,   label = val_Y)
    dtest  = DMatrix(test_X,  label = test_Y)   

    watch_list = [(dtrain,"train"), (dval,"val")]
    param      = Dict{ASCIIString,Any}("max_depth"=>2,"eta"=>0.01,"nthread"=>68,
                                       "objective"=>"multi:softmax","silent"=>1,
                                       "alpha"=>0.85,
                                       "sub_sample"=>0.6,"num_class"=>num_class)
   
    bst = xgboost(dtrain,num_round,watchlist=watch_list,param=param,metrics=["merror"],seed=2015)
    save(bst,"xgb_night.model")
    test_preds = XGBoost.predict(bst, test_X)    
    mean_f1 = acc(test_preds, get_info(dtest,"label"))
    @info("on test dataset mean-f1-score is $mean_f1")
    test_preds = map(x->convert(Int64,x), test_preds)
    test_Y = map(x->convert(Int64,x), test_Y)
    num_class = convert(Int64, num_class)

    ROC_print(test_Y.+1,test_preds.+1,num_class = num_class, challenge_list=class_names())

end
