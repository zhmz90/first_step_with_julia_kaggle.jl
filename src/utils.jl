
@doc """ print number of samples in  each cancer of a dataset
""" ->
function stat_class_sample(data,classes::Array{ASCIIString,1})
    #@pred("untested function numcancersample_data\n")
    max = maximum(data[:,end])
    min = minimum(data[:,end])

    output = Array{Pair{ASCIIString,Int64},1}()
    for i = min:max
        num = sum(data[:,end] .== i)
        if min == 0
            ind = convert(Int64,i+1)
        else
            ind = convert(Int64,i)
        end
        class = classes[ind]
        push!(output,Pair{ASCIIString,Int64}(class,num))
    end
    output = sort(output,by=x->x[2],rev=true)
    @info("--------------------------------------------------")
    @info("The number of samples in each class")
    @info("--------------------------------------------------")
    for i=1:length(output)
        num_sample = output[i][2]
        class     = output[i][1]
        @printf "%35s\t %4d\n" class num_sample
    end
    @info("--------------------------------------------------")
    nothing
end

@doc """
""" ->
function class_names()
    sample_label_dict = load(sample_label_fl,"sample_label_dict")
    sort(unique(collect(keys(sample_label_dict))))
end

@doc """eval ROC_value of model
"""->
function ROC_print(gt,pred;num_class=2,challenge_list=cancer_list)
    C = confusmat(num_class,gt,pred)
    println(C)

end
