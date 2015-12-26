

@doc """ Give an Array of pics file paths, return data
""" ->
function read_pics(fls::Array{UTF8String,1})
    imgs =  pmap(load, fls)
    dt = @parallel (vcat) for img in imgs
        data(reinterpret(UInt8, img))[:]'
    end
    
    convert(Array{Float32,2},dt)
end

@doc """ get train data
""" ->
function train_data()
    pics = readdir(trainresize_dir)
    pics_path = map(x->joinpath(trainresize_dir,x), pics)

    pics = map(y->convert(ASCIIString,y), map(x->x[1:end-4], pics))

    sample_label_dict = JLD.load(sample_label_fl,"sample_label_dict")
    labels = map(x->sample_label_dict[x], pics)
    read_pics(pics_path),labels
end

@doc """ get test data
""" ->
function test_data()
    pics = readdir(testresize_dir)
    pics_path = map(x->joinpath(testresize_dir,x), pics)

    pics,read_pics(pics_path)
end

@doc """ return sample_names=>labels dict
""" ->
function readlabel()
    dt = readcsv(train_lb,ASCIIString)
    dt =  dt[2:end,:]
    classes = sort(unique(dt[:,2]))
    sample_names = dt[:,1]
    num_sample = length(sample_names)
    labels = Array{Int64,1}(num_sample)
    for i=1:num_sample
        lbs = searchsorted(classes,dt[i,2])
        @assert length(lbs) == 1
        labels[i] = lbs[1]-1
    end
    sample_label_dict = Dict(zip(sample_names,labels))
    JLD.save(sample_label_fl,"sample_label_dict",sample_label_dict)

    label_sample_dict = Dict(map(reverse, collect(sample_label_dict)))
    JLD.save(label_sample_fl, "label_sample_dict", label_sample_dict)

    nothing
end
