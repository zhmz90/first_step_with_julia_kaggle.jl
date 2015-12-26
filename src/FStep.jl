module FStep

using Images
import JLD
using XGBoost
using MLBase
using Logging
@Logging.configure(level=DEBUG)

const debug = false

const data_dir = "../data"
const trainresize_dir = joinpath(data_dir, "trainResized")
const testresize_dir = joinpath(data_dir, "testResized")
const train_lb = joinpath(data_dir, "trainLabels.csv")
const sample_label_fl = joinpath(data_dir,"sample_label_dict.jld")
const label_sample_fl = joinpath(data_dir,"label_sample_dict.jld")

const train_fl = joinpath(data_dir, "train.jld")
const val_fl = joinpath(data_dir, "val.jld")
const test_fl = joinpath(data_dir, "test.jld")

include("preprocess.jl")
include("model.jl")
include("eval.jl")
include("dataflow.jl")
include("utils.jl")


end
