require "hdf5"
require "nn"
require "cunn"

local batch_size = 256

-- Model
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 64, 5, 5, 1, 1, 2, 2)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, 2, 2)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialConvolution(128, 128, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(128*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(128*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(84, 70))                   -- 62 classes
net:add(nn.LogSoftMax())

-- Criterion
criterion = nn.ClassNLLCriterion()

-- Load data
f = hdf5.open("data.hdf5")
XTr = f:read("XTr"):all()
yTr = f:read("yTr"):all()
f:close()
trainset = {data = XTr, label = yTr}
setmetatable(trainset, 
{ __index = function(t, i) 
     return {t.data[i], t.label[i]}
end
}
);

function trainset:size()
    return self.data:size(1)
end
print("yTr min max")
print(torch.min(yTr))
print(torch.max(yTr))


-- Optimization
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 5 -- just do 5 epochs of training.
trainer:train(trainset)

--[[
for i = 1,2500 do
  -- random sample
  local input= XTr;     -- normally distributed example in 2d
  local output= yTr;

  -- feed it to the neural network and the criterion
  criterion:forward(net:forward(input), output)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  net:zeroGradParameters()
  -- (2) accumulate gradients
  net:backward(input, criterion:backward(net.output, output))
  -- (3) update parameters with a 0.01 learning rate
  net:updateParameters(0.01)
end

--]]
