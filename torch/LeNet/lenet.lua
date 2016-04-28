-- Copyright (c) 2016 Robert Bosch LLC, USA.
-- All rights reserved.
--
-- This source code is licensed under the MIT license found in the
-- LICENSE file in the root directory of this source tree.

-- Timing of LeNet

require 'sys'
require 'nn'
require 'torch'


-- Initialize and Configure variables for LeNet benchmarking

collectgarbage()

torch.setdefaulttensortype('torch.FloatTensor')

function config()
    local conf = {}
    conf.image_width = 28   
    conf.batch_size = 64
    conf.num_measures = 200
    conf.num_dry_runs = 200
    conf.mode = 'GPU'  -- 'CPU' or 'GPU'
    conf.num_threads = 12
    return conf
end

local conf = config()
assert(conf.mode == 'GPU' or conf.mode == 'CPU', 'Only GPU or CPU mode supported for LeNet')


-- Import required packages for 'GPU' mode or set number of threads for 'CPU' mode
if (conf.mode == 'GPU') then
    require 'cudnn'
    require 'cutorch'
    require 'cunn'
else
    torch.setnumthreads(conf.num_threads)
end


-- Define the model for both CPU and GPU modes for our benchmarking
model = nn.Sequential()

if (conf.mode == 'GPU') then
    model:add(cudnn.SpatialConvolution(1,20,5,5,1,1))
    model:add(cudnn.SpatialMaxPooling(2,2,2,2))
    model:add(cudnn.Tanh())
    model:add(cudnn.SpatialConvolution(20,50,5,5,1,1))
    model:add(cudnn.SpatialMaxPooling(2,2,2,2))
    model:add(cudnn.Tanh())
    model:add(nn.View(50*4*4))
    model:add(nn.Linear(50*4*4,500))
    model:add(cudnn.ReLU())
    model:add(nn.Linear(500,10))
    model:add(nn.LogSoftMax())
    
elseif (conf.mode == 'CPU') then
    model:add(nn.SpatialConvolution(1,20,5,5,1,1))
    model:add(nn.SpatialMaxPooling(2,2,2,2))
    model:add(nn.Tanh())
    model:add(nn.SpatialConvolution(20,50,5,5,1,1))
    model:add(nn.SpatialMaxPooling(2,2,2,2))
    model:add(nn.Tanh())
    model:add(nn.View(50*4*4))
    model:add(nn.Linear(50*4*4,500))
    model:add(nn.ReLU())
    model:add(nn.Linear(500,10))
    model:add(nn.LogSoftMax())
end

-- Choose a loss function for the output
criterion = nn.ClassNLLCriterion()


-- Generate input image and label batch
local num_classes = 10
local inputs = torch.rand(conf.batch_size,1,conf.image_width,conf.image_width)
local targets = torch.ceil(torch.rand(conf.batch_size)*num_classes)

-- Initialize variables to gather forward and backward time
forward_time = torch.zeros(conf.num_measures)
backward_time = torch.zeros(conf.num_measures)


-- Move the model and data to GPU for the 'GPU' mode
if (conf.mode == 'GPU') then
   cutorch.synchronize()
   model = model:cuda()
   criterion = criterion:cuda()
   inputs = inputs:cuda() 
   targets = targets:cuda()
end


-- Loop through forward and backward passes for measuring time
for step=1,(conf.num_measures + conf.num_dry_runs) do
    
	if (conf.mode == 'GPU') then 
		cutorch.synchronize()
	end
	
	if (step > conf.num_dry_runs) then
        sys.tic()
        model:forward(inputs)
        if (conf.mode == 'GPU') then 
            cutorch.synchronize()
        end
        forward_time[step-conf.num_dry_runs] = sys.toc()
		
		criterion:forward(model.output, targets)
		
        sys.tic()
        -- Calculate gradients w.r.t the different layers using a backpropagation step
        model:backward(inputs, criterion:backward(model.output, targets))
        if (conf.mode == 'GPU') then
            cutorch.synchronize()
        end
        backward_time[step-conf.num_dry_runs] = sys.toc()
        
    else
        -- Dry runs
        criterion:forward(model:forward(inputs), targets)
        model:backward(inputs, criterion:backward(model.output, targets))
		
		if (conf.mode == 'GPU') then 
			cutorch.synchronize()
		end
    end -- dry run switch
end -- for loop


-- Print mean and standard deviation of forward and backward times
print('Printing time taken for forward and backward model execution')
print('All times are for one pass with one batch')
print('Mode: ', conf.mode)
print('Tot. number of batch runs: ', conf.num_measures)
print('Number of dry runs: ', conf.num_dry_runs)
print('Batch size: ', conf.batch_size)
print(string.format('forward: %.5f +/- %.5f sec per batch', torch.mean(forward_time), torch.std(forward_time)))
print(string.format('gradient computation: %.5f +/- %.5f sec per batch', torch.mean(backward_time + forward_time), torch.std(backward_time + forward_time)))
