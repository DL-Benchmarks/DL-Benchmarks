-- Copyright (c) 2016 Robert Bosch LLC, USA.
-- All rights reserved.
--
-- This source code is licensed under the MIT license found in the
-- LICENSE file in the root directory of this source tree.

-- Timing of Stacked Auto-encoder

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
    conf.num_threads = 6
    conf.hidden1 = 800
    conf.hidden2 = 1000
    conf.hidden3 = 2000
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


function createAE(hidden0, hidden1, mode)
    
    encoder = nn.Sequential()
    decoder = nn.Sequential()
    autoencoder = nn.Sequential()
    
    encoder:add(nn.Linear(hidden0, hidden1))
    decoder:add(nn.Linear(hidden1, hidden0))
    
    if (mode == 'GPU') then
        encoder:add(cudnn.Sigmoid())
        decoder:add(cudnn.Sigmoid())
    else
        encoder:add(nn.Sigmoid())
        decoder:add(nn.Sigmoid())
    end

    -- constraining weights so W=W^T
    decoder:get(1).weight = encoder:get(1).weight:t()
    decoder:get(1).gradWeight = encoder:get(1).gradWeight:t()

    if (mode == 'GPU') then
        encoder:cuda()
        decoder:cuda()
    end
    
    autoencoder:add(encoder)
    autoencoder:add(decoder)
    
    if (mode == 'GPU') then
        autoencoder:cuda()
    end
    
    return encoder, decoder, autoencoder
end


function timingPretraining(autoencoder, target_function, inputs, criterion, conf)
    
    forward_time = torch.zeros(conf.num_measures)
    backward_time = torch.zeros(conf.num_measures)
    
    for step=1,(conf.num_measures + conf.num_dry_runs) do

    if (conf.mode == 'GPU') then 
        cutorch.synchronize()
    end

	if (step > conf.num_dry_runs) then
        sys.tic()
        targets = target_function:forward(inputs)
        autoencoder:forward(targets)
        if (conf.mode == 'GPU') then 
            cutorch.synchronize()
        end
        forward_time[step-conf.num_dry_runs] = sys.toc()
		
		criterion:forward(autoencoder.output, targets)
		
        sys.tic()
        -- Calculate gradients w.r.t the different layers using a backpropagation step
        autoencoder:backward(targets, criterion:backward(autoencoder.output, targets))
        if (conf.mode == 'GPU') then 
            cutorch.synchronize()
        end
        backward_time[step-conf.num_dry_runs] = sys.toc()
        
    else
        -- Dry runs
        targets = target_function:forward(inputs)
        criterion:forward(autoencoder:forward(targets), targets)
        autoencoder:backward(targets, criterion:backward(autoencoder.output, targets))

        if (conf.mode == 'GPU') then 
            cutorch.synchronize()
        end

    end -- dry run switch
end -- for loop

return forward_time, backward_time

end


function timingFinetuning(model, inputs, targets, criterion, conf)
    
    forward_time = torch.zeros(conf.num_measures)
    backward_time = torch.zeros(conf.num_measures)
    
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

return forward_time, backward_time

end


-- Generate input image and label batch
local num_classes = 10
local inputs = torch.rand(conf.batch_size,conf.image_width*conf.image_width)
local targets = torch.ceil(torch.rand(conf.batch_size)*num_classes)


-- Define the model for both CPU and GPU modes for our benchmarking
model_finetuning = nn.Sequential()

encoder1, decoder1, autoencoder1 = createAE(conf.image_width*conf.image_width, conf.hidden1, conf.mode)
encoder2, decoder2, autoencoder2 = createAE(conf.hidden1, conf.hidden2, conf.mode)
encoder3, decoder3, autoencoder3 = createAE(conf.hidden2, conf.hidden3, conf.mode)

model_finetuning:add(encoder1)
model_finetuning:add(encoder2)
model_finetuning:add(encoder3)
model_finetuning:add(nn.Linear(conf.hidden3, num_classes))
model_finetuning:add(nn.SoftMax())


-- Choose loss functions for the output during pre-training and finetuning stages
criterion_autoencoder = nn.MSECriterion()
criterion_finetuning = nn.ClassNLLCriterion()

-- Move the model and data to GPU for the 'GPU' mode
if (conf.mode == 'GPU') then
   cutorch.synchronize()
   model_finetuning = model_finetuning:cuda()
   
   criterion_autoencoder = criterion_autoencoder:cuda()
   criterion_finetuning = criterion_finetuning:cuda()
   inputs = inputs:cuda() 
   targets = targets:cuda()
end

intermediate_inputs = nn.Sequential():add(nn.Identity())
forward_time_AE1, backward_time_AE1 = timingPretraining(autoencoder1, intermediate_inputs, inputs, criterion_autoencoder, conf)
forward_time_AE2, backward_time_AE2 = timingPretraining(autoencoder2, intermediate_inputs:add(encoder1), inputs, criterion_autoencoder, conf)
forward_time_AE3, backward_time_AE3 = timingPretraining(autoencoder3, intermediate_inputs:add(encoder2), inputs, criterion_autoencoder, conf)


forward_time_FT, backward_time_FT = timingFinetuning(model_finetuning, inputs, targets, criterion_finetuning, conf)


print('Printing time taken for forward and backward model execution')
print('All times are for one pass with one batch')
print('Mode: ', conf.mode)
print('Tot. number of batch runs: ', conf.num_measures)
print('Number of dry runs: ', conf.num_dry_runs)
print('Batch size: ', conf.batch_size)

print('Timing Results for AE1')
print(string.format('forward: %.5f +/- %.5f ms per batch', 1000*torch.mean(forward_time_AE1), 1000*torch.std(forward_time_AE1)))
print(string.format('gradient computation: %.5f +/- %.5f ms per batch', 1000*torch.mean(backward_time_AE1 + forward_time_AE1), 1000*torch.std(backward_time_AE1 + forward_time_AE1)))

print('Timing Results for AE2')
print(string.format('forward: %.5f +/- %.5f ms per batch', 1000*torch.mean(forward_time_AE2), 1000*torch.std(forward_time_AE2)))
print(string.format('gradient computation: %.5f +/- %.5f ms per batch', 1000*torch.mean(backward_time_AE2 + forward_time_AE2), 1000*torch.std(backward_time_AE2 + forward_time_AE2)))

print('Timing Results for AE3')
print(string.format('forward: %.5f +/- %.5f ms per batch', 1000*torch.mean(forward_time_AE3), 1000*torch.std(forward_time_AE3)))
print(string.format('gradient computation: %.5f +/- %.5f ms per batch', 1000*torch.mean(backward_time_AE3 + forward_time_AE3), 1000*torch.std(backward_time_AE3 + forward_time_AE3)))

print('Timing Results for FineTuning Stage')
print(string.format('forward: %.5f +/- %.5f ms per batch', 1000*torch.mean(forward_time_FT), 1000*torch.std(forward_time_FT)))
print(string.format('gradient computation: %.5f +/- %.5f ms per batch', 1000*torch.mean(backward_time_FT + forward_time_FT), 1000*torch.std(backward_time_FT + forward_time_FT)))
