-- Copyright (c) 2016 Robert Bosch LLC, USA.
-- All rights reserved.
--
-- This source code is licensed under the MIT license found in the
-- LICENSE file in the root directory of this source tree.

-- Timing of LSTM

require "rnn"
require "paths"

local stringx = require('pl.stringx')
local file = require('pl.file')


-- Initialize and Configure variables for LSTM benchmarking

collectgarbage()

torch.setdefaulttensortype('torch.FloatTensor')

function config()
    local conf = {}
    conf.data_path = "../../data/seqs_lengths.csv"
    conf.batch_size = 16
    conf.n_words = 10000
    conf.wordvec_dims = 128
    conf.maxlen = 100  -- maximum number of words in a sentence/review
    conf.num_measure_epochs = 5
    conf.num_dry_run_epochs = 1
    conf.mode = 'GPU'  -- 'CPU' or 'GPU'
    conf.num_threads = 12
    conf.learning_rate = 0.01
    return conf
end

local conf = config()

assert(conf.mode == 'GPU' or conf.mode == 'CPU', 'Only GPU or CPU mode supported for LSTM')


-- Import required packages for 'GPU' mode or set number of threads for 'CPU' mode

if (conf.mode == 'GPU') then
    require "cunn"
    require "cudnn"
    require "cutorch"
else
    torch.setnumthreads(conf.num_threads)
end


-- Define LSTM model along with the output layers
lstm = nn.Sequential()

if (conf.mode == 'GPU') then
    lstm = nn.Sequential()
    lstm:add(nn.LookupTableMaskZero(conf.n_words, conf.wordvec_dims, conf.batch_size))  -- convert indices to word vectors
    lstm:add(nn.SplitTable(1,2))  -- convert tensor to list of subtensors
    lstm:add(nn.Sequencer(nn.MaskZero(nn.FastLSTM(conf.wordvec_dims, conf.wordvec_dims), 2))) -- Seq to Seq', 0-Seq to 0-Seq
    lstm:add(nn.JoinTable(2)) -- stack list to tensor
    lstm:add(nn.View(conf.batch_size, -1, conf.wordvec_dims)) -- reshape tensor arbitrary y (max_length)
    lstm:add(nn.Copy(nil,nil,true)) -- this will force a copy, which will make it contiguous
    lstm:add(nn.Mean(2))  -- average over words
    lstm:add(nn.Linear(conf.wordvec_dims, 2)) -- bring to to classes
    lstm:add(cudnn.LogSoftMax())
else
    lstm = nn.Sequential()
    lstm:add(nn.LookupTableMaskZero(conf.n_words, conf.wordvec_dims, conf.batch_size))  -- convert indices to word vectors
    lstm:add(nn.SplitTable(1,2))  -- convert tensor to list of subtensors
    lstm:add(nn.Sequencer(nn.MaskZero(nn.FastLSTM(conf.wordvec_dims, conf.wordvec_dims), 2))) -- Seq to Seq', 0-Seq to 0-Seq
    lstm:add(nn.JoinTable(2)) -- stack list to tensor
    lstm:add(nn.View(conf.batch_size, -1, conf.wordvec_dims)) -- reshape tensor arbitrary y (max_length)
    lstm:add(nn.Copy(nil,nil,true)) -- this will force a copy, which will make it contiguous
    lstm:add(nn.Mean(2))  -- average over words
    lstm:add(nn.Linear(conf.wordvec_dims, 2)) -- bring to to classes
    lstm:add(nn.LogSoftMax())
end

criterion = nn.ClassNLLCriterion()


local data = file.read(conf.data_path)
data = torch.Tensor(stringx.split(data))
local total_num_batches = #data
total_num_batches = torch.floor(total_num_batches[1]/conf.batch_size)

if (conf.mode == 'GPU') then
   lstm:cuda()
   criterion:cuda()
end


forward_batch = torch.zeros(conf.num_measure_epochs)
backward_batch = torch.zeros(conf.num_measure_epochs)

for epoch = 1, (conf.num_dry_run_epochs + conf.num_measure_epochs) do
    
    local forward_epoch = 0.0
    local backward_epoch = 0.0

-- Loop through each batch to average the forward and backward times
    for step = 1, total_num_batches do

        -- copy the lengths of the sequences for each batch and find max length
        seq_lengths_batch = data[{{(step-1)*conf.batch_size+1, step*conf.batch_size}}]  --
        max_length = torch.max(seq_lengths_batch)
        if max_length > conf.maxlen then
            max_length = conf.maxlen
        end
        
        inputs = torch.zeros(conf.batch_size, max_length)
        for i=1, conf.batch_size do
            inputs[{{i},{max_length-seq_lengths_batch[i]+1, max_length}}] = torch.floor(torch.rand(seq_lengths_batch[i])*(conf.n_words-1))
        end
        labels = torch.ceil(torch.rand(conf.batch_size)*2) -- create labels of 1s and 2s
        
        if (conf.mode == 'GPU') then
           inputs = inputs:cuda()
           labels = labels:cuda()
		   cutorch.synchronize()
        end
        
        if (epoch <= conf.num_dry_run_epochs) then
            -- Dry runs
            lstm_output = lstm:forward(inputs)
            criterion:forward(lstm_output, labels)
            gradOut = criterion:backward(lstm_output, labels)
            lstm:backward(inputs, gradOut)
            
			if (conf.mode == 'GPU') then
                cutorch.synchronize()
            end
			
        else
            
            if (conf.mode == 'GPU') then
                cutorch.synchronize()
            end
			
            -- forward timing
            sys.tic()
            lstm_output = lstm:forward(inputs)
            if (conf.mode == 'GPU') then
                cutorch.synchronize()
            end
            forward_epoch = forward_epoch + sys.toc()

            criterion:forward(lstm_output, labels)
            
            -- backward timing
            sys.tic()
            lstm:backward(inputs, criterion:backward(lstm_output, labels))
            if (conf.mode == 'GPU') then
                cutorch.synchronize()
            end
            backward_epoch = backward_epoch + sys.toc()
            --print('step', step)
        end
        
        lstm:updateParameters(conf.learning_rate)
        if (conf.mode == 'GPU') then
            cutorch.synchronize()
        end
			
    end
    
    if (epoch > conf.num_dry_run_epochs) then
        forward_batch[epoch - conf.num_dry_run_epochs] = forward_epoch/total_num_batches
        backward_batch[epoch - conf.num_dry_run_epochs] = backward_epoch/total_num_batches
    end
    
end

print('measurements done')
print('mode', conf.mode)
if conf.mode == 'CPU' then
    print('number of threads', conf.number_of_threads)
end
print('batch size', conf.batch_size)


print(string.format('forward: %.5f +/- %.5f sec per batch', torch.mean(forward_batch), torch.std(forward_batch)))
print(string.format('gradient computation: %.5f +/- %.5f sec per batch', torch.mean(backward_batch + forward_batch), torch.std(backward_batch + forward_batch)))
