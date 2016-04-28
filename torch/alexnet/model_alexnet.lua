-- This source code is from Soumith Chintala's imagenet-multiGPU.torch code
--   https://github.com/soumith/imagenet-multiGPU.torch
-- This source code is licensed under the BSD 2-clause license found in the
-- 3rd-party-licenses.txt file in the root directory of this source tree.

-- The original code was modified by Robert Bosch LLC, USA to time AlexNet
-- All modifications are licensed under the MIT license found in the
-- LICENSE file in the root directory of this source tree.


function createModel(nGPU, backend)

   assert(nGPU == 1 or nGPU == 0, '1-GPU or CPU supported for AlexNet')
   local features
   local nClasses = 1000
   
   if (nGPU == 1) then
      require 'cudnn'
   end
   
   if (nGPU == 1) and (backend == 'fbcunn') then
      require 'fbcunn'
   end
   
   features = nn.Concat(2)
   
   local fb1 = nn.Sequential() -- branch 1
   
   if (nGPU == 1) and (backend == 'cudnn') then
       fb1:add(cudnn.SpatialConvolution(3,48,11,11,4,4,2,2))       -- 224 -> 55
       fb1:add(cudnn.ReLU(true))
       fb1:add(cudnn.SpatialCrossMapLRN(5,0.0001,0.75,1.0))
       fb1:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
       fb1:add(cudnn.SpatialConvolution(48,128,5,5,1,1,2,2,2))       --  27 -> 27     groups=2
       fb1:add(cudnn.ReLU(true))
       fb1:add(cudnn.SpatialCrossMapLRN(5,0.0001,0.75,1.0))
       fb1:add(cudnn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
       fb1:add(cudnn.SpatialConvolution(128,192,3,3,1,1,1,1))      --  13 ->  13
       fb1:add(cudnn.ReLU(true))
       fb1:add(cudnn.SpatialConvolution(192,192,3,3,1,1,1,1,2))      --  13 ->  13    groups=2
       fb1:add(cudnn.ReLU(true))
       fb1:add(cudnn.SpatialConvolution(192,128,3,3,1,1,1,1,2))      --  13 ->  13    groups=2
       fb1:add(cudnn.ReLU(true))
       fb1:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6        
       
       fb2 = fb1:clone() -- branch 2
       for k,v in ipairs(fb2:findModules('cudnn.SpatialConvolution')) do
           v:reset() -- reset branch 2's weights
       end
   
   elseif (nGPU == 1) and (backend == 'fbcunn') then
       fb1 = nn.Sequential() -- branch 1
       fb1:add(cudnn.SpatialConvolution(3,48,11,11,4,4,2,2))       -- 224 -> 55
       fb1:add(cudnn.ReLU(true))
       fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
       fb1:add(nn.SpatialZeroPadding(2,2,2,2))
       fb1:add(nn.SpatialConvolutionCuFFT(48,128,5,5,1,1))       --  27 -> 27
       fb1:add(cudnn.ReLU(true))
       fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
       fb1:add(nn.SpatialZeroPadding(1,1,1,1))
       fb1:add(nn.SpatialConvolutionCuFFT(128,192,3,3,1,1))      --  13 ->  13
       fb1:add(cudnn.ReLU(true))
       fb1:add(nn.SpatialZeroPadding(1,1,1,1))
       fb1:add(nn.SpatialConvolutionCuFFT(192,192,3,3,1,1))      --  13 ->  13
       fb1:add(cudnn.ReLU(true))
       fb1:add(nn.SpatialZeroPadding(1,1,1,1))
       fb1:add(nn.SpatialConvolutionCuFFT(192,128,3,3,1,1))      --  13 ->  13
       fb1:add(cudnn.ReLU(true))
       fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

       fb2 = fb1:clone() -- branch 2
       for k,v in ipairs(fb2:findModules('nn.SpatialConvolutionCuFFT')) do
          v:reset() -- reset branch 2's weights
       end
       for k,v in ipairs(fb2:findModules('cudnn.SpatialConvolution')) do
          v:reset() -- reset branch 2's weights
       end

   elseif (nGPU == 0) then
       fb1:add(nn.SpatialConvolution(3,48,11,11,4,4,2,2))       -- 224 -> 55
       fb1:add(nn.ReLU(true))
       fb1:add(nn.SpatialCrossMapLRN(5,0.0001,0.75,1.0))
       fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
       fb1:add(nn.SpatialConvolution(48,128,5,5,1,1,2,2,2))       --  27 -> 27     groups=2
       fb1:add(nn.ReLU(true))
       fb1:add(nn.SpatialCrossMapLRN(5,0.0001,0.75,1.0))
       fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
       fb1:add(nn.SpatialConvolution(128,192,3,3,1,1,1,1))      --  13 ->  13
       fb1:add(nn.ReLU(true))
       fb1:add(nn.SpatialConvolution(192,192,3,3,1,1,1,1,2))      --  13 ->  13    groups=2
       fb1:add(nn.ReLU(true))
       fb1:add(nn.SpatialConvolution(192,128,3,3,1,1,1,1,2))      --  13 ->  13
       fb1:add(nn.ReLU(true))
       fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

       fb2 = fb1:clone() -- branch 2
       for k,v in ipairs(fb2:findModules('nn.SpatialConvolution')) do
           v:reset() -- reset branch 2's weights
       end
       
   end
   
    
   features:add(fb1)
   features:add(fb2)

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, nClasses))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)

   return model
end
